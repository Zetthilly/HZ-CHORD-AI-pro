"""
Evaluates a trained NoteDetector checkpoint against GuitarSet — real guitar
recordings with genuine note-level ground truth (pitch, onset, offset per
string), openly licensed for research. This is the actual "compare against
real, already-correct arpeggios/notes" check, and it gates whether a new
checkpoint actually improved before anything gets committed.

GuitarSet is used purely as an EVALUATION set here — the model is trained
only on synthetic MIDI-rendered arpeggios (generate_synthetic_arpeggios.py),
never on GuitarSet itself, so there's no train/test leakage concern and the
full dataset is fair to evaluate against.

Requires: torch, torchaudio, soundfile, numpy, jams
    pip install torch torchaudio soundfile numpy jams

Usage:
    python3 evaluate_guitarset.py \
        --checkpoint checkpoints/note_detector.pt \
        --guitarset-root /path/to/guitarset \
        --best-score-path best_score.json \
        --max-files 20   # optional — evaluate a subset for speed

Expects --guitarset-root to contain, per GuitarSet's standard release layout:
    annotation/*.jams
    audio_mono-mic/*_mic.wav
(file-pairing below assumes that naming convention — verify it against
whatever you actually download, since dataset release layouts do shift
between versions.)

Writes:
    eval_results.json, gate_decision.txt — same pattern as
    stem_separator/evaluate_museval.py.
"""
import argparse
import glob
import json
import os

import jams
import numpy as np
import soundfile as sf
import torch
import torchaudio

from model import NoteDetector, N_MELS
from generate_synthetic_arpeggios import SAMPLE_RATE, NOTE_MIN, NOTE_MAX, NUM_NOTES
from dataset import HOP_LENGTH, N_FFT


def load_ground_truth_labels(jams_path: str, num_frames: int) -> tuple[np.ndarray, int]:
    """
    Builds a [num_frames, NUM_NOTES] binary label matrix from a GuitarSet
    .jams file's six per-string note_midi annotations, using the same
    frame-from-time convention as generate_synthetic_arpeggios.py. Returns
    the label matrix and a count of ground-truth notes that fell outside
    [NOTE_MIN, NOTE_MAX] and had to be dropped — the synthetic training data
    only covers C3-B6, so a guitar's open low E/A/D strings (MIDI 40/45/50)
    sit below that range and are structurally impossible for this model to
    predict. Reporting the drop count keeps the resulting score honest
    instead of silently inflating it.
    """
    jam = jams.load(jams_path)
    labels = np.zeros((num_frames, NUM_NOTES), dtype=np.float32)
    out_of_range_count = 0
    total_notes = 0

    for ann in jam.search(namespace="note_midi"):
        for obs in ann.data:
            total_notes += 1
            note = int(round(obs.value))
            if note < NOTE_MIN or note > NOTE_MAX:
                out_of_range_count += 1
                continue
            note_index = note - NOTE_MIN
            frame_start = int(obs.time * SAMPLE_RATE / HOP_LENGTH)
            frame_end = int(np.ceil((obs.time + obs.duration) * SAMPLE_RATE / HOP_LENGTH))
            frame_end = min(frame_end, num_frames)
            if frame_start < num_frames:
                labels[frame_start:frame_end, note_index] = 1.0

    return labels, out_of_range_count, total_notes


def find_guitarset_pairs(root: str) -> list[tuple[str, str]]:
    """Pairs each annotation .jams file with its audio_mono-mic .wav counterpart."""
    jams_files = sorted(glob.glob(os.path.join(root, "annotation", "*.jams")))
    pairs = []
    for jpath in jams_files:
        stem = os.path.splitext(os.path.basename(jpath))[0]
        wav_path = os.path.join(root, "audio_mono-mic", f"{stem}_mic.wav")
        if os.path.exists(wav_path):
            pairs.append((jpath, wav_path))
        else:
            print(f"Warning: no matching audio for {jpath} (looked for {wav_path}), skipping")
    return pairs


def compute_log_mel(waveform: torch.Tensor) -> torch.Tensor:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    spec = mel(waveform).squeeze(0)
    return torch.log(spec.clamp(min=1e-6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--guitarset-root", required=True)
    parser.add_argument("--max-files", type=int, default=0, help="0 = evaluate all files")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--best-score-path", default="best_score.json")
    parser.add_argument("--results-path", default="eval_results.json")
    parser.add_argument("--gate-path", default="gate_decision.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}")

    model = NoteDetector().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()

    pairs = find_guitarset_pairs(args.guitarset_root)
    if args.max_files > 0:
        pairs = pairs[: args.max_files]
    if not pairs:
        raise RuntimeError(f"No GuitarSet file pairs found under {args.guitarset_root} — check the layout.")
    print(f"Evaluating against {len(pairs)} GuitarSet recordings")

    all_preds, all_labels = [], []
    total_oor_notes, total_notes = 0, 0

    with torch.no_grad():
        for i, (jams_path, wav_path) in enumerate(pairs):
            print(f"[{i+1}/{len(pairs)}] {os.path.basename(wav_path)}")
            audio, sr = sf.read(wav_path, always_2d=True, dtype="float32")
            waveform = torch.from_numpy(audio.T.copy())
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            mel = compute_log_mel(waveform).unsqueeze(0).to(device)  # [1, N_MELS, T]
            logits = model(mel.unsqueeze(1))  # [1, T, NUM_NOTES] — model is fully convolutional, any T works
            preds = (torch.sigmoid(logits) >= args.threshold).float().squeeze(0).cpu().numpy()

            labels, oor_count, note_count = load_ground_truth_labels(jams_path, preds.shape[0])
            total_oor_notes += oor_count
            total_notes += note_count

            all_preds.append(preds)
            all_labels.append(labels)

    preds_concat = np.concatenate(all_preds, axis=0)
    labels_concat = np.concatenate(all_labels, axis=0)

    from sklearn.metrics import f1_score, precision_score, recall_score
    micro_f1 = float(f1_score(labels_concat, preds_concat, average="micro", zero_division=0))
    precision = float(precision_score(labels_concat, preds_concat, average="micro", zero_division=0))
    recall = float(recall_score(labels_concat, preds_concat, average="micro", zero_division=0))

    results = {
        "frame_micro_f1": micro_f1,
        "frame_precision": precision,
        "frame_recall": recall,
        "num_files_evaluated": len(pairs),
        "ground_truth_notes_total": total_notes,
        "ground_truth_notes_out_of_range": total_oor_notes,
        "out_of_range_fraction": total_oor_notes / total_notes if total_notes else 0.0,
    }
    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFrame-wise micro F1: {micro_f1:.4f} (precision {precision:.4f}, recall {recall:.4f})")
    print(f"Ground truth notes outside the model's C3-B6 range: {total_oor_notes}/{total_notes} "
          f"({results['out_of_range_fraction']*100:.1f}%) — these are structurally unreachable, not model errors.")

    # --- Gating ---
    if os.path.exists(args.best_score_path):
        with open(args.best_score_path) as f:
            previous = json.load(f)
        previous_f1 = previous["frame_micro_f1"]
        improved = micro_f1 > previous_f1
        print(f"Previous best F1: {previous_f1:.4f} -> New: {micro_f1:.4f} -> {'IMPROVED' if improved else 'NOT IMPROVED'}")
        decision = "IMPROVED" if improved else "NOT_IMPROVED"
    else:
        print("No previous best_score.json found — treating this as the new baseline.")
        decision = "NO_BASELINE"

    if decision != "NOT_IMPROVED":
        with open(args.best_score_path, "w") as f:
            json.dump({"frame_micro_f1": micro_f1, "frame_precision": precision, "frame_recall": recall}, f, indent=2)

    with open(args.gate_path, "w") as f:
        f.write(decision)

    print(f"\nGate decision: {decision} (written to {args.gate_path})")


if __name__ == "__main__":
    main()
