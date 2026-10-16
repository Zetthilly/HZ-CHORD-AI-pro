"""
Evaluates a trained LightweightSeparatorUNet checkpoint against MUSDB18's
real, professionally-separated reference stems, using museval (the standard
SDR/SIR/SAR metrics for source separation). This is the actual "compare
against already-separated audio" check — MUSDB18's test set already IS
songs paired with real separated stems; this script is what measures the
model against them and decides whether a new checkpoint is actually better
than the last one, rather than just assuming more training = improvement.

Usage:
    python3 evaluate_museval.py \
        --checkpoint lightweight_unet.pt \
        --musdb-root /path/to/musdb18 \
        --best-score-path best_score.json \
        --max-tracks 10   # optional — evaluate a subset for speed

Writes:
    eval_results.json   — full per-stem, per-track scores
    gate_decision.txt   — "IMPROVED", "NOT_IMPROVED", or "NO_BASELINE" (first run)

Exit code is 0 in all cases — the gate decision is read from
gate_decision.txt by the calling workflow/notebook, not from a failing
process, since "the new model isn't better" is a normal outcome, not an error.
"""
import argparse
import json
import os

import musdb
import museval  # API assumed here: museval.EvalStore(), .add_track(), .df with
                 # columns [target, metric, score] — matches museval's documented
                 # usage pattern as of this writing; verify against the installed
                 # version if this script errors on the museval calls below.
import numpy as np
import torch

from lightweight_unet import LightweightSeparatorUNet, STEMS, SAMPLE_RATE, SEGMENT_SAMPLES

OVERLAP_SAMPLES = SAMPLE_RATE * 1  # 1s overlap between chunks, cross-faded


def chunked_separate(model: LightweightSeparatorUNet, mixture: np.ndarray, device: str) -> np.ndarray:
    """
    Runs the model over a full track in SEGMENT_SAMPLES-sized chunks with
    overlap-add, exactly mirroring how TFLiteStemSeparator.kt processes a
    real file on-device — so this evaluation reflects actual on-device
    behavior, not an idealized single-pass inference that the exported
    model will never actually run.
    """
    n_stems = len(STEMS)
    total_len = len(mixture)
    step = SEGMENT_SAMPLES - OVERLAP_SAMPLES

    stem_accum = np.zeros((n_stems, total_len), dtype=np.float64)
    weight_accum = np.zeros(total_len, dtype=np.float64)

    offset = 0
    with torch.no_grad():
        while offset < total_len:
            chunk_end = min(offset + SEGMENT_SAMPLES, total_len)
            chunk_len = chunk_end - offset

            chunk = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
            chunk[:chunk_len] = mixture[offset:chunk_end]

            input_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, SEGMENT_SAMPLES]
            output = model(input_tensor).squeeze(0).cpu().numpy()  # [n_stems, SEGMENT_SAMPLES]

            for i in range(chunk_len):
                idx = offset + i
                fade_in = min(1.0, i / OVERLAP_SAMPLES) if OVERLAP_SAMPLES > 0 else 1.0
                fade_out = min(1.0, (chunk_len - i) / OVERLAP_SAMPLES) if OVERLAP_SAMPLES > 0 else 1.0
                weight = max(min(fade_in, fade_out), 1e-4)
                weight_accum[idx] += weight
                stem_accum[:, idx] += output[:, i] * weight

            offset += step

    weight_accum[weight_accum == 0] = 1.0
    return (stem_accum / weight_accum).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--musdb-root", required=True)
    parser.add_argument("--max-tracks", type=int, default=0, help="0 = evaluate all test tracks")
    parser.add_argument("--best-score-path", default="best_score.json")
    parser.add_argument("--results-path", default="eval_results.json")
    parser.add_argument("--gate-path", default="gate_decision.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}")

    model = LightweightSeparatorUNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()

    mus = musdb.DB(root=args.musdb_root, subsets="test", is_wav=False)
    tracks = list(mus.tracks)
    if args.max_tracks > 0:
        tracks = tracks[: args.max_tracks]
    print(f"Evaluating against {len(tracks)} MUSDB18 test tracks")

    store = museval.EvalStore()
    per_track_summary = []

    for i, track in enumerate(tracks):
        print(f"[{i+1}/{len(tracks)}] {track.name}")
        mixture = np.asarray(track.audio, dtype=np.float32)
        mixture_mono = mixture.mean(axis=1) if mixture.ndim == 2 else mixture

        separated = chunked_separate(model, mixture_mono, device)  # [n_stems, N]

        # museval expects stereo [samples, channels] per target — duplicate our
        # mono estimate to both channels (the model only ever produces mono,
        # consistent with the rest of the app always working in mono).
        estimates = {
            name: np.stack([separated[idx], separated[idx]], axis=1)
            for idx, name in enumerate(STEMS)
        }

        scores = museval.eval_mus_track(track, estimates)
        store.add_track(scores)

        track_sdr = {name: float(np.nanmedian(scores.df[scores.df.target == name]["score"]
                                               [scores.df[scores.df.target == name].metric == "SDR"]))
                     for name in STEMS}
        per_track_summary.append({"track": track.name, "sdr_by_stem": track_sdr})

    results_df = store.df
    per_stem_median_sdr = {
        name: float(np.nanmedian(results_df[(results_df.target == name) & (results_df.metric == "SDR")]["score"]))
        for name in STEMS
    }
    overall_sdr = float(np.nanmean(list(per_stem_median_sdr.values())))

    results = {
        "overall_median_sdr": overall_sdr,
        "per_stem_median_sdr": per_stem_median_sdr,
        "num_tracks_evaluated": len(tracks),
        "per_track": per_track_summary,
    }
    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nOverall median SDR: {overall_sdr:.3f} dB")
    print(f"Per-stem: {per_stem_median_sdr}")

    # --- Gating: only a genuine improvement is worth committing ---
    if os.path.exists(args.best_score_path):
        with open(args.best_score_path) as f:
            previous = json.load(f)
        previous_sdr = previous["overall_median_sdr"]
        improved = overall_sdr > previous_sdr
        print(f"Previous best: {previous_sdr:.3f} dB -> New: {overall_sdr:.3f} dB -> {'IMPROVED' if improved else 'NOT IMPROVED'}")
        decision = "IMPROVED" if improved else "NOT_IMPROVED"
    else:
        print("No previous best_score.json found — treating this as the new baseline.")
        decision = "NO_BASELINE"

    if decision != "NOT_IMPROVED":
        with open(args.best_score_path, "w") as f:
            json.dump({"overall_median_sdr": overall_sdr, "per_stem_median_sdr": per_stem_median_sdr}, f, indent=2)

    with open(args.gate_path, "w") as f:
        f.write(decision)

    print(f"\nGate decision: {decision} (written to {args.gate_path})")


if __name__ == "__main__":
    main()
