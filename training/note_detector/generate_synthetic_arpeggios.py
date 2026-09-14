"""
Generates a synthetic training set of arpeggiated chords: audio + per-frame
"which notes are active" labels. Fully offline after `pip install` — audio is
rendered with pretty_midi's built-in additive synthesis (no soundfont
required), or optionally through a real soundfont via FluidSynth for more
realistic timbre if you have one (--soundfont path/to/file.sf2).

Usage:
    pip install pretty_midi soundfile numpy
    python generate_synthetic_arpeggios.py --output_dir data/arpeggios --examples_per_combo 20

Each example is one (root, quality, direction) arpeggio, rendered with
randomized tempo/velocity/octave/instrument for variety, so the model sees
many acoustic variations of the same underlying note sequence rather than
memorizing one exact recording.
"""

import argparse
import csv
import os
import random

import numpy as np
import pretty_midi
import soundfile as sf

from music_theory import ALL_QUALITIES, ALL_ROOTS, NOTE_NAMES, chord_midi_notes

SAMPLE_RATE = 44100
HOP_LENGTH = 512          # frame hop for labels — must match the hop_length used later in dataset.py's mel spectrogram
NOTE_MIN = 48             # C3 — lower edge of the label/output note range
NOTE_MAX = 95             # B6 — upper edge (48 notes total, covers typical guitar/piano/vocal arpeggio range)
NUM_NOTES = NOTE_MAX - NOTE_MIN + 1

DIRECTIONS = ["up", "down", "up_down", "down_up", "random"]


def order_notes(notes: list[int], direction: str, rng: random.Random) -> list[int]:
    if direction == "up":
        return sorted(notes)
    if direction == "down":
        return sorted(notes, reverse=True)
    if direction == "up_down":
        asc = sorted(notes)
        return asc + asc[-2::-1]  # up then back down, without repeating the top note
    if direction == "down_up":
        desc = sorted(notes, reverse=True)
        return desc + desc[-2::-1]
    if direction == "random":
        shuffled = notes.copy()
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(direction)


def render_arpeggio(midi_notes: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Returns (waveform, labels) where labels is [num_frames, NUM_NOTES] float32."""
    note_duration = rng.uniform(0.12, 0.35)
    lead_in = rng.uniform(0.05, 0.2)
    velocity = rng.randint(60, 110)
    repeats = rng.choice([1, 1, 2])  # mostly single pass, sometimes looped

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=rng.choice([0, 24, 25, 4]))  # piano, nylon/steel guitar, e-piano

    events = []  # (midi_note, start, end)
    t = lead_in
    sequence = midi_notes * repeats
    for note in sequence:
        start, end = t, t + note_duration
        events.append((note, start, end))
        instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=note, start=start, end=end))
        t = end

    pm.instruments.append(instrument)
    tail = rng.uniform(0.1, 0.3)
    total_duration = t + tail

    waveform = pm.synthesize(fs=SAMPLE_RATE)
    target_length = int(total_duration * SAMPLE_RATE)
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    else:
        waveform = waveform[:target_length]

    # Light augmentation so the model doesn't overfit to pretty_midi's exact timbre.
    waveform = waveform.astype(np.float32)
    waveform += np.random.normal(0, 0.003, size=waveform.shape).astype(np.float32)  # faint noise floor
    peak = np.max(np.abs(waveform)) + 1e-8
    waveform = (waveform / peak) * rng.uniform(0.5, 0.95)

    num_frames = int(np.ceil(len(waveform) / HOP_LENGTH))
    labels = np.zeros((num_frames, NUM_NOTES), dtype=np.float32)
    for note, start, end in events:
        note_index = note - NOTE_MIN
        if not (0 <= note_index < NUM_NOTES):
            continue  # outside the modeled range — skip rather than corrupt labels
        frame_start = int(start * SAMPLE_RATE / HOP_LENGTH)
        frame_end = int(np.ceil(end * SAMPLE_RATE / HOP_LENGTH))
        labels[frame_start:frame_end, note_index] = 1.0

    return waveform, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--examples_per_combo", type=int, default=20,
                         help="How many randomized renders per (root, quality, direction) combo.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    audio_dir = os.path.join(args.output_dir, "audio")
    label_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    count = 0
    with open(manifest_path, "w", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["id", "root", "quality", "direction", "audio_path", "label_path"])

        for root in ALL_ROOTS:
            for quality in ALL_QUALITIES:
                base_notes = chord_midi_notes(root, quality)
                for direction in DIRECTIONS:
                    for i in range(args.examples_per_combo):
                        octave_shift = rng.choice([-12, 0, 0, 0, 12])  # mostly one register, some variety
                        notes = order_notes([n + octave_shift for n in base_notes], direction, rng)

                        waveform, labels = render_arpeggio(notes, rng)

                        example_id = f"{NOTE_NAMES[root]}_{quality}_{direction}_{i:03d}"
                        audio_path = os.path.join(audio_dir, f"{example_id}.wav")
                        label_path = os.path.join(label_dir, f"{example_id}.npy")

                        sf.write(audio_path, waveform, SAMPLE_RATE)
                        np.save(label_path, labels)

                        writer.writerow([example_id, root, quality, direction, audio_path, label_path])
                        count += 1

    print(f"Generated {count} examples -> {args.output_dir}")
    print(f"NOTE_MIN={NOTE_MIN}, NOTE_MAX={NOTE_MAX}, NUM_NOTES={NUM_NOTES}, HOP_LENGTH={HOP_LENGTH} "
          f"— keep these in sync with dataset.py and the exported model's expected shape.")


if __name__ == "__main__":
    main()
