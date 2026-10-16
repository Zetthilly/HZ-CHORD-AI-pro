# Training: Note Detector (advanced — real note+octave, not just pitch class)

This is the scaled-up sibling to `../note_detection/` (the simple 12-dim
chroma MLP that's already trained and embedded in the app as
`TrainedPitchClassifier.kt`). That one tells you *which pitch classes* are
sounding; this one is designed to tell you the *actual notes with octave*
(e.g. "E4", not just "E") frame-by-frame across a whole arpeggio — a more
useful output for note-by-note display, at the cost of needing a real
training run on a GPU rather than something that fits in a chat sandbox.

**Fastest way to actually run this**: open `colab_train_note_detector.ipynb`
(in this folder) directly in [Google Colab](https://colab.research.google.com/)
— trains, evaluates against real GuitarSet recordings, and pushes the
result back to GitHub if (and only if) it actually improved.

**Fully automatic, no Colab session needed**:
`.github/workflows/train-note-detector-cpu.yml` runs the same pipeline on
GitHub's free CPU-only runners (scaled down — fewer synthetic examples,
fewer epochs — so it actually finishes without a GPU) whenever this folder
changes. Expect a weaker model than a proper GPU run via the notebook; this
trades some accuracy for "happens automatically, for free, on every push."

## Why this design

- **Synthetic data, generated from MIDI** (`generate_synthetic_arpeggios.py`)
  — no dataset to download, license, or clean, for *training*. It renders
  arpeggios across every chord quality `ChordTemplates.kt` already knows
  (`music_theory.py` mirrors that file exactly), in all 5 arpeggio
  directions, with randomized tempo/velocity/octave/instrument.
- **No RNN/LSTM layers** (`model.py`) — dilated 1D convolutions provide
  temporal context instead, keeping TFLite/LiteRT conversion straightforward.
- **Frame-wise multi-pitch output**, log-mel spectrogram in, per-frame
  note-activation logits out — the same shape `PolyphonicNoteDetector`
  already produces on-device, so this is a drop-in upgrade path.

## Evaluation & gating — against REAL recordings, not synthetic data

Training is synthetic-only, but **evaluation isn't**: `evaluate_guitarset.py`
scores the trained model against [GuitarSet](https://guitarset.weebly.com/)
— real guitar recordings with genuine note/string/fret ground truth, openly
licensed (CC BY 4.0) for research. Since the model never trains on
GuitarSet, there's no train/test leakage — the full dataset is a fair,
honest, out-of-domain check on whether the synthetic training actually
generalizes.

`prepare_guitarset.py` downloads it (cached across runs, same pattern as
MUSDB18 elsewhere in this project — check
[guitarset.weebly.com](https://guitarset.weebly.com/) for the current URL if
the one in that script 404s).

Each run:
1. Scores frame-wise micro-F1 against GuitarSet's real note annotations.
2. Compares against `best_score.json` (committed in this folder — the
   running record of the best score achieved so far).
3. Only commits/pushes the new model if it actually scored higher.
   `gate_decision.txt` records `IMPROVED`, `NOT_IMPROVED`, or `NO_BASELINE`
   (first run ever) — both the CI workflow and the Colab notebook read this
   file and skip the push entirely on `NOT_IMPROVED`, so a training run that
   made things worse can't silently overwrite a better model.

**Honest limitation of the evaluation itself**: the synthetic training data
only covers C3-B6 (MIDI 48-95), so a guitar's open low E/A/D strings (MIDI
40/45/50) sit below that range and are structurally impossible for this
model to predict — not a model failure, just outside its labeled range.
`evaluate_guitarset.py` reports what fraction of ground-truth notes fell
into this category so the F1 score stays honest rather than silently
deflated by notes the model was never going to get right regardless of
training quality.

## Running it by hand

```bash
pip install torch torchaudio pretty_midi soundfile numpy litert-torch jams scikit-learn

python generate_synthetic_arpeggios.py --output_dir data/arpeggios --examples_per_combo 20
python train.py --data_dir data/arpeggios --epochs 30 --output checkpoints/note_detector.pt

python prepare_guitarset.py --output-dir data/guitarset
python evaluate_guitarset.py --checkpoint checkpoints/note_detector.pt --guitarset-root data/guitarset --max-files 30

python export_to_tflite.py --checkpoint checkpoints/note_detector.pt --output note_detector.tflite
```

`--examples_per_combo 20` × 12 roots × 9 qualities × 5 directions ≈ 10,800
examples — enough to start; increase it if training loss plateaus too early.

## What to watch during training

`train.py` prints train/val loss and an "exact frame accuracy" (every note's
on/off state correct simultaneously — a strict metric). Val loss trending
down and away from train loss diverging upward is the main thing to watch.
The real signal for whether training quality is actually improving, though,
is the GuitarSet F1 score from the evaluation step above — that's measured
against real audio, not synthetic data the model was trained to fit.

## After training: bring it into the app

**Note**: the CI workflow and Colab notebook both already automate
generation → training → evaluation → gating → committing `note_detector.tflite`
into `app/src/main/assets/models/`. The one remaining manual step: **nothing
in the app consumes that file yet** — it needs a small `Interpreter`-based
Kotlin wrapper class in `core-audio` (an `Interpreter`-based sibling to
`TrainedPitchClassifier`, following the same loading pattern
`TFLiteStemSeparator` already uses). Compute the log-mel spectrogram
on-device the same way `dataset.py` does here (same `N_FFT`/`HOP_LENGTH`/
`N_MELS` constants), feed it through the interpreter, and map the output
note-activation grid back to `VoicedNote`s using `NOTE_MIN`/`NOTE_MAX` from
`generate_synthetic_arpeggios.py`. Once that exists, it can replace
`HybridArpeggioNoteDetector` as the engine behind Note-by-Note mode — same
integration point, better (and now continuously gated) model.
