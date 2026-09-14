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
— it's a ready-to-run walkthrough of everything below, free GPU tier is
enough. The steps below explain what that notebook does, if you'd rather
run it another way (your own machine, a different cloud GPU, etc).

**Fully automatic, no Colab session needed**:
`.github/workflows/train-note-detector-cpu.yml` runs this same pipeline on
GitHub's free CPU-only runners (scaled down — fewer synthetic examples,
fewer epochs — so it actually finishes without a GPU) whenever this folder
changes, and commits the resulting `.tflite` straight into
`app/src/main/assets/models/`. Expect a weaker model than a proper GPU run
via the notebook above; this trades some accuracy for "happens automatically,
for free, on every push." Trigger a bigger one-off run manually from the
Actions tab (it takes `examples_per_combo`/`epochs` inputs) whenever you
want more without touching Colab.

**Not executed anywhere yet** — needs `torch`, `torchaudio`, `pretty_midi`,
and either a GPU or patience on CPU, none of which are available in the
sandbox that built the rest of this project. Run this on Colab or your own
machine.

## Why this design

- **Synthetic data, generated from MIDI** (`generate_synthetic_arpeggios.py`)
  — no dataset to download, license, or clean. It renders arpeggios across
  every chord quality `ChordTemplates.kt` already knows (`music_theory.py`
  mirrors that file exactly), in all 5 arpeggio directions, with randomized
  tempo/velocity/octave/instrument so the model sees many variations rather
  than memorizing one exact rendering. Uses `pretty_midi`'s built-in additive
  synthesis (no soundfont required, though you can pass `--soundfont` for
  more realistic timbre if you have one).
- **No RNN/LSTM layers** (`model.py`) — dilated 1D convolutions provide
  temporal context instead. This is a deliberate choice: recurrent layers
  are one of the most common causes of painful or failed TFLite conversion
  (dynamic shapes, stateful ops). Sticking to convolutions keeps the export
  step (below) straightforward.
- **Frame-wise multi-pitch output**, log-mel spectrogram in, per-frame
  note-activation logits out — the same "which notes are active right now"
  shape `PolyphonicNoteDetector` already produces on-device, so this is a
  drop-in upgrade path rather than a redesign of the app's data flow.

## Running it

```bash
pip install torch torchaudio pretty_midi soundfile numpy ai-edge-torch

python generate_synthetic_arpeggios.py --output_dir data/arpeggios --examples_per_combo 20
python train.py --data_dir data/arpeggios --epochs 30 --output checkpoints/note_detector.pt
python export_to_tflite.py --checkpoint checkpoints/note_detector.pt --output note_detector.tflite
```

`--examples_per_combo 20` × 12 roots × 9 qualities × 5 directions ≈ 10,800
examples — enough to start; increase it if training loss plateaus too early.

## What to watch during training

`train.py` prints train/val loss and an "exact frame accuracy" (every note's
on/off state correct simultaneously — a strict metric). Val loss trending
down and away from train loss diverging upward is the main thing to watch;
exact-frame-accuracy will look low-ish even for a good model (it's a hard,
strict metric across 48 simultaneous binary decisions) — per-note precision/
recall would be a more forgiving metric to add if you want a clearer picture
mid-training.

## After training: bring it into the app

**Note**: `train-note-detector-cpu.yml` already automates steps 1-2 below —
it commits `note_detector.tflite` into `app/src/main/assets/models/`
automatically. Step 3 (the Kotlin wrapper that actually reads and uses that
file) is still a one-time manual task — nothing in the app consumes this
asset yet, so a fresh CI-trained file sits there unused until that class
exists. Worth doing once, since after that every retrain (CI or Colab) just
updates behavior automatically with no further code changes.

1. `export_to_tflite.py` (in this folder) converts the trained checkpoint to
   `.tflite` via `ai-edge-torch`.
2. Drop the resulting file into `app/src/main/assets/models/`.
3. This is a genuinely different model shape than `TFLiteStemSeparator`
   expects (log-mel spectrogram in, note-activation-over-time out, not
   waveform-in/waveform-out) — it needs its own small Kotlin wrapper class
   in `core-audio` (an `Interpreter`-based sibling to `TrainedPitchClassifier`,
   following the same loading pattern `TFLiteStemSeparator` already uses)
   rather than reusing either of those directly. Compute the log-mel
   spectrogram on-device the same way `dataset.py` does here (same
   `N_FFT`/`HOP_LENGTH`/`N_MELS` constants), feed it through the interpreter,
   and map the output note-activation grid back to `VoicedNote`s using
   `NOTE_MIN`/`NOTE_MAX` from `generate_synthetic_arpeggios.py`.
4. Once that wrapper exists, it can replace `HybridArpeggioNoteDetector` as
   the engine behind Note-by-Note mode — same integration point, better model.

## Honest limitation

Synthetic-from-MIDI is a big step up from raw sine synthesis (real ADSR-ish
envelopes, real instrument programs via General MIDI), but it's still not
real recorded instruments, rooms, or playing technique. If actual arpeggio
recordings are available (your own guitar/piano recordings, or datasets like
GuitarSet/MAESTRO), mixing those into training data — even a modest amount —
would likely close a meaningful chunk of the synthetic-to-real gap.
