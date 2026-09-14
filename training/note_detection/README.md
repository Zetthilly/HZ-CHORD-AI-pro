# Training: Note-by-Note Arpeggio Detection

This one was actually trained — the whole pipeline below ran end-to-end on
plain CPU (no GPU needed for a model this size) and produced the real,
working weights baked into
`core-audio/src/main/java/.../core/theory/TrainedPitchClassifier.kt` in the
Android project.

## What it does

Input: a 12-dim chroma vector (the same one `ChromaExtractor.kt` already
computes on-device). Output: 12 probabilities — one per pitch class (C, C#,
D, ... B) — for whether that pitch class is *actually* sounding right now,
as opposed to just showing up as a harmonic of some other note.

That's the specific, narrow problem worth training a model for here: a
single played note lights up multiple chroma bins (its own harmonics), and a
fixed threshold or hand-tuned harmonic-suppression rule (which is what
`PolyphonicNoteDetector.kt` uses on its own) will sometimes get fooled by a
strong overtone. A model trained on many mixtures can learn the *pattern* of
"this combination of bins usually means one real note with harmonics," not
just react to the loudest peaks.

## Pipeline (run in order)

```
python3 synth_dataset.py               # sanity-check: prints a few examples
python3 train_mlp.py                   # generates 24,000 examples, trains, evaluates
python3 export_weights_to_kotlin.py    # writes TrainedPitchClassifier.kt
```

- **`chroma_features.py`** — a Python port of `ChromaExtractor.kt`, kept in
  lockstep so training data matches what the app computes on-device. If you
  change the Kotlin version, mirror the change here before regenerating data.
- **`synth_dataset.py`** — synthesizes 1-4 simultaneous notes (with 5
  harmonics each, random decay/detune/phase, background noise) across a
  ~4-octave range, and labels which pitch classes are truly present.
- **`train_mlp.py`** — trains a small multilabel MLP (12 → 32 → 12,
  scikit-learn's `MLPClassifier`) and evaluates on a disjoint held-out set.
- **`export_weights_to_kotlin.py`** — writes the learned weights (812 float
  values total) directly as Kotlin array literals with a hand-written
  forward pass. No TFLite/ONNX runtime needed on-device for a model this
  small — it's genuinely lighter to just embed the numbers.

## Actual results (see `training_run_log.txt`)

On 4,000 held-out synthetic examples (disjoint random seed from training):

- **Per-pitch-class F1: 0.930** (micro-averaged)
- **Hamming loss: 0.026** — individual pitch-class labels are correct ~97.4%
  of the time
- **Exact whole-chord match: 74.6%** — all 12 pitch classes correct
  simultaneously, the strictest possible metric

## The honest limitation: this is trained on synthetic audio

The harmonic-mixture synthesis is a reasonable stand-in for the *specific*
problem (telling fundamentals from overtones), but it doesn't capture real
instrument timbre, room noise, amp distortion, reverb, or genre-specific
playing style. Treat `TrainedPitchClassifier` as a real, working first
model and a correct scaffold — not a finished, studio-grade one.

## Scaling this up with real data

To meaningfully improve on real recordings:

1. **Get labeled real audio.** [MAESTRO](https://magenta.tensorflow.org/datasets/maestro)
   (piano, MIDI-aligned) and [GuitarSet](https://guitarset.weebly.com/)
   (guitar, annotated) are the standard open datasets for this task. Neither
   needs a network call *from the app* — you download them once, on your own
   machine, for training.
2. **Move to a real deep-learning framework.** A dataset that size wants a
   proper frame-sequence model (a small CNN or CRNN over a CQT/log-mel
   spectrogram, closer to Spotify's open-source **Basic Pitch** architecture)
   trained in PyTorch on a GPU — [Google Colab's free tier](https://colab.research.google.com/)
   is enough for this scale of model. That's genuinely a different, heavier
   pipeline than the one here; happy to build that training script too if
   you want to go that route.
3. **Re-run `export_weights_to_kotlin.py`-equivalent for the new model.** A
   bigger CNN/CRNN won't fit as embedded array literals — at that size,
   convert to TFLite instead (same `ai-edge-torch` path described in the
   stem-separation README) and load it the same way `TFLiteStemSeparator`
   already does.

Or, faster than training anything further yourself: **Basic Pitch is already
trained, open-source (MIT license), and specifically built for this task**
(including arpeggios). Converting *it* to TFLite would very likely beat
further tuning of this synthetic-only model, for a fraction of the effort.
Worth trying before investing in a bigger custom training run.
