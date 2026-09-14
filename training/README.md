# Training your own models for Ichi

Three pipelines, for the two places in the app that genuinely need a
trained model rather than a formula. Two of them target the same job
(note-by-note detection) at different levels of sophistication.

| Pipeline | Status | Replaces/feeds | Input → Output |
|---|---|---|---|
| `note_detection/` | **Already trained** — weights are live in the app right now | `TrainedPitchClassifier` (feeds `HybridArpeggioNoteDetector`) | Chroma vector → which pitch classes are sounding |
| `note_detector/` | **Automated on free GitHub CPU runners** (scaled-down) via `.github/workflows/train-note-detector-cpu.yml`, or full-strength via the Colab notebook | Would upgrade/replace the above — Kotlin consumer not yet wired up, see `note_detector/README.md` | Log-mel spectrogram → which exact notes (with octave) are sounding, frame-by-frame |
| `stem_separator/` | **Automatable via a self-hosted GPU runner** (`.github/workflows/train-stem-separator-selfhosted.yml`) — GitHub's free runners have no GPU and can't run this one | `TFLiteStemSeparator` | Mixture waveform → per-stem waveform |

## `note_detection/` — already done

This one was actually generated, trained, evaluated, and exported in this
project's own chat session — no GPU, no dataset download, just numpy/
scikit-learn on CPU, because the task (chroma → pitch-class) and model
(12 → 32 → 12 MLP) are small enough for that to be enough. Real results on
held-out synthetic data: 93% per-pitch-class F1. Its weights are already
sitting in `core-audio/.../TrainedPitchClassifier.kt`, wired into
`HybridArpeggioNoteDetector`, which is what Note-by-Note mode uses today.
`.github/workflows/train-note-detection.yml` re-runs this pipeline and
commits the updated weights automatically whenever this folder changes.
See `note_detection/README.md` for the full pipeline and results.

## `note_detector/` — the upgrade path

Chroma → pitch-class throws away octave information and only sees one
frame at a time. `note_detector/` is a full CNN (log-mel spectrogram in,
per-frame note-*with-octave* activations out, no RNN layers so TFLite
conversion stays simple) trained on MIDI-synthesized arpeggios covering
every chord quality the app knows. This needs an actual training run
(`pip install torch torchaudio pretty_midi`, then a GPU or a patient CPU) —
a genuinely good weekend project, not something a chat sandbox can execute.
See `note_detector/README.md`.

## `stem_separator/` — hardest, budget the most time

Source separation needs real recorded multi-track music to learn from, not
synthetic audio — synthesizing "vocals" or "drums" convincingly enough to
train a separator on is its own unsolved problem, so this one leans on
either fine-tuning Demucs' existing pretrained weights on your own
multitrack recordings (Path A, recommended, needs isolated stems you
already have), or training a lighter model from scratch on the MUSDB18
dataset (Path B, needs a ~30GB download + real GPU-hours). See
`stem_separator/README.md`.

## Shared conventions

All three:
- Work at **44.1kHz mono**, matching what `AudioFileDecoder` already
  resamples everything to on-device.
- Export via **`ai-edge-torch`** (PyTorch → `.tflite` directly) where a
  TFLite model is the end goal — `note_detection/`'s MLP is small enough to
  skip this and embed weights as Kotlin array literals instead.
- End up loaded through the same on-device pattern already in the app: an
  `Interpreter` over a memory-mapped asset (`TFLiteStemSeparator` is the
  reference implementation of that pattern).

Start with `note_detection/` — it's done. `note_detector/` is the natural
next step if you want real note+octave output instead of just pitch class.
`stem_separator/` is a separate, larger undertaking whenever you're ready
for it.
