# HZ Chord AI (Ichi) — Chord Detector + Stem Splitter

Two fully decoupled modules that can each run standalone, or hand audio to
each other through an explicit, opt-in bridge. Built by Joseph Hilary Zulukwa.

## Real file import

Stem Splitter's "Import Audio" button opens the system Storage Access
Framework document picker (`ActivityResultContracts.OpenDocument`, filtered
to `audio/*`) — no storage permission needed, works the same across Android
versions. The picked file is decoded by `AudioFileDecoder` (`core-audio`)
using Android's own `MediaExtractor` + `MediaCodec` — real decode of
mp3/m4a/aac/wav/ogg/flac (whatever codecs the device ships with), downmixed
to mono and resampled to 44.1kHz via `LinearResampler`, entirely offline, no
third-party decoding library. Decoding runs on `Dispatchers.IO` so the UI
stays responsive on a long file; a "Decoding…" state and a failure message
are both surfaced if something goes wrong.

## Module independence

Chord Detector and Stem Splitter do not depend on each other's code. The
only connection is `AudioResourceBus` in `core-audio` — a one-slot mailbox
both already depend on:

- **Stem Splitter → Chord Detector**: "Send to Chord Detector" on any stem
  row posts that stem's isolated audio to the bus; Chord Detector picks it
  up once when its screen opens.
- **Chord Detector → Stem Splitter**: Chord Detector keeps a rolling 15s
  capture of live mic audio; "Send Captured Audio to Stem Splitter" posts
  that back through the same bus.

Stem Splitter's own screen never computes or displays a chord.

## Playback, everywhere

- `PcmTrackPlayer` (single clip) and `MultiStemPlayer` (keeps several clips
  in sync under one transport) live in `core-audio`, exposed through a
  shared `PlaybackControls` widget (`core-ui`) used by both screens:
  play/pause, scrub bar, speed toggle (0.5x–2x).
- **Chord Detector**: once a clip is loaded (an imported stem, or anything
  sent via the bus), you get a real transport, and chord/note analysis
  tracks wherever you are in the clip, live.
- **Stem Splitter**: one shared transport moves all six stems together so
  they can't drift out of sync; each stem still has its own mute/solo/volume.

## Export

- **Save to Device** (`WavFileWriter`) writes a stem to a real 16-bit PCM
  `.wav` under `.../exports/`, playable by any audio app.
- **Send to Chord Detector** uses the bus hand-off above.

## Chord Detector controls

| Control | Effect |
|---|---|
| **Display**: Chord / Note-by-Note | Chord mode: chroma → template matching. Note-by-Note: `HybridArpeggioNoteDetector` — a genuinely trained model (`TrainedPitchClassifier`, see `/training/note_detection/`) decides which pitch classes are sounding, spectral peak-picking supplies the octave for each. |
| **Voicing**: General / Exact | General: standard ascending arrangement. Exact: matches each chord tone to a real detected spectral peak (actual register played). |
| **Complexity**: Simple / Complete | Simple: plain triad. Complete: full detected quality (7ths, extensions). |
| **Transpose** (−12…+12) | Relabels the chord/notes/keyboard by N semitones (capo-style), applied at display time — instant, no re-analysis. Does not change playback pitch. |
| **Speed** (0.5x–2x) | Real varispeed on the loaded clip — pitch moves with speed. Only active once a clip is loaded. |

**Training your own models**: `/training/` at the project root has three
pipelines — one already trained and embedded (the pitch-class model above),
one ready to run for real note+octave detection, and one for stem
separation. See `/training/README.md`.

---

## Getting a trained model into this app (the part that genuinely needs one)

Everything described above — chord templates, chroma, peak-picking,
transpose, playback, file decode, WAV export — is deterministic DSP and
music-theory *math*, not a trained model. It works today, offline, with no
model file, because chord/pitch/tempo recognition from a clean or
moderately-isolated signal doesn't need a neural net to do well.

**Source separation is different.** Splitting a mixed recording into clean
vocals/drums/bass/etc. is not something a formula can do well — it requires
a model that has actually learned what "vocals" or "drums" sound like from
thousands of examples. That's the one piece in this app where "just write
the math" doesn't work, and a trained model is the only real path. Here's
how to get one in, running fully on-device:

### 1. Get a pretrained checkpoint
Use an existing open, permissively-licensed model rather than training from
scratch:
- **Demucs v4 / htdemucs** (Meta, MIT license) — best quality, larger/slower.
- **Spleeter** (Deezer, MIT license) — lighter, faster, slightly lower quality.

Both publish pretrained weights on GitHub/Hugging Face — no need to train
anything yourself for a first working version.

### 2. Convert it to TensorFlow Lite — offline, no cloud service
- **Simplest path**: `ai-edge-torch` (Google) converts a PyTorch `nn.Module`
  directly to `.tflite` in one step. Try this first.
- **Fallback path** if a model's ops aren't yet supported by ai-edge-torch:
  PyTorch → ONNX (`torch.onnx.export`) → TensorFlow (`onnx-tf`) → TFLite
  (`tf.lite.TFLiteConverter`).

Both are one-time, run-on-your-laptop conversion steps — nothing about the
*app* needs network access; only the conversion tooling (which you run
yourself, once) does.

### 3. Quantize
Apply post-training quantization (dynamic-range at minimum, full int8 with a
representative dataset for best results). This is what takes a model from
"too big/slow for a phone" to something that actually runs — expect to
trade off some quality for size/speed here, and to iterate.

### 4. Drop the file in and rebuild
Place the `.tflite` file at `app/src/main/assets/models/stem_separator.tflite`
(or update `modelAssetPath` in `TFLiteStemSeparator` to match your filename).
`app/build.gradle.kts` already sets `noCompress += "tflite"` so the file is
memory-mapped directly at load time instead of being decompressed first.

### 5. Match the shape constants to your specific model
`TFLiteStemSeparator` in `feature-stems` is a real, working `Interpreter`
integration — chunked inference with overlap-add reconstruction so long
files don't need to fit in one inference call, cross-faded so chunk
boundaries don't click. But exact input/output tensor shapes and stem
ordering vary by model and by how you converted it, so three constants at
the top of that class need to match your specific `.tflite` file:

```kotlin
private val MODEL_CHUNK_SAMPLES = 44100 * 8   // how many samples per inference call
private val MODEL_OVERLAP_SAMPLES = 44100 * 1 // overlap between chunks, for the cross-fade
private val MODEL_STEM_COUNT = 4              // how many stems the model outputs
```

Inspect your converted model in Python first
(`interpreter.get_input_details()` / `get_output_details()` from
`tensorflow.lite`) to get these exactly right, and check `assembleStems()`
in the same file — most open 4-stem models output
`[drums, bass, other, vocals]`, which is what that function assumes;
reorder it if your model's convention differs. A 4-stem model also can't
distinguish guitar from keys, so both currently map to "other" — using a
6-stem model (if you find or train one) would let you split those out too.

### 6. Run it — entirely on-device
`org.tensorflow:tensorflow-lite` (already added to `feature-stems`'s
Gradle file) does the inference locally via `Interpreter.run(...)` — no
network call, no server round-trip, nothing sent anywhere. That's what
"offline, without depending on calculations" means in practice: the output
comes from a real trained model's forward pass on your phone's CPU (or GPU/
NNAPI, optionally — see the commented-out delegate lines in
`TFLiteStemSeparator`), not from a DSP formula standing in for one.

If the model asset isn't there yet, Stem Splitter surfaces a clear
`StemModelNotFoundException` message in the UI rather than failing silently
— you'll know exactly what's missing.

---

## Building

Open the project in Android Studio — it'll offer to generate the Gradle
wrapper (`gradlew`/`gradlew.bat`) the first time, using its bundled Gradle.
From the command line without that wrapper yet, use a local Gradle install:

```
gradle assembleDebug
```

Minimum SDK 24, target/compile SDK 34, Kotlin 1.9.24, Compose BOM 2024.06.00.

## Continuous integration

Three GitHub Actions workflows live in `.github/workflows/`:

| Workflow | Runs on | What it actually does |
|---|---|---|
| `android-build.yml` | every push/PR to `main` that touches Android source | Builds the debug APK and uploads it as an artifact. Provisions Gradle itself (`gradle/actions/setup-gradle`), so it doesn't depend on a checked-in wrapper. |
| `train-note-detection.yml` | every push touching `training/note_detection/**` | **Genuinely retrains** the pitch-classifier MLP on GitHub's free CPU runners (no GPU needed for this one), then commits the updated `TrainedPitchClassifier.kt` straight back into `core-audio/` if the weights changed. Can also be triggered manually from the Actions tab. |
| `train-heavy-smoke-test.yml` | pushes/PRs touching `training/note_detector/**` or `training/stem_separator/**` | **Not real training** — just catches broken code (import errors, shape mismatches, API drift) in CPU-minutes, before you find out after a multi-hour real run. |
| `train-note-detector-cpu.yml` | every push touching `training/note_detector/**` | **Real training**, scaled down to fit GitHub's free CPU runners (no GPU here either) — fewer synthetic examples, fewer epochs than the Colab notebook's defaults. Commits the resulting `.tflite` into `app/src/main/assets/models/` automatically. Manually trigger with larger `examples_per_combo`/`epochs` inputs for a stronger pass without touching Colab. |
| `train-stem-separator-selfhosted.yml` | manual trigger only, on a **self-hosted GPU runner** | GitHub's free runners can't run this one at all (no GPU, no room for the ~30GB dataset) — this workflow targets a GPU machine you register yourself (own PC or a rented cloud GPU box). See `training/stem_separator/README.md` for the runner setup steps; until one's registered with the `gpu` label, this workflow just sits queued. |

## Project structure

```
app/               — navigation, dashboard, module launcher, theme
core-audio/        — FFT, chroma, chord templates/detection, polyphonic note
                      detection, voicing/complexity arrangement, transpose,
                      PcmTrackPlayer, MultiStemPlayer, WavFileWriter,
                      AudioFileDecoder, AudioResourceBus
core-ui/           — PlaybackControls composable, shared by both features
feature-chord/     — Chord Detector screen + ViewModel
feature-keyboard/  — Piano keyboard Compose widget
feature-stems/     — Stem Splitter screen + ViewModel, TFLiteStemSeparator
                      (real TF Lite integration — needs a model file, see above)
```

## Next steps

1. `./gradlew assembleDebug` in Android Studio, fix any local SDK/AGP version drift.
2. Convert a model per the steps above — the single biggest remaining lift.
3. Verify `MODEL_CHUNK_SAMPLES` / `MODEL_STEM_COUNT` / `assembleStems()`
   against your specific converted model.
4. If real-time pitch-preserving time-stretch or actual audio transposition
   is wanted later, that's a new DSP module (phase vocoder / PSOLA) —
   current Speed and Transpose controls intentionally keep those separate.
