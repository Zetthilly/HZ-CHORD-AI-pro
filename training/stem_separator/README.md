# HZ-CHORD-AI Stem Separator Training

This folder now contains a **single, consistent training/export contract**:

- Dataset: MUSDB18, four targets: `drums`, `bass`, `other`, `vocals`.
- Sample rate: **44.1 kHz**.
- Training chunk: **8 seconds = 352,800 samples**.
- Model input: `[batch, 1, 352800]` internally.
- LiteRT/TFLite input: `[1, 352800]` float32.
- LiteRT/TFLite output: `[1, 4, 352800]` float32.
- Output order: `drums, bass, other, vocals`.
- Android's existing `TFLiteStemSeparator.kt` is therefore compatible with the exported model shape.

## What was fixed

The previous implementation trained a spectrogram-mask U-Net but attempted to export it as if it accepted raw waveforms. That could not produce a model matching the Android runtime. The new model is a compact **waveform U-Net**, so training and deployment use the same input/output contract.

The old export used the deprecated `ai-edge-torch` package. The workflow now uses Google's current `litert-torch` package. LiteRT Torch's documented flow is `litert_torch.convert(model, sample_inputs)` followed by `.export("model.tflite")`. urlLiteRT Torch conversion guidehttps://github.com/google-ai-edge/litert-torch/blob/main/docs/pytorch_converter/README.md

The MUSDB18 archive is taken from the official Zenodo record. The regular MUSDB18 release is 4.7 GB compressed and contains 100 training and 50 test tracks. urlMUSDB18 Zenodo recordhttps://zenodo.org/records/1117372

## Files

- `lightweight_unet.py` — model, MUSDB loader, mixed-precision GPU training, checkpoint resume, spectral + waveform loss.
- `export_to_tflite.py` — loads a checkpoint, converts with `litert-torch`, and performs a PyTorch/LiteRT numerical smoke test.
- `requirements.txt` — pinned minimum runtime dependencies, including `litert-torch` and an ffmpeg provider.
- `.github/workflows/train-stem-separator-selfhosted.yml` — reproducible GPU training/export workflow.

## GitHub Actions requirements

The workflow intentionally uses:

```yaml
runs-on: [self-hosted, gpu]
```

This is not a normal GitHub-hosted runner. A machine with an NVIDIA GPU must be registered to the repository as a self-hosted runner and have the `gpu` label. If GitHub shows **"Waiting for a runner to pick up this job"**, the training code has not started yet; the runner is missing/offline/missing the label.

Once the runner is online, manually run **Actions → Train Stem Separator (self-hosted GPU) → Run workflow**.

The workflow:

1. Checks CUDA.
2. Creates an isolated Python environment.
3. Installs the training/export dependencies.
4. Provides ffmpeg for MUSDB18 decoding.
5. Downloads MUSDB18 once into the runner tool cache.
6. Trains or resumes the waveform separator.
7. Exports and numerically validates the `.tflite` model.
8. Copies it to `app/src/main/assets/models/stem_separator.tflite`.
9. Uploads the checkpoint/model as workflow artifacts.
10. Commits the validated TFLite model to the branch.

## First run recommendation

Do not start with 50 epochs. Use a smoke run first:

- epochs: `1`
- batch size: `1` or `2`
- segments per track: `1`

This verifies CUDA, MUSDB decoding, training, checkpoint writing and LiteRT conversion. After that succeeds, run a real training job with more epochs/segments.

## Important quality note

This compact model is intended to be a practical HZ-CHORD-AI on-device model, not a replacement for a large pretrained Demucs model. It is a genuine trainable four-stem model and produces the exact waveform interface required by the Android app. More training data, more epochs, and a stronger architecture can improve separation quality later.

MUSDB18 is provided for educational purposes under its dataset terms; review those terms before distributing a model trained on it commercially.
