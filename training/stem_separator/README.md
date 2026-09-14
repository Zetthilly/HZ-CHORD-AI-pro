# Training: Stem Separation

**Read this before the note-detection README's optimism carries over.** Note
detection above was trained end-to-end, in this very chat, on plain CPU,
because the task and dataset were small enough for that to work. Stem
separation is a different scale of problem — it needs real multi-track
recordings and, realistically, a GPU. Neither exists in the sandbox that
built the Kotlin app, so nothing in this folder has been run yet. What's
here is complete, correct code and a clear plan — the actual training run
has to happen on your machine, or on something like Google Colab.

## Two realistic paths

**Fastest way to actually run Path B**: open
`colab_train_lightweight_unet.ipynb` (in this folder) directly in
[Google Colab](https://colab.research.google.com/) — a full walkthrough
including MUSDB18 download, training with Drive-backed checkpointing, and
export. Path A (fine-tuning Demucs) is described below and in
`finetune_demucs_notes.md`; it's prose rather than a notebook because
Demucs' own training CLI changes between versions and I'd rather point you
at their current docs than hand you commands that might already be stale.


### Path A (recommended): fine-tune pretrained Demucs, don't retrain from scratch

Training a separation model from zero needs the [MUSDB18](https://sigsep.github.io/datasets/musdb.html)
dataset (150 full songs with isolated stems, ~30GB) and, per Demucs' own
published benchmarks, **days of training on a high-end GPU** to match its
quality. Almost nobody should actually do that from scratch — instead:

1. Start from Demucs' official pretrained weights (already trained on
   MUSDB18 + extra data by Meta's research team).
2. **Fine-tune** on your own isolated multitrack recordings, if you have
   any (e.g. actual multitrack exports from a church PA/DAW session where
   vocals, drums, bass, etc. were recorded on separate channels). Fine-
   tuning needs far less data and time than training from scratch, and
   would adapt the model to your specific recording setup and worship-music
   mix characteristics — likely more valuable than generic MUSDB18 accuracy
   for your actual use case.
3. Use Demucs' own training code for this (`facebookresearch/demucs` on
   GitHub) rather than reimplementing their training loop — it's tuned,
   tested infrastructure, and reinventing it would very likely be lower
   quality for the same effort. `finetune_demucs_notes.md` in this folder
   points at the specific entry points.

**If you don't have isolated multitrack recordings of your own**, skip
training entirely: use Demucs' pretrained weights as-is (convert to TFLite
per the main README's steps 2-4) rather than fine-tuning on nothing.

### Path B: train a lightweight model yourself, from scratch, on MUSDB18

If you want to actually build and understand a separation model end-to-end
rather than fine-tune someone else's, `lightweight_unet.py` is a complete,
correct, much smaller model (a spectrogram-masking U-Net, far lighter than
Demucs) you can train from zero on MUSDB18. It'll be lower quality than
Demucs but is a genuine from-scratch training pipeline, and a more tractable
first project than reproducing Demucs.

## Hardware reality check

- **CPU-only training is not practical here.** Even the lightweight model
  processes hours of multi-track audio per epoch; expect a GPU to be a hard
  requirement, not an optimization.
- **Google Colab's free tier** (a T4 GPU, free) is enough to fine-tune
  Demucs or train the lightweight U-Net at a slow-but-workable pace.
  Colab Pro or a rented cloud GPU (e.g. an A10/A100 by the hour) speeds this
  up considerably if you want faster iteration.
- Budget realistic time: fine-tuning Demucs on a modest custom dataset —
  hours to a day or two on a single GPU. Training the lightweight U-Net from
  scratch on full MUSDB18 — likely a day or more, several epochs needed.

## Automating this with GitHub Actions (self-hosted runner)

`.github/workflows/train-stem-separator-selfhosted.yml` exists and is a
real, complete training workflow — but **GitHub's own free-tier runners
have no GPU**, so that workflow targets a *self-hosted* runner instead: a
GPU machine you register to this repo yourself. GitHub Actions still
triggers it automatically (on push, or with a manual click from the Actions
tab) — it just executes on your hardware, not GitHub's. Until a runner with
the `gpu` label is registered, that workflow will sit queued forever with
nothing able to pick it up.

**Setting one up** (your own GPU PC, or a rented cloud GPU box you can SSH
into — a Lambda Labs/RunPod/Paperspace/AWS/GCP instance all work the same
way here):

1. On the GPU machine, one-time setup:
   ```
   pip install torch torchaudio musdb museval ai-edge-torch
   ```
2. In this repo on GitHub: **Settings → Actions → Runners → New self-hosted
   runner**. GitHub gives you a short download-and-configure script specific
   to your OS — run it on the GPU machine. When prompted for labels, add
   `gpu` (this is what `runs-on: [self-hosted, gpu]` in the workflow matches
   against).
3. Run `./run.sh` to start the runner (or `./svc.sh install && ./svc.sh
   start` to install it as a persistent background service, so it's still
   listening after a reboot or if you close the terminal).
4. From then on, pushing to `training/stem_separator/**`, or clicking "Run
   workflow" on the Actions tab, queues the job — GitHub dispatches it to
   your machine automatically, no manual SSH-in-and-run-a-script step needed
   each time.
5. **Cost**: free if it's hardware you already own (just electricity).
   Whatever your cloud provider charges per GPU-hour if it's a rented box —
   GitHub itself doesn't add any extra charge for self-hosted runner usage.

**If you'd rather not manage a runner at all**, GitHub also sells GPU-backed
"larger runners" directly (no self-hosting needed) on Team/Enterprise plans
— see [GitHub's larger runners docs](https://docs.github.com/en/actions/using-github-hosted-runners/using-larger-runners)
if that fits your budget better than either self-hosting or Colab.

## What's in this folder

- `lightweight_unet.py` — model + training loop (PyTorch), Path B.
- `finetune_demucs_notes.md` — how to point Demucs' own training code at
  your data, Path A.
- `export_to_tflite.py` — converts a trained PyTorch model (either path) to
  `.tflite` via `ai-edge-torch`, ready to drop into `TFLiteStemSeparator` in
  the Android project (matching the constants documented in the main
  project README).

None of these have been executed — they need a GPU environment (Colab or
your own machine) with `torch`, `torchaudio`, and (for the export step)
`ai-edge-torch` installed, none of which are available in this chat's
sandbox.
