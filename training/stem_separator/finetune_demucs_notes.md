# Fine-tuning Demucs (Path A — recommended)

This is notes, not a turnkey script — Demucs' own training code changes
between versions, so treat this as a map of where to look rather than exact
commands to paste. Confirm against the current
[facebookresearch/demucs](https://github.com/facebookresearch/demucs) README
when you actually do this.

## Why fine-tune instead of retrain

Demucs' released weights were trained on MUSDB18 plus a large amount of
extra licensed music, by a team with a GPU cluster, for far longer than is
practical solo. Retraining that from scratch to matching quality isn't
realistic. **Fine-tuning** — continuing training from those weights on a
*small* amount of your own data — needs far less compute and data, and can
meaningfully adapt the model to your specific recordings (worship-service
mixes, a particular PA/mic setup, particular instrumentation) in a way the
generic weights won't be tuned for.

## What you need

**Isolated multitrack recordings** — actual separate audio for vocals,
drums, bass, etc. from the same performances, not just the final mix. If
your church's sound desk or DAW exports multitrack sessions (many digital
mixers do), that's exactly this. A surprisingly small number of songs
(dozens, not hundreds) can be useful for fine-tuning, unlike training from
scratch which needs MUSDB18-scale data.

If you don't have this, there's nothing to fine-tune with — use the
pretrained weights as-is rather than skipping straight to Path B's
from-scratch training on generic MUSDB18 data, which won't be any more
tailored to your use case than the pretrained weights already are.

## Rough steps

1. `pip install demucs` (or clone the repo directly for training code —
   the pip package is inference-focused; training entry points live in the
   GitHub repo's `demucs/` and `tools/` directories).
2. Format your multitrack recordings to match Demucs' expected dataset
   layout — the repo's training docs describe the exact folder structure
   (per-track folders containing separate stem `.wav` files, similar to
   MUSDB18's layout).
3. Use the repo's training entry point (check `demucs/train.py` or the
   documented CLI in the current version) with `--continue_pretrained` or
   equivalent flag pointing at the released checkpoint, and a low learning
   rate — fine-tuning wants small updates, not training from a random init.
4. Train for a modest number of epochs (tens, not hundreds) and evaluate
   using `museval` against a held-out portion of your own multitrack set.
5. Export the fine-tuned checkpoint the same way as the base model: PyTorch
   → `ai-edge-torch` → `.tflite` (see `export_to_tflite.py` in this folder),
   then drop it into `app/src/main/assets/models/` per the main README.

## Compute expectation

A single GPU (Colab's free T4 is workable, though slow; a rented A10/A100
by the hour is faster for iteration) for a few hours to a day, depending on
how much custom data you have and how many epochs you fine-tune for.
