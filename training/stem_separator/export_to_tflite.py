"""Export the HZ-CHORD-AI waveform separator to LiteRT/TFLite.

Input : [1, 352800] float32
Output: [1, 4, 352800] float32
Order : drums, bass, other, vocals
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lightweight_unet import LightweightSeparatorUNet, SEGMENT_SAMPLES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="stem_separator.tflite")
    p.add_argument("--chunk-samples", type=int, default=SEGMENT_SAMPLES)
    args = p.parse_args()

    if args.chunk_samples != SEGMENT_SAMPLES:
        raise ValueError(
            f"This Android build expects {SEGMENT_SAMPLES} samples (8s @ 44.1kHz); "
            f"got {args.chunk_samples}."
        )

    try:
        import litert_torch
    except ImportError as exc:
        raise SystemExit("Install litert-torch>=0.9.4 before export: pip install litert-torch") from exc

    model = LightweightSeparatorUNet().eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt))

    sample = torch.zeros(1, 1, args.chunk_samples, dtype=torch.float32)
    with torch.no_grad():
        torch_out = model(sample).cpu().numpy()

    print("PyTorch input :", tuple(sample.shape))
    print("PyTorch output:", tuple(torch_out.shape))

    # The Android side passes [1, N], so use a tiny wrapper to keep the public
    # TFLite signature exactly [1, N] while the network remains [B, C, N].
    class AndroidWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, waveform):
            return self.inner(waveform.unsqueeze(1))

    wrapper = AndroidWrapper(model).eval()
    android_input = torch.zeros(1, args.chunk_samples, dtype=torch.float32)
    edge_model = litert_torch.convert(wrapper, (android_input,))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(output))

    # Conversion smoke test before CI claims success.
    edge_out = np.asarray(edge_model(android_input))
    expected = torch_out
    if edge_out.shape != expected.shape:
        raise RuntimeError(f"TFLite shape mismatch: got {edge_out.shape}, expected {expected.shape}")
    if not np.allclose(edge_out, expected, atol=2e-3, rtol=2e-3):
        max_err = float(np.max(np.abs(edge_out - expected)))
        raise RuntimeError(f"TFLite/PyTorch mismatch; max abs error={max_err}")

    print(f"Wrote validated LiteRT model: {output}")
    print("Input : [1, 352800] float32")
    print("Output: [1, 4, 352800] float32")
    print("Stems : drums, bass, other, vocals")


if __name__ == "__main__":
    main()
