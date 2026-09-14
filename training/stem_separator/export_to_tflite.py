"""
Converts a trained PyTorch separation model (either the fine-tuned Demucs
checkpoint from Path A, or lightweight_unet.py's checkpoint from Path B) to
TensorFlow Lite, ready to drop into the Android project's
TFLiteStemSeparator.

Requires: torch, ai-edge-torch
    pip install torch ai-edge-torch

NOT executed in the sandbox that produced this file — no torch, no
ai-edge-torch, no network to install them there. Run this on the same
machine/environment where you trained the model.

    python3 export_to_tflite.py --checkpoint lightweight_unet.pt --output stem_separator.tflite
"""
import argparse
import torch
import ai_edge_torch

from lightweight_unet import LightweightSeparatorUNet, N_FFT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pt checkpoint")
    parser.add_argument("--output", default="stem_separator.tflite")
    parser.add_argument(
        "--chunk-samples", type=int, default=44100 * 8,
        help="Must match MODEL_CHUNK_SAMPLES in TFLiteStemSeparator.kt",
    )
    args = parser.parse_args()

    model = LightweightSeparatorUNet()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    # IMPORTANT: TFLiteStemSeparator.kt currently sends a raw waveform chunk
    # and expects a raw waveform back per stem. lightweight_unet.py's
    # LightweightSeparatorUNet operates on a spectrogram internally — if you
    # use this exact model, wrap it in a module that does STFT -> mask ->
    # iSTFT internally so the exported .tflite model's input/output are both
    # raw waveforms, matching what the Kotlin side sends. A minimal wrapper:
    #
    #   class WaveformInOutWrapper(torch.nn.Module):
    #       def __init__(self, inner_model):
    #           super().__init__()
    #           self.inner = inner_model
    #       def forward(self, waveform):
    #           mag, phase = stft_mag_phase(waveform)
    #           masks = self.inner(mag.unsqueeze(1))
    #           stem_mags = masks * mag.unsqueeze(1)
    #           return torch.stack([
    #               istft_from_mag_phase(stem_mags[:, i], phase) for i in range(masks.shape[1])
    #           ], dim=1)
    #
    # Wrap before conversion below. Adjust MODEL_CHUNK_SAMPLES /
    # MODEL_STEM_COUNT in TFLiteStemSeparator.kt to match this model's
    # actual chunk size and stem count once converted.

    example_input = (torch.randn(1, args.chunk_samples),)

    edge_model = ai_edge_torch.convert(model, example_input)
    edge_model.export(args.output)
    print(f"Wrote {args.output}")
    print(
        "Now copy this file to app/src/main/assets/models/, update "
        "modelAssetPath in TFLiteStemSeparator.kt if the filename differs, "
        "and verify MODEL_CHUNK_SAMPLES / MODEL_STEM_COUNT match this model."
    )


if __name__ == "__main__":
    main()
