"""
Convert a trained NoteDetector PyTorch checkpoint to LiteRT/TFLite.

The project previously used the deprecated ai-edge-torch package.
The package has now been renamed to litert-torch.

Install:

    pip install litert-torch

Usage:

    python3 export_to_tflite.py \
        --checkpoint checkpoints/note_detector.pt \
        --output note_detector.tflite
"""

import argparse
import os

import torch
import litert_torch

from model import NoteDetector, N_MELS
from dataset import CROP_FRAMES


def load_checkpoint(model, checkpoint_path):
    """
    Load a PyTorch checkpoint safely.

    The training code currently saves model.state_dict(), but this
    function also handles a checkpoint dictionary containing a
    'state_dict' key so the exporter is more tolerant of future
    training changes.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle checkpoints saved from DataParallel/DDP.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert NoteDetector PyTorch checkpoint to LiteRT/TFLite."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained PyTorch checkpoint.",
    )

    parser.add_argument(
        "--output",
        default="note_detector.tflite",
        help="Output .tflite file.",
    )

    parser.add_argument(
        "--crop-frames",
        type=int,
        default=CROP_FRAMES,
        help=(
            "Number of log-mel time frames used by the model. "
            "Must match the Android inference pipeline."
        ),
    )

    args = parser.parse_args()

    if args.crop_frames <= 0:
        raise ValueError(
            "--crop-frames must be greater than zero."
        )

    output_dir = os.path.dirname(
        os.path.abspath(args.output)
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print("==============================================")
    print("HZ CHORD AI - Note Detector TFLite Export")
    print("==============================================")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {args.output}")
    print(f"N_MELS:     {N_MELS}")
    print(f"Frames:     {args.crop_frames}")

    # ---------------------------------------------------------
    # Build model.
    # ---------------------------------------------------------
    model = NoteDetector()

    # ---------------------------------------------------------
    # Load trained weights.
    # ---------------------------------------------------------
    print("Loading checkpoint...")

    load_checkpoint(
        model,
        args.checkpoint,
    )

    model.eval()

    # ---------------------------------------------------------
    # Create representative/example input.
    #
    # Model input:
    #
    #     [batch, channels, mel_bins, frames]
    #
    #     [1, 1, 128, 128]
    # ---------------------------------------------------------
    example_input = torch.randn(
        1,
        1,
        N_MELS,
        args.crop_frames,
        dtype=torch.float32,
    )

    print(
        "Example input shape:",
        tuple(example_input.shape),
    )

    # ---------------------------------------------------------
    # Verify the PyTorch model before conversion.
    # ---------------------------------------------------------
    print("Running PyTorch forward-pass check...")

    with torch.no_grad():
        pytorch_output = model(
            example_input
        )

    expected_output_shape = (
        1,
        args.crop_frames,
        48,
    )

    print(
        "PyTorch output shape:",
        tuple(pytorch_output.shape),
    )

    if tuple(pytorch_output.shape) != expected_output_shape:
        raise RuntimeError(
            "Unexpected model output shape. "
            f"Expected {expected_output_shape}, "
            f"got {tuple(pytorch_output.shape)}."
        )

    # ---------------------------------------------------------
    # Convert using the current LiteRT Torch API.
    #
    # litert_torch.convert() returns a converted model whose
    # export() method writes the TFLite/LiteRT flatbuffer.
    # ---------------------------------------------------------
    print("Converting PyTorch model to LiteRT/TFLite...")

    edge_model = litert_torch.convert(
        model,
        (example_input,),
    )

    print("Conversion successful.")

    # ---------------------------------------------------------
    # Export .tflite file.
    # ---------------------------------------------------------
    print("Writing TFLite model...")

    edge_model.export(
        args.output
    )

    # ---------------------------------------------------------
    # Verify output file.
    # ---------------------------------------------------------
    if not os.path.exists(args.output):
        raise RuntimeError(
            "LiteRT conversion completed but the output "
            f"file was not created: {args.output}"
        )

    output_size = os.path.getsize(
        args.output
    )

    if output_size <= 0:
        raise RuntimeError(
            f"Generated TFLite file is empty: {args.output}"
        )

    print()
    print("==============================================")
    print("EXPORT SUCCESSFUL")
    print("==============================================")
    print(
        f"TFLite file: {args.output}"
    )
    print(
        f"File size:   {output_size / (1024 * 1024):.2f} MB"
    )
    print()
    print(
        "Input shape:"
        f" [1, 1, {N_MELS}, {args.crop_frames}]"
    )
    print(
        "Output shape:"
        f" [1, {args.crop_frames}, 48]"
    )
    print()
    print(
        "Output values are note-activation logits."
    )
    print(
        "Apply sigmoid on the Android/Kotlin side."
    )
    print()
    print(
        "Copy the generated model to:"
    )
    print(
        "app/src/main/assets/models/note_detector.tflite"
    )


if __name__ == "__main__":
    main()
