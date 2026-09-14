"""
Converts a trained NoteDetector checkpoint to TensorFlow Lite via ai-edge-torch.

Requires: torch, ai-edge-torch
    pip install torch ai-edge-torch

NOT executed in the sandbox that produced this file — no torch, no
ai-edge-torch there. Run this in the same environment where you trained
the model (train.py).

    python3 export_to_tflite.py --checkpoint checkpoints/note_detector.pt --output note_detector.tflite
"""
import argparse
import torch
import ai_edge_torch

from model import NoteDetector, N_MELS
from dataset import CROP_FRAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="note_detector.tflite")
    parser.add_argument(
        "--crop-frames", type=int, default=CROP_FRAMES,
        help="Must match the log-mel window size the Kotlin wrapper computes at inference time.",
    )
    args = parser.parse_args()

    model = NoteDetector()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    # Input shape: [1, 1, N_MELS, crop_frames] — a log-mel spectrogram window.
    example_input = (torch.randn(1, 1, N_MELS, args.crop_frames),)

    edge_model = ai_edge_torch.convert(model, example_input)
    edge_model.export(args.output)
    print(f"Wrote {args.output}")
    print(
        f"Input shape the .tflite model expects: [1, 1, {N_MELS}, {args.crop_frames}] "
        f"(a log-mel spectrogram window). Output: [1, {args.crop_frames}, NUM_NOTES] "
        f"per-frame note-activation logits (apply sigmoid on the Kotlin side)."
    )
    print(
        "Copy this file to app/src/main/assets/models/ and write a small Interpreter-based "
        "Kotlin wrapper (see this folder's README, 'After training: bring it into the app')."
    )


if __name__ == "__main__":
    main()
