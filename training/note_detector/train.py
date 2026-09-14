"""
Trains NoteDetector on the synthetic arpeggio dataset.

Usage:

    python generate_synthetic_arpeggios.py \
        --output_dir data/arpeggios \
        --examples_per_combo 20

    python train.py \
        --data_dir data/arpeggios \
        --epochs 30 \
        --output checkpoints/note_detector.pt
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split

from dataset import ArpeggioDataset
from model import NoteDetector


def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_frames = 0
    correct_frames = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for mel, labels in loader:
            mel = mel.to(device)
            labels = labels.to(device)

            logits = model(mel)

            loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * mel.size(0)
            )

            predictions = (
                torch.sigmoid(logits) > 0.5
            ).float()

            # A frame is considered exactly correct only when
            # every note's on/off state matches.
            frame_correct = (
                predictions == labels
            ).all(dim=-1)

            correct_frames += (
                frame_correct.sum().item()
            )

            total_frames += (
                frame_correct.numel()
            )

    model.train()

    average_loss = (
        total_loss
        / max(len(loader.dataset), 1)
    )

    frame_accuracy = (
        correct_frames
        / max(total_frames, 1)
    )

    return average_loss, frame_accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="checkpoints/note_detector.pt",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Select CPU/GPU automatically.
    # ---------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Training on {device}")

    # ---------------------------------------------------------
    # Load dataset.
    # ---------------------------------------------------------
    full_dataset = ArpeggioDataset(
        args.data_dir
    )

    dataset_size = len(full_dataset)

    if dataset_size < 2:
        raise RuntimeError(
            "The dataset must contain at least "
            "2 examples so that training and "
            "validation sets can be created."
        )

    # ---------------------------------------------------------
    # Create train/validation split.
    # ---------------------------------------------------------
    val_size = int(
        dataset_size * args.val_fraction
    )

    # Make sure the validation set is not empty.
    val_size = max(1, val_size)

    # Make sure at least one training example remains.
    if val_size >= dataset_size:
        val_size = dataset_size - 1

    train_size = (
        dataset_size - val_size
    )

    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
    )

    print(
        f"Dataset: {dataset_size} examples | "
        f"train: {train_size} | "
        f"validation: {val_size}"
    )

    # ---------------------------------------------------------
    # DataLoader.
    #
    # num_workers=0 is intentional.
    #
    # GitHub Actions CPU runners are more reliable with the
    # dataset executed in the main Python process, and this
    # gives much clearer tracebacks when an audio file fails.
    # ---------------------------------------------------------
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # ---------------------------------------------------------
    # Model.
    # ---------------------------------------------------------
    model = NoteDetector().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=3,
            factor=0.5,
        )
    )

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )

    # ---------------------------------------------------------
    # Prepare output directory.
    # ---------------------------------------------------------
    output_dir = os.path.dirname(
        args.output
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    best_val_loss = float("inf")

    # ---------------------------------------------------------
    # Training loop.
    # ---------------------------------------------------------
    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        running_loss = 0.0

        for mel, labels in train_loader:
            mel = mel.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(mel)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * mel.size(0)
            )

        train_loss = (
            running_loss
            / max(len(train_set), 1)
        )

        val_loss, frame_accuracy = evaluate(
            model,
            val_loader,
            device,
        )

        scheduler.step(val_loss)

        print(
            f"epoch {epoch:3d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"exact_frame_accuracy "
            f"{frame_accuracy:.3f}"
        )

        # -----------------------------------------------------
        # Save the best checkpoint.
        # -----------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                args.output,
            )

            print(
                f"  -> saved new best checkpoint "
                f"to {args.output}"
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
