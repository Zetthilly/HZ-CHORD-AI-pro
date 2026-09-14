"""
Trains NoteDetector on the synthetic arpeggio dataset.

Usage:
    python generate_synthetic_arpeggios.py --output_dir data/arpeggios --examples_per_combo 20
    python train.py --data_dir data/arpeggios --epochs 30 --output checkpoints/note_detector.pt
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split

from dataset import ArpeggioDataset
from model import NoteDetector


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_frames, correct_frames = 0.0, 0, 0
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for mel, labels in loader:
            mel, labels = mel.to(device), labels.to(device)
            logits = model(mel)
            loss = criterion(logits, labels)
            total_loss += loss.item() * mel.size(0)

            predictions = (torch.sigmoid(logits) > 0.5).float()
            # "Frame exactly right" = every note's on/off state matches — a strict metric,
            # useful for tracking progress but not the only thing worth watching (see README).
            frame_correct = (predictions == labels).all(dim=-1)
            correct_frames += frame_correct.sum().item()
            total_frames += frame_correct.numel()
    model.train()
    return total_loss / len(loader.dataset), correct_frames / max(total_frames, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="checkpoints/note_detector.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    full_dataset = ArpeggioDataset(args.data_dir)
    val_size = int(len(full_dataset) * args.val_fraction)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = NoteDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    criterion = torch.nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * mel.size(0)

        train_loss = running_loss / len(train_set)
        val_loss, frame_accuracy = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| exact_frame_accuracy {frame_accuracy:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            print(f"  -> saved new best checkpoint to {args.output}")

    print("Training complete.")


if __name__ == "__main__":
    main()
