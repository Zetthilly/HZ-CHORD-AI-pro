"""Train a compact 4-stem waveform separator for HZ-CHORD-AI.

The exported model is intentionally waveform-in / waveform-out so it matches
feature-stems/TFLiteStemSeparator.kt exactly:
    input  [1, 352800]
    output [1, 4, 352800]

MUSDB18 provides the four targets: drums, bass, other, vocals.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

STEMS = ["drums", "bass", "other", "vocals"]
SAMPLE_RATE = 44_100
SEGMENT_SECONDS = 8
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS  # 352800; Android contract


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightweightSeparatorUNet(nn.Module):
    """Small waveform U-Net: mono mixture -> 4 mono stems.

    It uses only standard 1-D convolutions, ReLU, GroupNorm and interpolation,
    which keeps the graph much friendlier to PyTorch -> LiteRT conversion than
    the previous STFT/iSTFT-in-the-model design.
    """

    def __init__(self, n_stems: int = 4, base_channels: int = 8):
        super().__init__()
        self.enc1 = ConvBlock1D(1, base_channels)
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, 8, stride=2, padding=3)
        self.enc2 = ConvBlock1D(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 4, 8, stride=2, padding=3)
        self.enc3 = ConvBlock1D(base_channels * 4, base_channels * 4)
        self.down3 = nn.Conv1d(base_channels * 4, base_channels * 8, 8, stride=2, padding=3)
        self.bottleneck = ConvBlock1D(base_channels * 8, base_channels * 8)

        self.dec3 = ConvBlock1D(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock1D(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock1D(base_channels * 2 + base_channels, base_channels)
        self.head = nn.Conv1d(base_channels, n_stems, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[-1], mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, N]
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        d3 = self.dec3(torch.cat([self._up(b, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        # tanh bounds the generated audio and avoids runaway output levels.
        return torch.tanh(self.head(d1))


class MusdbSegmentDataset(Dataset):
    """Random fixed-length segments decoded lazily from MUSDB18."""

    def __init__(
        self,
        musdb_root: str,
        subset: str = "train",
        segments_per_track: int = 20,
        max_tracks: int = 0,
    ):
        import musdb

        self.mus = musdb.DB(root=musdb_root, subsets=subset, is_wav=False)
        tracks = list(self.mus.tracks)
        if max_tracks > 0:
            tracks = tracks[:max_tracks]
        if not tracks:
            raise RuntimeError(f"No MUSDB tracks found in {musdb_root!r} subset={subset!r}")
        self.tracks = tracks
        self.segments_per_track = segments_per_track
        self.segment_len = SEGMENT_SAMPLES

    def __len__(self) -> int:
        return len(self.tracks) * self.segments_per_track

    def __getitem__(self, idx: int):
        track = self.tracks[idx % len(self.tracks)]
        total = int(track.audio.shape[0])
        max_start = max(0, total - self.segment_len)
        start = random.randint(0, max_start) if max_start else 0
        end = start + self.segment_len

        mixture = np.asarray(track.audio[start:end], dtype=np.float32)
        mixture = mixture.mean(axis=1) if mixture.ndim == 2 else mixture

        stem_arrays = []
        for name in STEMS:
            audio = np.asarray(track.targets[name].audio[start:end], dtype=np.float32)
            audio = audio.mean(axis=1) if audio.ndim == 2 else audio
            if len(audio) < self.segment_len:
                audio = np.pad(audio, (0, self.segment_len - len(audio)))
            stem_arrays.append(audio[: self.segment_len])

        if len(mixture) < self.segment_len:
            mixture = np.pad(mixture, (0, self.segment_len - len(mixture)))
        mixture = mixture[: self.segment_len]
        return torch.from_numpy(mixture), torch.from_numpy(np.stack(stem_arrays, axis=0))


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cheap spectral magnitude loss used alongside waveform L1."""
    n_fft = 1024
    hop = 256
    window = torch.hann_window(n_fft, device=pred.device)
    pred_s = torch.stft(pred.reshape(-1, pred.shape[-1]), n_fft=n_fft, hop_length=hop,
                        window=window, return_complex=True).abs()
    tgt_s = torch.stft(target.reshape(-1, target.shape[-1]), n_fft=n_fft, hop_length=hop,
                        window=window, return_complex=True).abs()
    return F.l1_loss(torch.log1p(pred_s), torch.log1p(tgt_s))


def save_checkpoint(path: str, model, optimizer, scheduler, epoch, best_loss):
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_loss": best_loss,
        "stems": STEMS,
        "sample_rate": SAMPLE_RATE,
        "segment_samples": SEGMENT_SAMPLES,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_loss", float("inf")))


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This workflow is intentionally GPU-only.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

    dataset = MusdbSegmentDataset(
        args.musdb_root,
        subset="train",
        segments_per_track=args.segments_per_track,
        max_tracks=args.max_tracks,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    model = LightweightSeparatorUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)

    start_epoch = 0
    best_loss = float("inf")
    if args.resume_from and Path(args.resume_from).exists():
        start_epoch, best_loss = load_checkpoint(args.resume_from, model, optimizer, scheduler, device)
        print(f"Resumed from epoch {start_epoch}; best loss={best_loss:.6f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        steps = 0
        for mixture, stems in loader:
            mixture = mixture.to(device, non_blocking=True).unsqueeze(1)
            stems = stems.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not args.no_amp):
                pred = model(mixture)
                wave = F.l1_loss(pred, stems)
                spec = spectral_loss(pred, stems)
                loss = wave + args.spectral_weight * spec

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().item())
            steps += 1

        scheduler.step()
        avg = total / max(1, steps)
        print(f"Epoch {epoch + 1}/{args.epochs} loss={avg:.6f} lr={scheduler.get_last_lr()[0]:.3e}")

        save_checkpoint(args.checkpoint_path, model, optimizer, scheduler, epoch + 1, min(best_loss, avg))
        if avg < best_loss:
            best_loss = avg
            best_path = str(Path(args.checkpoint_path).with_name("lightweight_unet_best.pt"))
            save_checkpoint(best_path, model, optimizer, scheduler, epoch + 1, best_loss)
            print(f"Saved new best checkpoint: {best_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--musdb-root", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--segments-per-track", type=int, default=20)
    p.add_argument("--max-tracks", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--spectral-weight", type=float, default=0.25)
    p.add_argument("--checkpoint-path", default="lightweight_unet.pt")
    p.add_argument("--resume-from", default="")
    p.add_argument("--no-amp", action="store_true")
    train(p.parse_args())
