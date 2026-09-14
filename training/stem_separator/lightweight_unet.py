"""
A compact spectrogram-masking U-Net for 4-stem source separation
(drums/bass/other/vocals) — much smaller than Demucs, meant as a genuine,
tractable from-scratch training project rather than a Demucs replacement.

Requires: torch, torchaudio, musdb, museval
    pip install torch torchaudio musdb museval

Requires the MUSDB18 dataset: https://sigsep.github.io/datasets/musdb.html
(150 songs, ~30GB, isolated stems). This script expects it unpacked at
--musdb-root with the standard musdb directory layout.

NOT executed in the sandbox that produced this file — no GPU, no torch, no
dataset available there. Run this on Colab or a machine with a CUDA GPU.

    python3 lightweight_unet.py --musdb-root /path/to/musdb18 --epochs 50
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

STEMS = ["drums", "bass", "other", "vocals"]
N_FFT = 2048
HOP_LENGTH = 512
SEGMENT_SECONDS = 6
SAMPLE_RATE = 44100


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LightweightSeparatorUNet(nn.Module):
    """
    Operates on a mono magnitude spectrogram and predicts one soft mask per
    stem (values in [0,1]); the mask is applied to the input spectrogram's
    magnitude, then combined with the original phase for reconstruction.
    This is the same general approach as Open-Unmix, at a smaller scale.
    """

    def __init__(self, n_stems: int = len(STEMS), base_channels: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(1, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.mask_head = nn.Conv2d(base_channels, n_stems, 1)

    def forward(self, spec_mag: torch.Tensor) -> torch.Tensor:
        # spec_mag: [batch, 1, freq_bins, time_frames]
        e1 = self.enc1(spec_mag)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        masks = torch.sigmoid(self.mask_head(d1))  # [batch, n_stems, freq, time]
        return masks


def stft_mag_phase(waveform: torch.Tensor):
    spec = torch.stft(
        waveform, n_fft=N_FFT, hop_length=HOP_LENGTH,
        window=torch.hann_window(N_FFT, device=waveform.device),
        return_complex=True,
    )
    return spec.abs(), torch.angle(spec)


def istft_from_mag_phase(mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    complex_spec = torch.polar(mag, phase)
    return torch.istft(
        complex_spec, n_fft=N_FFT, hop_length=HOP_LENGTH,
        window=torch.hann_window(N_FFT, device=mag.device),
    )


class MusdbSegmentDataset(torch.utils.data.Dataset):
    """Loads random SEGMENT_SECONDS-long mixture/stem pairs from MUSDB18 via the `musdb` package."""

    def __init__(self, musdb_root: str, subset: str = "train", segments_per_track: int = 20):
        import musdb
        self.mus = musdb.DB(root=musdb_root, subsets=subset)
        self.segments_per_track = segments_per_track
        self.segment_len = SEGMENT_SECONDS * SAMPLE_RATE

    def __len__(self):
        return len(self.mus.tracks) * self.segments_per_track

    def __getitem__(self, idx):
        track = self.mus.tracks[idx % len(self.mus.tracks)]
        max_start = max(0, track.audio.shape[0] - self.segment_len)
        start = torch.randint(0, max(1, max_start), (1,)).item()

        mixture = track.audio[start:start + self.segment_len].mean(axis=1)  # downmix to mono
        stems = [track.targets[name].audio[start:start + self.segment_len].mean(axis=1) for name in STEMS]

        mixture_t = torch.tensor(mixture, dtype=torch.float32)
        stems_t = torch.stack([torch.tensor(s, dtype=torch.float32) for s in stems])
        return mixture_t, stems_t


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    dataset = MusdbSegmentDataset(args.musdb_root, subset="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = LightweightSeparatorUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for mixture, stems in loader:
            mixture, stems = mixture.to(device), stems.to(device)

            mix_mag, mix_phase = stft_mag_phase(mixture)
            mix_mag_in = mix_mag.unsqueeze(1)  # [batch, 1, freq, time]

            masks = model(mix_mag_in)  # [batch, n_stems, freq, time]
            predicted_mags = masks * mix_mag.unsqueeze(1)

            target_mags = torch.stack([stft_mag_phase(stems[:, i])[0] for i in range(len(STEMS))], dim=1)

            loss = F.l1_loss(predicted_mags, target_mags)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} — avg L1 loss: {total_loss / len(loader):.4f}")
        torch.save(model.state_dict(), args.checkpoint_path)
        print(f"Saved checkpoint to {args.checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--musdb-root", required=True, help="Path to unpacked MUSDB18 dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-path", default="lightweight_unet.pt")
    args = parser.parse_args()
    train(args)
