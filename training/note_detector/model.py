"""
A small, fully-convolutional multi-pitch detector: log-mel spectrogram in,
per-frame note-activation logits out. Deliberately avoids RNN/LSTM layers —
dilated 1D convolutions give temporal context instead, because recurrent
layers are the single most common cause of painful/failed TFLite conversion
(dynamic shapes, stateful ops). This is a "frame activation" model — for
each time frame, which of NUM_NOTES pitches are sounding — matching what
PolyphonicNoteDetector already outputs on-device, so it's a drop-in
upgrade path, not a redesign of the app's data flow.
"""

import torch
import torch.nn as nn

N_MELS = 128
NUM_NOTES = 48  # must match NOTE_MAX - NOTE_MIN + 1 in generate_synthetic_arpeggios.py


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))  # halves frequency resolution, keeps time resolution intact

    def forward(self, x):
        return self.pool(torch.relu(self.bn(self.conv(x))))


class DilatedTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = dilation  # kernel_size=3 with this padding keeps sequence length constant
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x))) + x  # residual connection


class NoteDetector(nn.Module):
    def __init__(self, n_mels: int = N_MELS, num_notes: int = NUM_NOTES):
        super().__init__()
        self.conv_stack = nn.Sequential(
            ConvBlock(1, 32),    # n_mels -> n_mels/2
            ConvBlock(32, 64),   # -> n_mels/4
            ConvBlock(64, 128),  # -> n_mels/8
        )
        freq_after_pools = n_mels // 8
        self.freq_collapse = nn.Conv2d(128, 128, kernel_size=(freq_after_pools, 1))  # squeezes frequency to 1

        self.temporal_stack = nn.Sequential(
            DilatedTemporalBlock(128, dilation=1),
            DilatedTemporalBlock(128, dilation=2),
            DilatedTemporalBlock(128, dilation=4),
            DilatedTemporalBlock(128, dilation=8),
        )

        self.classifier = nn.Conv1d(128, num_notes, kernel_size=1)

    def forward(self, x):
        # x: [B, 1, N_MELS, T]
        x = self.conv_stack(x)               # [B, 128, N_MELS/8, T]
        x = self.freq_collapse(x)            # [B, 128, 1, T]
        x = x.squeeze(2)                     # [B, 128, T]
        x = self.temporal_stack(x)           # [B, 128, T]
        logits = self.classifier(x)          # [B, num_notes, T]
        return logits.permute(0, 2, 1)       # [B, T, num_notes] — matches label shape from dataset.py
