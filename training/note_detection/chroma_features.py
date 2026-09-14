"""
Python port of core-audio's ChromaExtractor.kt — kept in lockstep with the
Kotlin version so training data and on-device inference see the same feature
distribution. If you change ChromaExtractor.kt, mirror the change here before
regenerating training data.
"""
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def hann_window(frame: np.ndarray) -> np.ndarray:
    n = len(frame)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))
    return frame * w


def chroma_vector(frame: np.ndarray, sample_rate: int) -> np.ndarray:
    windowed = hann_window(frame)
    fft_size = next_pow2(len(frame))
    spectrum = np.abs(np.fft.rfft(windowed, n=fft_size))

    bin_hz = sample_rate / fft_size
    min_bin = max(1, int(55.0 / bin_hz))
    max_bin = min(int(5000.0 / bin_hz), len(spectrum) - 1)

    chroma = np.zeros(12, dtype=np.float64)
    bins = np.arange(min_bin, max_bin + 1)
    freqs = bins * bin_hz
    mags = spectrum[bins]

    valid = mags > 0
    freqs, mags = freqs[valid], mags[valid]

    semitones_from_a4 = 12.0 * np.log2(freqs / 440.0)
    pitch_classes = (np.round(semitones_from_a4).astype(int) % 12 + 12 + 9) % 12

    for pc, mag in zip(pitch_classes, mags):
        chroma[pc] += np.log1p(mag)

    total = chroma.sum()
    if total > 0:
        chroma /= total
    return chroma
