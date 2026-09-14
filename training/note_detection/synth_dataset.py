"""
Generates synthetic labeled training data for pitch-class detection: mixes of
1-4 simultaneously-sounding notes (harmonics + noise, roughly instrument-like)
paired with the ground-truth set of active pitch classes.

This is a *synthetic* dataset — it teaches the model to disentangle harmonic
overtones from true fundamentals in a controlled setting, which is exactly
where naive peak-picking gets confused (a single note's 2nd/3rd harmonic can
look like a "different" pitch class). It is not a substitute for training on
real recorded arpeggios (see README.md) but is a legitimate, honest starting
point that runs anywhere, with no dataset download and no GPU.
"""
import numpy as np
from chroma_features import chroma_vector

SAMPLE_RATE = 44100
FRAME_SIZE = 4096
MIDI_MIN = 40  # ~E2
MIDI_MAX = 84  # ~C6
HARMONICS = 5
RNG_SEED = 42


def midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def synth_note(freq: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n_samples) / SAMPLE_RATE
    wave = np.zeros(n_samples)
    for h in range(1, HARMONICS + 1):
        # Decaying harmonic amplitudes + tiny random detune/phase, loosely
        # approximating a plucked/bowed/vocal-ish timbre rather than a pure tone.
        amp = (1.0 / h) * rng.uniform(0.7, 1.0)
        detune = rng.uniform(-0.003, 0.003)
        phase = rng.uniform(0, 2 * np.pi)
        wave += amp * np.sin(2 * np.pi * freq * h * (1 + detune) * t + phase)
    # Simple decay envelope so it's not a flat drone.
    envelope = np.exp(-t * rng.uniform(0.5, 2.5))
    return wave * envelope


def generate_example(rng: np.random.Generator):
    n_notes = rng.integers(1, 5)  # 1..4 simultaneous notes
    midi_notes = rng.choice(np.arange(MIDI_MIN, MIDI_MAX + 1), size=n_notes, replace=False)

    mixture = np.zeros(FRAME_SIZE)
    for midi in midi_notes:
        freq = midi_to_freq(int(midi))
        amp = rng.uniform(0.5, 1.0)
        mixture += amp * synth_note(freq, FRAME_SIZE, rng)

    # Background noise — real recordings are never silent otherwise.
    mixture += rng.normal(0, 0.01, FRAME_SIZE)

    peak = np.max(np.abs(mixture))
    if peak > 0:
        mixture /= peak

    label = np.zeros(12, dtype=np.float32)
    for midi in midi_notes:
        label[int(midi) % 12] = 1.0

    features = chroma_vector(mixture.astype(np.float32), SAMPLE_RATE)
    return features, label


def generate_dataset(n_examples: int, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_examples, 12), dtype=np.float32)
    y = np.zeros((n_examples, 12), dtype=np.float32)
    for i in range(n_examples):
        X[i], y[i] = generate_example(rng)
    return X, y


if __name__ == "__main__":
    X, y = generate_dataset(5)
    print("Sample chroma features:\n", X)
    print("Sample labels (active pitch classes):\n", y)
