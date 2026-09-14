"""
Chord definitions mirroring core-audio/theory/ChordTemplates.kt exactly, so
the synthetic training data covers precisely the chord vocabulary the app
already recognizes. If you add a quality to the Kotlin side, add it here too.
"""

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# semitone offsets from root — must match ChordTemplates.QUALITY_INTERVALS
QUALITY_INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dom7": [0, 4, 7, 10],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

ALL_QUALITIES = list(QUALITY_INTERVALS.keys())
ALL_ROOTS = list(range(12))  # 0=C .. 11=B


def chord_midi_notes(root_pitch_class: int, quality: str, base_octave_midi: int = 60) -> list[int]:
    """base_octave_midi=60 is MIDI C4. Returns absolute MIDI note numbers for one octave of the chord."""
    return [base_octave_midi + root_pitch_class + interval for interval in QUALITY_INTERVALS[quality]]
