package com.zetthilly.ichi.core.theory

data class ChordDefinition(
    val root: Int,           // 0=C ... 11=B
    val quality: String,     // "maj", "min", "maj7", "min7", "dom7", "dim", "aug", "sus2", "sus4"
    val intervals: IntArray  // semitone offsets from root, e.g. major = [0,4,7]
) {
    val name: String get() = ChordTemplates.NOTE_NAMES[root] + ChordTemplates.QUALITY_SUFFIX.getValue(quality)
}

/**
 * Binary chroma templates for common chord qualities. Chord detection works by
 * rotating each template through all 12 roots and correlating against the
 * observed chroma vector (cosine similarity) — the best match wins.
 */
object ChordTemplates {
    val NOTE_NAMES = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

    val QUALITY_SUFFIX = mapOf(
        "maj" to "", "min" to "m", "maj7" to "maj7", "min7" to "m7",
        "dom7" to "7", "dim" to "dim", "aug" to "aug", "sus2" to "sus2", "sus4" to "sus4"
    )

    // Interval sets relative to root (semitones)
    val QUALITY_INTERVALS: Map<String, IntArray> = mapOf(
        "maj" to intArrayOf(0, 4, 7),
        "min" to intArrayOf(0, 3, 7),
        "maj7" to intArrayOf(0, 4, 7, 11),
        "min7" to intArrayOf(0, 3, 7, 10),
        "dom7" to intArrayOf(0, 4, 7, 10),
        "dim" to intArrayOf(0, 3, 6),
        "aug" to intArrayOf(0, 4, 8),
        "sus2" to intArrayOf(0, 2, 7),
        "sus4" to intArrayOf(0, 5, 7)
    )

    /** All 12 roots x all qualities = the full search space for chord detection. */
    fun allChordDefinitions(): List<ChordDefinition> =
        (0 until 12).flatMap { root ->
            QUALITY_INTERVALS.map { (quality, intervals) ->
                ChordDefinition(root, quality, intervals)
            }
        }

    /** Builds a 12-bin chroma template for a given chord, root-rotated. */
    fun template(chord: ChordDefinition): DoubleArray {
        val vec = DoubleArray(12)
        for (interval in chord.intervals) {
            vec[(chord.root + interval) % 12] = 1.0
        }
        return vec
    }
}
