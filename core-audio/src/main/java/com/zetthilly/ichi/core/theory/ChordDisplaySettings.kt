package com.zetthilly.ichi.core.theory

import com.zetthilly.ichi.core.dsp.VoicedNote

/** Detection/display mode: full chord recognition vs. individual notes. */
enum class ChordDisplayMode { CHORD_MODE, NOTE_BY_NOTE_MODE }

/** Whether shown notes reflect the actual register played, or a canonical arrangement. */
enum class VoicingMode { GENERAL_VOICING, EXACT_VOICING }

/** Whether to show the full detected chord quality (7ths/extensions) or reduce to a plain triad. */
enum class ChordComplexity { SIMPLE_CHORD, COMPLETE_CHORD }

data class ChordAnalysisSettings(
    val displayMode: ChordDisplayMode = ChordDisplayMode.CHORD_MODE,
    val voicingMode: VoicingMode = VoicingMode.GENERAL_VOICING,
    val complexity: ChordComplexity = ChordComplexity.COMPLETE_CHORD
)

/**
 * Reduces a detected chord to a plain triad (root/3rd/5th) when Simple mode
 * is selected — drops 7ths and other extensions while keeping the chord's
 * essential quality (major/minor/dim/aug/sus).
 */
object ChordSimplifier {
    fun simplify(chord: ChordDefinition): ChordDefinition {
        // Every quality in ChordTemplates is defined root-first, so the first
        // three intervals are always the triad-defining tones.
        val triadIntervals = chord.intervals.take(3).toIntArray()
        val triadQuality = when (chord.quality) {
            "maj7", "dom7" -> "maj"
            "min7" -> "min"
            else -> chord.quality // dim, aug, sus2, sus4, maj, min already triads
        }
        return ChordDefinition(chord.root, triadQuality, triadIntervals)
    }
}

/**
 * Arranges a chord's notes for display, honoring the complexity and voicing
 * settings. In EXACT_VOICING mode it tries to match each required pitch
 * class to an actually-detected spectral peak (real octave as played); in
 * GENERAL_VOICING mode it lays notes out in a standard ascending arrangement
 * starting at octave 4, independent of what was actually played.
 */
object ChordArranger {

    fun arrange(
        rawChord: ChordDefinition,
        settings: ChordAnalysisSettings,
        detectedNotes: List<VoicedNote> = emptyList()
    ): List<VoicedNote> {
        val chord = if (settings.complexity == ChordComplexity.SIMPLE_CHORD) {
            ChordSimplifier.simplify(rawChord)
        } else rawChord

        val pitchClasses = chord.intervals.map { (chord.root + it) % 12 }

        return if (settings.voicingMode == VoicingMode.EXACT_VOICING && detectedNotes.isNotEmpty()) {
            arrangeExact(pitchClasses, detectedNotes)
        } else {
            arrangeGeneral(pitchClasses)
        }
    }

    private fun arrangeGeneral(pitchClasses: List<Int>): List<VoicedNote> {
        val notes = mutableListOf<VoicedNote>()
        var octave = 4
        var previousClass = -1
        for (pc in pitchClasses) {
            if (pc <= previousClass) octave += 1
            notes.add(VoicedNote(ChordTemplates.NOTE_NAMES[pc], octave, noteFrequency(pc, octave), magnitude = 1.0))
            previousClass = pc
        }
        return notes
    }

    private fun arrangeExact(pitchClasses: List<Int>, detectedNotes: List<VoicedNote>): List<VoicedNote> {
        val byPitchClass = detectedNotes.groupBy { it.pitchClass }
        val fallbackGeneral = arrangeGeneral(pitchClasses).associateBy { pc -> ChordTemplates.NOTE_NAMES.indexOf(pc.noteName) }

        return pitchClasses.map { pc ->
            val match = byPitchClass[pc]?.maxByOrNull { it.magnitude }
            match ?: fallbackGeneral.getValue(pc)
        }.sortedBy { it.octave * 12 + it.pitchClass }
    }

    private fun noteFrequency(pitchClass: Int, octave: Int): Double {
        val semitonesFromA4 = (octave - 4) * 12 + (pitchClass - 9)
        return 440.0 * Math.pow(2.0, semitonesFromA4 / 12.0)
    }
}
