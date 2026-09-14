package com.zetthilly.ichi.core.theory

import com.zetthilly.ichi.core.dsp.VoicedNote
import kotlin.math.pow

/**
 * Shifts detected chords and notes by a semitone offset for *display* only —
 * this relabels what's shown (chord name, keyboard highlighting) the way a
 * capo or "Transpose" button does in chord-chart apps. It does not alter the
 * underlying audio or its playback pitch; that's a separate, much heavier
 * DSP problem (real-time pitch-shifting) that isn't in scope here.
 */
object Transposer {

    fun transposeChord(chord: ChordDefinition, semitones: Int): ChordDefinition {
        if (semitones == 0) return chord
        val newRoot = ((chord.root + semitones) % 12 + 12) % 12
        return chord.copy(root = newRoot)
    }

    fun transposeNote(note: VoicedNote, semitones: Int): VoicedNote {
        if (semitones == 0) return note
        val originalPitchClass = ChordTemplates.NOTE_NAMES.indexOf(note.noteName)
        val totalSemitone = note.octave * 12 + originalPitchClass + semitones
        val newOctave = Math.floorDiv(totalSemitone, 12)
        val newPitchClass = ((totalSemitone % 12) + 12) % 12
        val newFrequency = note.frequencyHz * 2.0.pow(semitones / 12.0)
        return note.copy(
            noteName = ChordTemplates.NOTE_NAMES[newPitchClass],
            octave = newOctave,
            frequencyHz = newFrequency
        )
    }
}
