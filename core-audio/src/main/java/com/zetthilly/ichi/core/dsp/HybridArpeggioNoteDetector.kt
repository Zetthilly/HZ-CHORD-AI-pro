package com.zetthilly.ichi.core.dsp

import com.zetthilly.ichi.core.theory.TrainedPitchClassifier

/**
 * The model backing "Note-by-Note" mode's arpeggio picking. It's a hybrid,
 * and deliberately so:
 *
 * - [TrainedPitchClassifier] (a small MLP, genuinely trained — see
 *   /training/note_detection/ at the project root) decides WHICH pitch
 *   classes are actually sounding from the frame's chroma vector. This is
 *   exactly the part naive peak-picking struggles with: a single note's
 *   2nd/3rd harmonic can look like a "different" pitch class, and a trained
 *   classifier can learn to see through that in a way a fixed threshold can't.
 * - [PolyphonicNoteDetector] (unchanged, still spectral peak-picking) then
 *   supplies the specific octave/frequency for each pitch class the model
 *   confirmed — a simpler, more geometric problem that doesn't need learning.
 *
 * If a confirmed pitch class has no matching spectral peak (can happen on a
 * weak/masked note), it's dropped rather than guessed — better to show fewer,
 * higher-confidence notes while picking through an arpeggio than a wrong octave.
 */
class HybridArpeggioNoteDetector(sampleRate: Int) {

    private val chromaExtractor = ChromaExtractor(sampleRate)
    private val peakDetector = PolyphonicNoteDetector(sampleRate)

    fun detect(frame: FloatArray, confidenceThreshold: Float = 0.5f): List<VoicedNote> {
        val chroma = chromaExtractor.extract(frame)
        val confirmedPitchClasses = TrainedPitchClassifier.activePitchClasses(chroma, confidenceThreshold).toSet()
        if (confirmedPitchClasses.isEmpty()) return emptyList()

        val candidatePeaks = peakDetector.detect(frame)

        return confirmedPitchClasses
            .mapNotNull { pc -> candidatePeaks.filter { it.pitchClass == pc }.maxByOrNull { it.magnitude } }
            .sortedBy { it.frequencyHz }
    }
}
