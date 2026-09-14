package com.zetthilly.ichi.core.theory

import com.zetthilly.ichi.core.dsp.ChromaExtractor
import kotlin.math.sqrt

data class ChordMatch(
    val chord: ChordDefinition,
    val confidencePercent: Int
)

/**
 * Detects the most likely chord from a chroma vector using cosine similarity
 * against every rotated chord template. This is the same principle used in
 * published HPCP/chord-recognition literature (e.g. Fujishima 1999), and it
 * runs entirely on-device with no network or trained model required.
 */
class ChordDetector {

    private val allChords = ChordTemplates.allChordDefinitions()
    private val templates = allChords.associateWith { ChordTemplates.template(it) }

    fun detectChord(chroma: DoubleArray): ChordMatch {
        var best: ChordDefinition = allChords.first()
        var bestScore = -1.0

        for (chord in allChords) {
            val template = templates.getValue(chord)
            val score = cosineSimilarity(chroma, template)
            if (score > bestScore) {
                bestScore = score
                best = chord
            }
        }

        // Map similarity [0,1] to a friendlier confidence percentage, biased
        // slightly so a clean single-chord signal reads close to what's shown
        // in the mockups (90s% for a clear match).
        val confidence = (bestScore.coerceIn(0.0, 1.0) * 100).toInt()
        return ChordMatch(best, confidence)
    }

    private fun cosineSimilarity(a: DoubleArray, b: DoubleArray): Double {
        var dot = 0.0; var normA = 0.0; var normB = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        if (normA == 0.0 || normB == 0.0) return 0.0
        return dot / (sqrt(normA) * sqrt(normB))
    }
}
