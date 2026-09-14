package com.zetthilly.ichi.core.dsp

import kotlin.math.log2
import kotlin.math.round

data class VoicedNote(
    val noteName: String,
    val octave: Int,
    val frequencyHz: Double,
    val magnitude: Double
) {
    val pitchClass: Int get() = ChordTemplateNoteIndex.indexOf(noteName)
}

private val ChordTemplateNoteIndex = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

/**
 * Detects multiple simultaneous pitches in a single frame via spectral
 * peak-picking, with harmonic suppression so a single played note's
 * overtones don't get reported as separate notes. This is what backs
 * "Note-by-Note" mode — it's a real, working heuristic (no ML), though
 * it's less precise on dense/noisy mixes than a trained polyphonic model
 * would be. Good enough for isolated instruments or a single stem.
 */
class PolyphonicNoteDetector(
    private val sampleRate: Int,
    private val maxNotes: Int = 6,
    private val magnitudeThresholdRatio: Double = 0.15, // relative to the loudest peak in the frame
    private val harmonicToleranceCents: Double = 35.0
) {

    fun detect(frame: FloatArray): List<VoicedNote> {
        val windowed = Window.hann(frame)
        val spectrum = FFT.magnitudeSpectrum(windowed)
        val fftSize = FFT.nextPowerOfTwo(frame.size)
        val binHz = sampleRate.toDouble() / fftSize

        val minBin = (40.0 / binHz).toInt().coerceAtLeast(1)   // ~E1, below typical instrument range
        val maxBin = (2000.0 / binHz).toInt().coerceAtMost(spectrum.size - 2)

        // 1. Find local maxima in the spectrum within range.
        val peaks = mutableListOf<Pair<Int, Double>>() // bin, magnitude
        for (bin in minBin..maxBin) {
            val mag = spectrum[bin]
            if (mag > spectrum[bin - 1] && mag > spectrum[bin + 1]) {
                peaks.add(bin to mag)
            }
        }
        if (peaks.isEmpty()) return emptyList()

        val loudest = peaks.maxOf { it.second }
        if (loudest <= 0.0) return emptyList()

        val candidates = peaks
            .filter { it.second >= loudest * magnitudeThresholdRatio }
            .sortedByDescending { it.second }

        // 2. Harmonic suppression: accept peaks strongest-first, reject any
        // later candidate that's a near-integer multiple of an already-accepted
        // fundamental (i.e. an overtone of a note we already picked).
        val acceptedFreqs = mutableListOf<Double>()
        val accepted = mutableListOf<Pair<Double, Double>>() // freq, magnitude

        for ((bin, mag) in candidates) {
            if (accepted.size >= maxNotes) break
            val freq = bin * binHz
            val isHarmonicOfExisting = acceptedFreqs.any { fundamental ->
                isNearHarmonic(freq, fundamental, harmonicToleranceCents)
            }
            if (!isHarmonicOfExisting) {
                acceptedFreqs.add(freq)
                accepted.add(freq to mag)
            }
        }

        return accepted
            .map { (freq, mag) -> frequencyToVoicedNote(freq, mag) }
            .sortedBy { it.frequencyHz }
    }

    private fun isNearHarmonic(freq: Double, fundamental: Double, toleranceCents: Double): Boolean {
        if (freq <= fundamental) return false
        val ratio = freq / fundamental
        val nearestHarmonic = round(ratio)
        if (nearestHarmonic < 2.0) return false
        val centsOff = 1200.0 * log2(ratio / nearestHarmonic)
        return kotlin.math.abs(centsOff) <= toleranceCents
    }

    private fun frequencyToVoicedNote(freq: Double, magnitude: Double): VoicedNote {
        val semitonesFromA4 = 12.0 * log2(freq / 440.0)
        val nearestSemitone = round(semitonesFromA4).toInt()
        val noteIndex = ((nearestSemitone % 12) + 12 + 9) % 12
        val octave = 4 + (nearestSemitone + 9).let { if (it < 0) (it - 11) / 12 else it / 12 }
        return VoicedNote(ChordTemplateNoteIndex[noteIndex], octave, freq, magnitude)
    }
}
