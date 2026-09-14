package com.zetthilly.ichi.core.dsp

import kotlin.math.ln
import kotlin.math.log2
import kotlin.math.pow

/**
 * Converts a windowed audio frame into a 12-dimensional chroma vector
 * (relative energy of each pitch class C, C#, D, ... B), regardless of octave.
 * This is the standard front-end feature for chord recognition (HPCP-style).
 */
class ChromaExtractor(private val sampleRate: Int) {

    private val referenceA4 = 440.0
    private val minFreq = 55.0   // A1 — below this is mostly noise/rumble for our use case
    private val maxFreq = 5000.0 // covers guitar/piano/vocal harmonics relevant to chords

    /** Returns a 12-element DoubleArray, index 0 = C, 1 = C#, ... 11 = B, normalized to sum 1. */
    fun extract(frame: FloatArray): DoubleArray {
        val windowed = Window.hann(frame)
        val spectrum = FFT.magnitudeSpectrum(windowed)
        val fftSize = FFT.nextPowerOfTwo(frame.size)
        val chroma = DoubleArray(12)

        val binHz = sampleRate.toDouble() / fftSize
        val minBin = (minFreq / binHz).toInt().coerceAtLeast(1)
        val maxBin = (maxFreq / binHz).toInt().coerceAtMost(spectrum.size - 1)

        for (bin in minBin..maxBin) {
            val freq = bin * binHz
            val magnitude = spectrum[bin]
            if (magnitude <= 0.0) continue

            // Which pitch class does this frequency belong to?
            val semitoneFromA4 = 12.0 * log2(freq / referenceA4)
            var pitchClass = (Math.round(semitoneFromA4).toInt() % 12 + 12 + 9) % 12
            // +9 shifts so index 0 = C (A4 is pitch class 9 if C=0)

            // Weight by log-magnitude so loud low harmonics don't totally dominate
            chroma[pitchClass] += ln(1.0 + magnitude)
        }

        val sum = chroma.sum()
        return if (sum > 0.0) DoubleArray(12) { chroma[it] / sum } else chroma
    }

    companion object {
        val NOTE_NAMES = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    }
}
