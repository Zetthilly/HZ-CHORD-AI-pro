package com.zetthilly.ichi.core.dsp

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Radix-2 Cooley-Tukey FFT, computed in place on parallel real/imag arrays.
 * `real.size` must be a power of two. This is the base primitive every
 * other analysis module (chroma, pitch, tempo) builds on top of.
 */
object FFT {

    fun transform(real: DoubleArray, imag: DoubleArray) {
        val n = real.size
        require(n and (n - 1) == 0) { "FFT size must be a power of two, got $n" }
        if (n <= 1) return

        // Bit-reversal permutation
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                val tr = real[i]; real[i] = real[j]; real[j] = tr
                val ti = imag[i]; imag[i] = imag[j]; imag[j] = ti
            }
        }

        // Iterative butterfly
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wRealStep = cos(ang)
            val wImagStep = sin(ang)
            var i = 0
            while (i < n) {
                var wr = 1.0
                var wi = 0.0
                for (k in 0 until len / 2) {
                    val uRe = real[i + k]
                    val uIm = imag[i + k]
                    val vRe = real[i + k + len / 2] * wr - imag[i + k + len / 2] * wi
                    val vIm = real[i + k + len / 2] * wi + imag[i + k + len / 2] * wr

                    real[i + k] = uRe + vRe
                    imag[i + k] = uIm + vIm
                    real[i + k + len / 2] = uRe - vRe
                    imag[i + k + len / 2] = uIm - vIm

                    val nextWr = wr * wRealStep - wi * wImagStep
                    val nextWi = wr * wImagStep + wi * wRealStep
                    wr = nextWr
                    wi = nextWi
                }
                i += len
            }
            len = len shl 1
        }
    }

    /** Magnitude spectrum (length n/2 + 1) of a real-valued windowed frame. Pads/truncates to next pow2. */
    fun magnitudeSpectrum(samples: FloatArray): DoubleArray {
        val n = nextPowerOfTwo(samples.size)
        val real = DoubleArray(n)
        val imag = DoubleArray(n)
        for (i in samples.indices) real[i] = samples[i].toDouble()
        transform(real, imag)
        val half = n / 2 + 1
        return DoubleArray(half) { i -> kotlin.math.hypot(real[i], imag[i]) }
    }

    fun nextPowerOfTwo(n: Int): Int {
        var p = 1
        while (p < n) p = p shl 1
        return p
    }
}
