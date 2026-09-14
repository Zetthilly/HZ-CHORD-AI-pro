package com.zetthilly.ichi.core.dsp

import kotlin.math.PI
import kotlin.math.cos

/** Applies a Hann window in place — standard for chord/pitch analysis frames. */
object Window {
    fun hann(frame: FloatArray): FloatArray {
        val n = frame.size
        val out = FloatArray(n)
        for (i in 0 until n) {
            val w = 0.5 * (1.0 - cos(2.0 * PI * i / (n - 1)))
            out[i] = (frame[i] * w).toFloat()
        }
        return out
    }
}
