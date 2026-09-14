package com.zetthilly.ichi.core.audio

import android.Manifest
import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.annotation.RequiresPermission
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.Dispatchers

/**
 * Wraps AudioRecord to stream raw PCM frames for on-device analysis.
 * Everything downstream (chord detection, tuner, BPM, etc.) consumes
 * FloatArray frames from this Flow — no cloud upload, ever.
 */
class AudioEngine(
    private val sampleRate: Int = 44100,
    private val frameSize: Int = 4096 // ~93ms at 44.1kHz — good balance of frequency + time resolution
) {

    @SuppressLint("MissingPermission")
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun pcmFrames(): Flow<FloatArray> = callbackFlow {
        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val bufferSize = maxOf(minBufferSize, frameSize * 2)

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            close(IllegalStateException("AudioRecord failed to initialize"))
            return@callbackFlow
        }

        recorder.startRecording()
        val shortBuffer = ShortArray(frameSize)

        val thread = Thread {
            while (isActive) {
                val read = recorder.read(shortBuffer, 0, frameSize)
                if (read > 0) {
                    val floatFrame = FloatArray(read) { i -> shortBuffer[i] / 32768.0f }
                    trySend(floatFrame)
                }
            }
        }
        thread.start()

        awaitClose {
            recorder.stop()
            recorder.release()
        }
    }.flowOn(Dispatchers.Default)

    companion object {
        const val DEFAULT_SAMPLE_RATE = 44100
    }
}
