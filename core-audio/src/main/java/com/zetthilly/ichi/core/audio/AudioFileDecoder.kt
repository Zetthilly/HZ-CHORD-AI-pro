package com.zetthilly.ichi.core.audio

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteOrder

/**
 * Decodes a user-picked audio file (mp3, m4a/aac, wav, ogg — whatever codecs
 * the device ships with) into mono 32-bit float PCM, resampled to
 * [targetSampleRate]. Uses Android's built-in MediaExtractor/MediaCodec —
 * no network, no third-party decoding library, fully on-device.
 *
 * This does real work (can take a moment for a long file) — call it from a
 * background dispatcher (e.g. Dispatchers.IO), never directly from the UI thread.
 */
object AudioFileDecoder {

    fun decodeToMonoPcm(context: Context, uri: Uri, targetSampleRate: Int = 44100): FloatArray? {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)

            var trackIndex = -1
            var format: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("audio/")) {
                    trackIndex = i
                    format = f
                    break
                }
            }
            if (trackIndex < 0 || format == null) return null

            extractor.selectTrack(trackIndex)
            val mime = format.getString(MediaFormat.KEY_MIME)!!
            val nativeSampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()

            val pcmChunks = mutableListOf<ShortArray>()
            val bufferInfo = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false

            while (!outputDone) {
                if (!inputDone) {
                    val inputBufferId = codec.dequeueInputBuffer(10_000)
                    if (inputBufferId >= 0) {
                        val inputBuffer = codec.getInputBuffer(inputBufferId)
                        if (inputBuffer != null) {
                            val sampleSize = extractor.readSampleData(inputBuffer, 0)
                            if (sampleSize < 0) {
                                codec.queueInputBuffer(inputBufferId, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                val presentationTimeUs = extractor.sampleTime
                                codec.queueInputBuffer(inputBufferId, 0, sampleSize, presentationTimeUs, 0)
                                extractor.advance()
                            }
                        }
                    }
                }

                val outputBufferId = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                if (outputBufferId >= 0) {
                    val outputBuffer = codec.getOutputBuffer(outputBufferId)
                    if (outputBuffer != null && bufferInfo.size > 0) {
                        outputBuffer.order(ByteOrder.LITTLE_ENDIAN)
                        outputBuffer.position(bufferInfo.offset)
                        outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                        val shorts = ShortArray(bufferInfo.size / 2)
                        outputBuffer.asShortBuffer().get(shorts)
                        pcmChunks.add(shorts)
                    }
                    codec.releaseOutputBuffer(outputBufferId, false)
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        outputDone = true
                    }
                }
            }

            val totalSamples = pcmChunks.sumOf { it.size }
            val interleaved = ShortArray(totalSamples)
            var pos = 0
            for (chunk in pcmChunks) {
                chunk.copyInto(interleaved, pos)
                pos += chunk.size
            }

            val mono = downmixToMono(interleaved, channelCount)
            val monoFloat = FloatArray(mono.size) { i -> mono[i] / 32768.0f }

            return if (nativeSampleRate == targetSampleRate) {
                monoFloat
            } else {
                LinearResampler.resample(monoFloat, nativeSampleRate, targetSampleRate)
            }
        } catch (e: Exception) {
            return null
        } finally {
            codec?.stop()
            codec?.release()
            extractor.release()
        }
    }

    private fun downmixToMono(interleaved: ShortArray, channelCount: Int): ShortArray {
        if (channelCount <= 1) return interleaved
        val frames = interleaved.size / channelCount
        val mono = ShortArray(frames)
        for (frame in 0 until frames) {
            var sum = 0
            for (ch in 0 until channelCount) sum += interleaved[frame * channelCount + ch]
            mono[frame] = (sum / channelCount).toInt().toShort()
        }
        return mono
    }
}

/** Adequate for bringing an arbitrary file's native sample rate to the app's working rate (44.1kHz). */
object LinearResampler {
    fun resample(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        if (fromRate == toRate || input.isEmpty()) return input
        val ratio = toRate.toDouble() / fromRate.toDouble()
        val outputLength = (input.size * ratio).toInt()
        val output = FloatArray(outputLength)
        for (i in output.indices) {
            val srcPos = i / ratio
            val srcIndex = srcPos.toInt()
            val frac = (srcPos - srcIndex).toFloat()
            val a = input.getOrElse(srcIndex) { input.last() }
            val b = input.getOrElse(srcIndex + 1) { input.last() }
            output[i] = a + (b - a) * frac
        }
        return output
    }
}
