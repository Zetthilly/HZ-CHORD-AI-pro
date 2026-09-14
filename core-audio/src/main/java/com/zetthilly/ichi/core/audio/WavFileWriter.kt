package com.zetthilly.ichi.core.audio

import java.io.File
import java.io.RandomAccessFile
import kotlin.math.roundToInt

/**
 * Writes a mono float PCM buffer ([-1.0, 1.0]) to a standard 16-bit PCM WAV
 * file. This is what backs "Save to Device" for an individual stem — a real,
 * playable .wav file on disk, independent of the in-memory hand-off bus.
 */
object WavFileWriter {

    fun write(samples: FloatArray, sampleRate: Int, outputFile: File) {
        outputFile.parentFile?.mkdirs()

        val channels = 1
        val bitsPerSample = 16
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8
        val dataSize = samples.size * 2 // 2 bytes per 16-bit sample
        val riffSize = 36 + dataSize

        RandomAccessFile(outputFile, "rw").use { raf ->
            raf.setLength(0) // overwrite if it already exists

            // RIFF header
            raf.writeAsciiBytes("RIFF")
            raf.writeLittleEndianInt(riffSize)
            raf.writeAsciiBytes("WAVE")

            // fmt subchunk
            raf.writeAsciiBytes("fmt ")
            raf.writeLittleEndianInt(16) // PCM fmt chunk size
            raf.writeLittleEndianShort(1) // audio format: 1 = PCM
            raf.writeLittleEndianShort(channels.toShort())
            raf.writeLittleEndianInt(sampleRate)
            raf.writeLittleEndianInt(byteRate)
            raf.writeLittleEndianShort(blockAlign.toShort())
            raf.writeLittleEndianShort(bitsPerSample.toShort())

            // data subchunk
            raf.writeAsciiBytes("data")
            raf.writeLittleEndianInt(dataSize)

            val pcm16 = ByteArray(dataSize)
            var offset = 0
            for (sample in samples) {
                val clamped = sample.coerceIn(-1.0f, 1.0f)
                val intSample = (clamped * 32767.0f).roundToInt().toShort()
                pcm16[offset] = (intSample.toInt() and 0xFF).toByte()
                pcm16[offset + 1] = ((intSample.toInt() shr 8) and 0xFF).toByte()
                offset += 2
            }
            raf.write(pcm16)
        }
    }

    private fun RandomAccessFile.writeAsciiBytes(text: String) = write(text.toByteArray(Charsets.US_ASCII))

    private fun RandomAccessFile.writeLittleEndianInt(value: Int) {
        write(byteArrayOf(
            (value and 0xFF).toByte(),
            ((value shr 8) and 0xFF).toByte(),
            ((value shr 16) and 0xFF).toByte(),
            ((value shr 24) and 0xFF).toByte()
        ))
    }

    private fun RandomAccessFile.writeLittleEndianShort(value: Short) {
        val v = value.toInt()
        write(byteArrayOf((v and 0xFF).toByte(), ((v shr 8) and 0xFF).toByte()))
    }
}
