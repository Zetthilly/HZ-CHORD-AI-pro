package com.zetthilly.ichi.core.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.PlaybackParams

data class PlaybackUiState(
    val isPlaying: Boolean = false,
    val positionMs: Long = 0,
    val durationMs: Long = 0,
    val speed: Float = 1.0f
)

/**
 * Wraps a single static AudioTrack so a fully-loaded-in-memory buffer (a
 * stem, an exported clip, a mic capture) can be played, paused, seeked, and
 * speed-adjusted. MODE_STATIC is what makes seeking reliable — the whole
 * buffer is written up front, and setPlaybackHeadPosition can jump anywhere
 * in it directly, no re-buffering required.
 */
class PcmTrackPlayer(private val sampleRate: Int) {

    private var track: AudioTrack? = null
    private var totalFrames: Int = 0

    fun load(samples: FloatArray) {
        release()
        totalFrames = samples.size

        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setSampleRate(sampleRate)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()

        val attributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
            .build()

        val bufferSizeBytes = samples.size * 4 // 4 bytes per float sample
        track = AudioTrack.Builder()
            .setAudioAttributes(attributes)
            .setAudioFormat(format)
            .setBufferSizeInBytes(bufferSizeBytes)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        track?.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
    }

    fun play() {
        track?.play()
    }

    fun pause() {
        track?.pause()
    }

    /** Stops, rewinds to a frame offset, and resumes playing only if it was already playing. */
    fun seekToMs(ms: Long) {
        val t = track ?: return
        val wasPlaying = t.playState == AudioTrack.PLAYSTATE_PLAYING
        val frame = ((ms / 1000.0) * sampleRate).toInt().coerceIn(0, totalFrames)
        t.pause()
        t.playbackHeadPosition = frame
        if (wasPlaying) t.play()
    }

    /** Standard varispeed: pitch moves with speed, same as changing turntable RPM. */
    fun setSpeed(speed: Float) {
        val t = track ?: return
        t.playbackParams = PlaybackParams().setSpeed(speed).setPitch(speed)
    }

    fun setVolume(volume: Float) {
        track?.setVolume(volume.coerceIn(0f, 1f))
    }

    fun positionMs(): Long {
        val t = track ?: return 0
        return (t.playbackHeadPosition.toLong() * 1000) / sampleRate
    }

    fun durationMs(): Long = (totalFrames.toLong() * 1000) / sampleRate

    fun isPlaying(): Boolean = track?.playState == AudioTrack.PLAYSTATE_PLAYING

    fun release() {
        track?.stop()
        track?.release()
        track = null
    }
}
