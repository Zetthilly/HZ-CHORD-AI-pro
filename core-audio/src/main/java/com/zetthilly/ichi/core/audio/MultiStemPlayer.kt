package com.zetthilly.ichi.core.audio

/**
 * Plays several isolated stems back in sync under a single transport (one
 * play/pause/seek moves all of them together), while each stem's volume is
 * set independently — that's how mute/solo/volume sliders work without the
 * stems drifting out of alignment. Generic over string keys so it has no
 * dependency on any feature module's stem-type enum.
 */
class MultiStemPlayer(private val sampleRate: Int) {

    private val players = mutableMapOf<String, PcmTrackPlayer>()

    fun loadStems(stems: Map<String, FloatArray>) {
        releaseAll()
        stems.forEach { (key, samples) ->
            players[key] = PcmTrackPlayer(sampleRate).apply { load(samples) }
        }
    }

    fun playAll() = players.values.forEach { it.play() }
    fun pauseAll() = players.values.forEach { it.pause() }
    fun seekAllToMs(ms: Long) = players.values.forEach { it.seekToMs(ms) }
    fun setSpeedAll(speed: Float) = players.values.forEach { it.setSpeed(speed) }

    /** Sets one stem's gain directly — callers compute the effective value (mute/solo/volume combined). */
    fun setVolume(key: String, volume: Float) {
        players[key]?.setVolume(volume)
    }

    fun referencePositionMs(): Long = players.values.firstOrNull()?.positionMs() ?: 0
    fun durationMs(): Long = players.values.maxOfOrNull { it.durationMs() } ?: 0
    fun isPlaying(): Boolean = players.values.any { it.isPlaying() }

    fun releaseAll() {
        players.values.forEach { it.release() }
        players.clear()
    }
}
