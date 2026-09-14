package com.zetthilly.ichi.stems

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.zetthilly.ichi.core.audio.AudioResourceBus
import com.zetthilly.ichi.core.audio.ExportedAudioResource
import com.zetthilly.ichi.core.audio.MultiStemPlayer
import com.zetthilly.ichi.core.audio.PlaybackUiState
import com.zetthilly.ichi.core.audio.WavFileWriter
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File

enum class StemType { VOCALS, DRUMS, BASS, GUITAR, KEYS, OTHER }

private val PLAYBACK_SPEEDS = listOf(0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f)

data class StemChannel(
    val type: StemType,
    val label: String,
    val samples: FloatArray,
    val isMuted: Boolean = false,
    val isSoloed: Boolean = false,
    val volume: Float = 1.0f
)

data class StemSplitterUiState(
    val isSeparating: Boolean = false,
    val separationProgress: Float = 0f,
    val stems: List<StemChannel> = emptyList(),
    val playback: PlaybackUiState = PlaybackUiState(),
    val lastExportedLabel: String? = null,   // "sent to Chord Detector" confirmation
    val lastSavedFilePath: String? = null,   // "saved to device" confirmation
    val separationError: String? = null      // e.g. model asset not bundled yet
)

/**
 * Stem Splitter is fully standalone: it separates, plays back (all stems in
 * sync under one transport, via [MultiStemPlayer]), and mixes stems, and
 * knows nothing about the Chord Detector. Two one-way, opt-in bridges exist:
 * [exportStemToChordDetector] drops a stem into [AudioResourceBus] for
 * whichever screen wants to pick it up, and [saveStemToDevice] writes it to
 * disk as a real .wav file — neither path pulls Chord Detector's code in
 * here, and this screen never displays chord data.
 */
class StemSplitterViewModel(
    application: Application,
    private val sampleRate: Int = 44100,
    private val stemSeparator: StemSeparator = TFLiteStemSeparator(application)
) : AndroidViewModel(application) {

    private val multiStemPlayer = MultiStemPlayer(sampleRate)
    private var positionTickerRunning = false

    private val _uiState = MutableStateFlow(StemSplitterUiState())
    val uiState: StateFlow<StemSplitterUiState> = _uiState.asStateFlow()

    fun separate(inputSamples: FloatArray, sourceFileName: String = "Imported Audio") {
        _uiState.value = _uiState.value.copy(isSeparating = true, separationProgress = 0f, separationError = null)
        viewModelScope.launch {
            try {
                val stems = stemSeparator.separate(inputSamples, sampleRate) { progress ->
                    _uiState.value = _uiState.value.copy(separationProgress = progress)
                }
                val channels = listOf(
                    StemChannel(StemType.VOCALS, "Vocals", stems.vocals),
                    StemChannel(StemType.DRUMS, "Drums", stems.drums),
                    StemChannel(StemType.BASS, "Bass", stems.bass),
                    StemChannel(StemType.GUITAR, "Guitar", stems.guitar),
                    StemChannel(StemType.KEYS, "Keys", stems.keys),
                    StemChannel(StemType.OTHER, "Other", stems.other)
                )
                multiStemPlayer.loadStems(channels.associate { it.type.name to it.samples })
                channels.forEach { multiStemPlayer.setVolume(it.type.name, effectiveVolume(it, channels)) }

                _uiState.value = _uiState.value.copy(
                    isSeparating = false,
                    stems = channels,
                    playback = PlaybackUiState(durationMs = multiStemPlayer.durationMs())
                )
                startPositionTickerIfNeeded()
            } catch (e: StemModelNotFoundException) {
                _uiState.value = _uiState.value.copy(
                    isSeparating = false,
                    separationError = "No separation model bundled yet — see the README for how to add one."
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isSeparating = false,
                    separationError = "Separation failed: ${e.message ?: e::class.simpleName}"
                )
            }
        }
    }

    fun togglePlayPause() {
        if (_uiState.value.stems.isEmpty()) return
        if (multiStemPlayer.isPlaying()) multiStemPlayer.pauseAll() else multiStemPlayer.playAll()
        _uiState.value = _uiState.value.copy(playback = _uiState.value.playback.copy(isPlaying = multiStemPlayer.isPlaying()))
    }

    fun seekTo(ms: Long) {
        if (_uiState.value.stems.isEmpty()) return
        multiStemPlayer.seekAllToMs(ms)
        _uiState.value = _uiState.value.copy(playback = _uiState.value.playback.copy(positionMs = ms))
    }

    fun cycleSpeed() {
        if (_uiState.value.stems.isEmpty()) return
        val current = _uiState.value.playback.speed
        val nextIndex = (PLAYBACK_SPEEDS.indexOf(current) + 1).let { if (it >= PLAYBACK_SPEEDS.size) 0 else it }
        val next = PLAYBACK_SPEEDS[nextIndex]
        multiStemPlayer.setSpeedAll(next)
        _uiState.value = _uiState.value.copy(playback = _uiState.value.playback.copy(speed = next))
    }

    fun toggleMute(type: StemType) = updateStemAndVolume(type) { it.copy(isMuted = !it.isMuted) }
    fun toggleSolo(type: StemType) = updateStemAndVolume(type) { it.copy(isSoloed = !it.isSoloed) }
    fun setVolume(type: StemType, volume: Float) = updateStemAndVolume(type) { it.copy(volume = volume) }

    /** Sends this single stem's isolated audio to whoever reads the shared bus next (Chord Detector, in this app). */
    fun exportStemToChordDetector(type: StemType, sourceFileName: String = "Imported Audio") {
        val stem = _uiState.value.stems.firstOrNull { it.type == type } ?: return
        AudioResourceBus.send(
            ExportedAudioResource(
                label = "${stem.label} ($sourceFileName)",
                samples = stem.samples,
                sampleRate = sampleRate,
                sourceModule = "Stem Splitter"
            )
        )
        _uiState.value = _uiState.value.copy(lastExportedLabel = stem.label, lastSavedFilePath = null)
    }

    /** Writes this stem's isolated audio to a real .wav file on device — independent of the bus hand-off. */
    fun saveStemToDevice(type: StemType, outputDirectory: File) {
        val stem = _uiState.value.stems.firstOrNull { it.type == type } ?: return
        val safeName = stem.label.replace(Regex("[^A-Za-z0-9]+"), "_")
        val file = File(outputDirectory, "${safeName}_${System.currentTimeMillis()}.wav")
        WavFileWriter.write(stem.samples, sampleRate, file)
        _uiState.value = _uiState.value.copy(lastSavedFilePath = file.absolutePath, lastExportedLabel = null)
    }

    private fun updateStemAndVolume(type: StemType, transform: (StemChannel) -> StemChannel) {
        val updated = _uiState.value.stems.map { if (it.type == type) transform(it) else it }
        _uiState.value = _uiState.value.copy(stems = updated)
        updated.forEach { multiStemPlayer.setVolume(it.type.name, effectiveVolume(it, updated)) }
    }

    /** Combines mute/solo/volume into the single gain value actually sent to the player. */
    private fun effectiveVolume(stem: StemChannel, allStems: List<StemChannel>): Float {
        val anySoloed = allStems.any { it.isSoloed }
        return when {
            stem.isMuted -> 0f
            anySoloed && !stem.isSoloed -> 0f
            else -> stem.volume
        }
    }

    private fun startPositionTickerIfNeeded() {
        if (positionTickerRunning) return
        positionTickerRunning = true
        viewModelScope.launch {
            while (isActive) {
                delay(150)
                if (_uiState.value.stems.isEmpty()) continue
                val posMs = multiStemPlayer.referencePositionMs()
                val playing = multiStemPlayer.isPlaying()
                val current = _uiState.value
                if (posMs != current.playback.positionMs || playing != current.playback.isPlaying) {
                    _uiState.value = current.copy(playback = current.playback.copy(positionMs = posMs, isPlaying = playing))
                }
            }
        }
    }

    override fun onCleared() {
        multiStemPlayer.releaseAll()
        (stemSeparator as? TFLiteStemSeparator)?.release()
        super.onCleared()
    }
}
