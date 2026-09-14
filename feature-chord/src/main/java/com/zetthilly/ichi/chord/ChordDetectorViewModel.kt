package com.zetthilly.ichi.chord

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.zetthilly.ichi.core.audio.AudioEngine
import com.zetthilly.ichi.core.audio.AudioResourceBus
import com.zetthilly.ichi.core.audio.ExportedAudioResource
import com.zetthilly.ichi.core.audio.PcmTrackPlayer
import com.zetthilly.ichi.core.audio.PlaybackUiState
import com.zetthilly.ichi.core.dsp.ChromaExtractor
import com.zetthilly.ichi.core.dsp.HybridArpeggioNoteDetector
import com.zetthilly.ichi.core.dsp.PolyphonicNoteDetector
import com.zetthilly.ichi.core.dsp.VoicedNote
import com.zetthilly.ichi.core.theory.ChordAnalysisSettings
import com.zetthilly.ichi.core.theory.ChordArranger
import com.zetthilly.ichi.core.theory.ChordDefinition
import com.zetthilly.ichi.core.theory.ChordDetector
import com.zetthilly.ichi.core.theory.ChordDisplayMode
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

private val PLAYBACK_SPEEDS = listOf(0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f)
private const val ANALYSIS_FRAME_SIZE = 4096

data class ChordDetectorUiState(
    val isListening: Boolean = false,
    val hasImportedAudio: Boolean = false,
    val sourceLabel: String = "Live Mic",
    val settings: ChordAnalysisSettings = ChordAnalysisSettings(),
    val transposeSemitones: Int = 0, // display-only shift — does not touch playback pitch
    val playback: PlaybackUiState = PlaybackUiState(),
    val currentChord: ChordDefinition? = null, // always the RAW detected chord (pre-transpose)
    val confidencePercent: Int = 0,
    val arrangedNotes: List<VoicedNote> = emptyList(), // RAW, pre-transpose
    val rawDetectedNotes: List<VoicedNote> = emptyList(), // note-by-note mode, pre-transpose
    val timeline: List<ChordDefinition> = emptyList() // RAW chords, pre-transpose
)

/**
 * Chord Detector is fully standalone: it doesn't know Stem Splitter exists.
 * The only external input it accepts is whatever's waiting in
 * [AudioResourceBus] (an isolated stem someone explicitly exported), picked
 * up once via [checkForImportedAudio]. When a clip is loaded this way, the
 * screen gets real playback (play/pause/seek/speed via [PcmTrackPlayer]) and
 * the chord/note analysis tracks the playback position as it plays or is
 * scrubbed. Transpose is applied at the UI layer (see [ChordDetectorScreen])
 * so changing it updates instantly without needing new audio.
 */
class ChordDetectorViewModel(
    private val audioEngine: AudioEngine = AudioEngine(),
    private val defaultSampleRate: Int = AudioEngine.DEFAULT_SAMPLE_RATE
) : ViewModel() {

    private var chromaExtractor = ChromaExtractor(defaultSampleRate)
    private var polyNoteDetector = PolyphonicNoteDetector(defaultSampleRate)
    private var arpeggioNoteDetector = HybridArpeggioNoteDetector(defaultSampleRate)
    private val chordDetector = ChordDetector()
    private val player = PcmTrackPlayer(defaultSampleRate)

    private var importedSamples: FloatArray = FloatArray(0)
    private var importedSampleRate: Int = defaultSampleRate
    private var positionTickerRunning = false

    // Rolling capture of live mic audio, capped at ~15s, so it can be sent
    // onward to Stem Splitter — the reverse direction of the same handoff.
    private val capturedSamples = ArrayDeque<Float>()
    private val maxCapturedSamples = defaultSampleRate * 15

    private val _uiState = MutableStateFlow(ChordDetectorUiState())
    val uiState: StateFlow<ChordDetectorUiState> = _uiState.asStateFlow()

    fun updateSettings(settings: ChordAnalysisSettings) {
        _uiState.value = _uiState.value.copy(settings = settings)
    }

    fun setTranspose(semitones: Int) {
        _uiState.value = _uiState.value.copy(transposeSemitones = semitones.coerceIn(-12, 12))
    }

    /** Sends whatever's been captured from the live mic to Stem Splitter, via the same bus. */
    fun exportCapturedAudioToStemSplitter() {
        if (capturedSamples.isEmpty()) return
        AudioResourceBus.send(
            ExportedAudioResource(
                label = "Chord Detector capture",
                samples = capturedSamples.toFloatArray(),
                sampleRate = defaultSampleRate,
                sourceModule = "Chord Detector"
            )
        )
    }

    /** Call once when the screen opens — picks up an exported stem, if any, and loads it for playback + analysis. */
    fun checkForImportedAudio() {
        val resource = AudioResourceBus.consume() ?: return
        loadImportedAudio(resource)
    }

    private fun loadImportedAudio(resource: ExportedAudioResource) {
        importedSamples = resource.samples
        importedSampleRate = resource.sampleRate
        if (resource.sampleRate != defaultSampleRate) {
            chromaExtractor = ChromaExtractor(resource.sampleRate)
            polyNoteDetector = PolyphonicNoteDetector(resource.sampleRate)
            arpeggioNoteDetector = HybridArpeggioNoteDetector(resource.sampleRate)
        }

        player.load(resource.samples)

        _uiState.value = _uiState.value.copy(
            hasImportedAudio = true,
            sourceLabel = resource.label,
            isListening = false,
            timeline = emptyList(),
            currentChord = null,
            playback = PlaybackUiState(durationMs = player.durationMs())
        )

        analyzeAtCurrentPosition()
        startPositionTickerIfNeeded()
    }

    fun togglePlayPause() {
        if (!_uiState.value.hasImportedAudio) return
        if (player.isPlaying()) player.pause() else player.play()
        _uiState.value = _uiState.value.copy(
            playback = _uiState.value.playback.copy(isPlaying = player.isPlaying())
        )
    }

    fun seekTo(ms: Long) {
        if (!_uiState.value.hasImportedAudio) return
        player.seekToMs(ms)
        _uiState.value = _uiState.value.copy(playback = _uiState.value.playback.copy(positionMs = ms))
        analyzeAtCurrentPosition()
    }

    fun cycleSpeed() {
        if (!_uiState.value.hasImportedAudio) return
        val current = _uiState.value.playback.speed
        val nextIndex = (PLAYBACK_SPEEDS.indexOf(current) + 1).let { if (it >= PLAYBACK_SPEEDS.size) 0 else it }
        val next = PLAYBACK_SPEEDS[nextIndex]
        player.setSpeed(next)
        _uiState.value = _uiState.value.copy(playback = _uiState.value.playback.copy(speed = next))
    }

    /** Requires RECORD_AUDIO permission already granted by the caller. */
    fun startListening() {
        player.pause()
        _uiState.value = _uiState.value.copy(isListening = true, sourceLabel = "Live Mic", hasImportedAudio = false)
        viewModelScope.launch {
            audioEngine.pcmFrames().collect { frame ->
                captureForExport(frame)
                applyFrame(frame)
            }
        }
    }

    fun stopListening() {
        _uiState.value = _uiState.value.copy(isListening = false)
    }

    private fun captureForExport(frame: FloatArray) {
        capturedSamples.addAll(frame.toList())
        while (capturedSamples.size > maxCapturedSamples) capturedSamples.removeFirst()
    }

    private fun startPositionTickerIfNeeded() {
        if (positionTickerRunning) return
        positionTickerRunning = true
        viewModelScope.launch {
            while (isActive) {
                delay(150)
                if (!_uiState.value.hasImportedAudio) continue
                val posMs = player.positionMs()
                val playing = player.isPlaying()
                val current = _uiState.value
                if (posMs != current.playback.positionMs || playing != current.playback.isPlaying) {
                    _uiState.value = current.copy(playback = current.playback.copy(positionMs = posMs, isPlaying = playing))
                    if (playing) analyzeAtCurrentPosition()
                }
            }
        }
    }

    /** Runs one analysis frame starting at the player's current position. */
    private fun analyzeAtCurrentPosition() {
        val posMs = player.positionMs()
        val startSample = ((posMs / 1000.0) * importedSampleRate).toInt()
        val endSample = (startSample + ANALYSIS_FRAME_SIZE).coerceAtMost(importedSamples.size)
        if (startSample >= endSample) return
        val frame = importedSamples.copyOfRange(startSample, endSample)
        if (frame.size < 512) return
        applyFrame(frame)
    }

    private fun applyFrame(frame: FloatArray) {
        val current = _uiState.value
        val settings = current.settings

        if (settings.displayMode == ChordDisplayMode.NOTE_BY_NOTE_MODE) {
            val notes = arpeggioNoteDetector.detect(frame)
            _uiState.value = current.copy(rawDetectedNotes = notes)
            return
        }

        val chroma = chromaExtractor.extract(frame)
        val match = chordDetector.detectChord(chroma)
        val rawNotes = polyNoteDetector.detect(frame)
        val arranged = ChordArranger.arrange(match.chord, settings, rawNotes)

        val latest = _uiState.value // re-read in case settings changed mid-flight
        val updatedTimeline = if (latest.currentChord?.name != match.chord.name) {
            (latest.timeline + match.chord).takeLast(8)
        } else latest.timeline

        _uiState.value = latest.copy(
            currentChord = match.chord,
            confidencePercent = match.confidencePercent,
            arrangedNotes = arranged,
            rawDetectedNotes = rawNotes,
            timeline = updatedTimeline
        )
    }

    override fun onCleared() {
        player.release()
        super.onCleared()
    }
}
