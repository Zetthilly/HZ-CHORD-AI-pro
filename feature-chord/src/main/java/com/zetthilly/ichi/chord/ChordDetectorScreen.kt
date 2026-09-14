package com.zetthilly.ichi.chord

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.zetthilly.ichi.core.theory.ChordComplexity
import com.zetthilly.ichi.core.theory.ChordDisplayMode
import com.zetthilly.ichi.core.theory.Transposer
import com.zetthilly.ichi.core.theory.VoicingMode
import com.zetthilly.ichi.core.ui.PlaybackControls
import com.zetthilly.ichi.keyboard.PianoKeyboard

@Composable
fun ChordDetectorScreen(
    viewModel: ChordDetectorViewModel = viewModel(),
    onBack: () -> Unit = {}
) {
    val state by viewModel.uiState.collectAsState()
    val settings = state.settings
    val transpose = state.transposeSemitones

    // Picks up an exported stem (or any other resource) waiting on the bus, if any.
    // A no-op if nothing was sent — this screen works standalone otherwise.
    LaunchedEffect(Unit) { viewModel.checkForImportedAudio() }

    // Transpose is applied here, at display time, so moving the stepper updates
    // the chord name / note list / keyboard instantly — no re-analysis needed.
    val displayChord = state.currentChord?.let { Transposer.transposeChord(it, transpose) }
    val displayArrangedNotes = state.arrangedNotes.map { Transposer.transposeNote(it, transpose) }
    val displayRawNotes = state.rawDetectedNotes.map { Transposer.transposeNote(it, transpose) }
    val displayTimeline = state.timeline.map { Transposer.transposeChord(it, transpose) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Chord Detector") },
                navigationIcon = { IconButton(onClick = onBack) { Text("←") } }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier.padding(padding).padding(16.dp).fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                AssistChip(onClick = {}, label = { Text("Source: ${state.sourceLabel}") })
                TransposeStepper(
                    semitones = transpose,
                    onChange = { viewModel.setTranspose(it) }
                )
            }

            // Transport only makes sense for a loaded clip — live mic has no seekable buffer.
            if (state.hasImportedAudio) {
                PlaybackControls(
                    state = state.playback,
                    onPlayPause = viewModel::togglePlayPause,
                    onSeek = viewModel::seekTo,
                    onCycleSpeed = viewModel::cycleSpeed
                )
                Button(onClick = { viewModel.startListening() }, modifier = Modifier.fillMaxWidth()) {
                    Text("Switch to Live Mic")
                }
            }

            // --- Mode toggles: these directly change what's detected and how it's arranged ---
            ToggleRow(
                label = "Display",
                options = listOf("Chord" to ChordDisplayMode.CHORD_MODE, "Note-by-Note" to ChordDisplayMode.NOTE_BY_NOTE_MODE),
                selected = settings.displayMode,
                onSelect = { viewModel.updateSettings(settings.copy(displayMode = it)) }
            )
            ToggleRow(
                label = "Voicing",
                options = listOf("General" to VoicingMode.GENERAL_VOICING, "Exact (as played)" to VoicingMode.EXACT_VOICING),
                selected = settings.voicingMode,
                onSelect = { viewModel.updateSettings(settings.copy(voicingMode = it)) },
                enabled = settings.displayMode == ChordDisplayMode.CHORD_MODE
            )
            ToggleRow(
                label = "Complexity",
                options = listOf("Simple (triad)" to ChordComplexity.SIMPLE_CHORD, "Complete" to ChordComplexity.COMPLETE_CHORD),
                selected = settings.complexity,
                onSelect = { viewModel.updateSettings(settings.copy(complexity = it)) },
                enabled = settings.displayMode == ChordDisplayMode.CHORD_MODE
            )

            HorizontalDivider()

            if (settings.displayMode == ChordDisplayMode.NOTE_BY_NOTE_MODE) {
                Text("Detected Notes", style = MaterialTheme.typography.titleMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    displayRawNotes.forEach { note ->
                        Card { Text("${note.noteName}${note.octave}", modifier = Modifier.padding(10.dp)) }
                    }
                }
                PianoKeyboard(highlightedNotes = displayRawNotes.map { it.noteName }.toSet())
            } else {
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Column {
                        Text(
                            text = displayChord?.name ?: "—",
                            style = MaterialTheme.typography.headlineLarge,
                            fontWeight = FontWeight.Bold
                        )
                        Text(text = displayChord?.quality ?: "Listening…")
                    }
                    Text(text = "${state.confidencePercent}% confidence")
                }

                Text("Voicing: " + displayArrangedNotes.joinToString(" – ") { "${it.noteName}${it.octave}" })

                PianoKeyboard(highlightedNotes = displayArrangedNotes.map { it.noteName }.toSet())

                Text(text = "Timeline", style = MaterialTheme.typography.titleSmall)
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(displayTimeline) { chord ->
                        Card { Text(text = chord.name, modifier = Modifier.padding(12.dp)) }
                    }
                }
            }

            Spacer(modifier = Modifier.weight(1f))

            if (!state.hasImportedAudio) {
                Button(
                    onClick = { if (state.isListening) viewModel.stopListening() else viewModel.startListening() },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(if (state.isListening) "Stop" else "Start Listening")
                }
                TextButton(
                    onClick = { viewModel.exportCapturedAudioToStemSplitter() },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Send Captured Audio to Stem Splitter")
                }
            }
        }
    }
}

@Composable
private fun TransposeStepper(semitones: Int, onChange: (Int) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("Transpose", style = MaterialTheme.typography.labelSmall)
        Spacer(modifier = Modifier.width(4.dp))
        IconButton(onClick = { onChange(semitones - 1) }) { Text("–") }
        Text(if (semitones >= 0) "+$semitones" else "$semitones")
        IconButton(onClick = { onChange(semitones + 1) }) { Text("+") }
    }
}

@Composable
private fun <T> ToggleRow(
    label: String,
    options: List<Pair<String, T>>,
    selected: T,
    onSelect: (T) -> Unit,
    enabled: Boolean = true
) {
    Column {
        Text(label, style = MaterialTheme.typography.labelMedium)
        Spacer(modifier = Modifier.height(4.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.forEach { (text, value) ->
                FilterChip(
                    selected = value == selected,
                    onClick = { if (enabled) onSelect(value) },
                    enabled = enabled,
                    label = { Text(text) }
                )
            }
        }
    }
}
