package com.zetthilly.ichi.stems

import android.content.Context
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.zetthilly.ichi.core.audio.AudioFileDecoder
import com.zetthilly.ichi.core.ui.PlaybackControls
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

@Composable
fun StemSplitterScreen(
    viewModel: StemSplitterViewModel = viewModel(),
    onExportedToChordDetector: () -> Unit = {}, // navigate away after a successful export, if desired
    onBack: () -> Unit = {}
) {
    val state by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    var isDecoding by remember { mutableStateOf(false) }
    var decodeFailed by remember { mutableStateOf(false) }

    // Real file picker: system Storage Access Framework document picker — no
    // storage permission needed, works the same across Android versions.
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        decodeFailed = false
        isDecoding = true
        coroutineScope.launch {
            val samples = withContext(Dispatchers.IO) {
                AudioFileDecoder.decodeToMonoPcm(context, uri)
            }
            isDecoding = false
            if (samples != null) {
                val fileName = uri.lastPathSegment?.substringAfterLast('/') ?: "Imported Audio"
                viewModel.separate(samples, sourceFileName = fileName)
            } else {
                decodeFailed = true
            }
        }
    }

    // Brief, self-clearing confirmations — this screen never shows chord data itself.
    LaunchedEffect(state.lastExportedLabel, state.lastSavedFilePath) {
        if (state.lastExportedLabel != null || state.lastSavedFilePath != null) delay(2500)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Stem Splitter") },
                navigationIcon = { IconButton(onClick = onBack) { Text("←") } }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier.padding(padding).padding(16.dp).fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {

            Button(
                onClick = { filePickerLauncher.launch(arrayOf("audio/*")) },
                modifier = Modifier.fillMaxWidth(),
                enabled = !isDecoding && !state.isSeparating
            ) {
                Text(if (isDecoding) "Decoding…" else "Import Audio")
            }
            if (decodeFailed) {
                Text("Couldn't decode that file — try a different format.", color = MaterialTheme.colorScheme.error)
            }
            state.separationError?.let { error ->
                Text(error, color = MaterialTheme.colorScheme.error)
            }

            if (state.isSeparating) {
                Text("Separating… ${(state.separationProgress * 100).toInt()}%")
                LinearProgressIndicator(
                    progress = state.separationProgress,
                    modifier = Modifier.fillMaxWidth()
                )
            }

            state.lastExportedLabel?.let { label ->
                AssistChip(onClick = {}, label = { Text("Sent \"$label\" to Chord Detector") })
            }
            state.lastSavedFilePath?.let { path ->
                AssistChip(onClick = {}, label = { Text("Saved to ${File(path).name}") })
            }

            if (state.stems.isNotEmpty()) {
                // One shared transport — all stems stay in sync under it.
                PlaybackControls(
                    state = state.playback,
                    onPlayPause = viewModel::togglePlayPause,
                    onSeek = viewModel::seekTo,
                    onCycleSpeed = viewModel::cycleSpeed
                )

                state.stems.forEach { stem ->
                    StemRow(
                        stem = stem,
                        onMute = { viewModel.toggleMute(stem.type) },
                        onSolo = { viewModel.toggleSolo(stem.type) },
                        onVolumeChange = { viewModel.setVolume(stem.type, it) },
                        onSendToChordDetector = {
                            viewModel.exportStemToChordDetector(stem.type)
                            onExportedToChordDetector()
                        },
                        onSaveToDevice = {
                            viewModel.saveStemToDevice(stem.type, exportsDirectory(context))
                        }
                    )
                }
            }
        }
    }
}

private fun exportsDirectory(context: Context): File =
    File(context.getExternalFilesDir(null), "exports").apply { mkdirs() }

@Composable
private fun StemRow(
    stem: StemChannel,
    onMute: () -> Unit,
    onSolo: () -> Unit,
    onVolumeChange: (Float) -> Unit,
    onSendToChordDetector: () -> Unit,
    onSaveToDevice: () -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(stem.label, fontWeight = FontWeight.Bold)
            Slider(value = stem.volume, onValueChange = onVolumeChange)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = stem.isSoloed, onClick = onSolo, label = { Text("S") })
                FilterChip(selected = stem.isMuted, onClick = onMute, label = { Text("M") })
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                TextButton(onClick = onSaveToDevice) { Text("Save to Device") }
                TextButton(onClick = onSendToChordDetector) { Text("Send to Chord Detector") }
            }
        }
    }
}
