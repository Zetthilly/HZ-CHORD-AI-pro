package com.zetthilly.ichi.core.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.zetthilly.ichi.core.audio.PlaybackUiState
import java.util.Locale

/**
 * Transport bar: play/pause, a seek slider with elapsed/total time, and a
 * speed toggle. Used as-is by the Chord Detector (single imported clip) and
 * the Stem Splitter (all stems moving together under one transport).
 */
@Composable
fun PlaybackControls(
    state: PlaybackUiState,
    onPlayPause: () -> Unit,
    onSeek: (Long) -> Unit,
    onCycleSpeed: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier.fillMaxWidth()) {
        Slider(
            value = state.positionMs.toFloat(),
            onValueChange = { onSeek(it.toLong()) },
            valueRange = 0f..(state.durationMs.coerceAtLeast(1).toFloat())
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(formatMs(state.positionMs), style = MaterialTheme.typography.labelSmall)

            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = onPlayPause) {
                    Icon(if (state.isPlaying) Icons.Filled.Pause else Icons.Filled.PlayArrow, contentDescription = "Play/Pause")
                }
                Spacer(modifier = Modifier.width(8.dp))
                AssistChip(onClick = onCycleSpeed, label = { Text(String.format(Locale.US, "%.2fx", state.speed)) })
            }

            Text(formatMs(state.durationMs), style = MaterialTheme.typography.labelSmall)
        }
    }
}

private fun formatMs(ms: Long): String {
    val totalSeconds = ms / 1000
    val minutes = totalSeconds / 60
    val seconds = totalSeconds % 60
    return String.format(Locale.US, "%02d:%02d", minutes, seconds)
}
