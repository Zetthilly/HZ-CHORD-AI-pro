package com.zetthilly.ichi.keyboard

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

private val WHITE_NOTE_ORDER = listOf("C", "D", "E", "F", "G", "A", "B")
private val BLACK_NOTE_AFTER = mapOf("C" to "C#", "D" to "D#", "F" to "F#", "G" to "G#", "A" to "A#")

/**
 * Renders one octave (extendable) of a piano keyboard, highlighting note
 * names present in [highlightedNotes] (e.g. the notes of a detected chord).
 * Purely a rendering composable — theory logic lives in core-audio.
 */
@Composable
fun PianoKeyboard(
    highlightedNotes: Set<String>,
    octaves: Int = 1,
    highlightColor: Color = Color(0xFF3B82F6),
    onKeyTap: (String) -> Unit = {}
) {
    Box(modifier = Modifier.fillMaxWidth().height(160.dp)) {
        Row(modifier = Modifier.fillMaxSize()) {
            repeat(octaves) {
                WHITE_NOTE_ORDER.forEach { note ->
                    WhiteKey(
                        label = note,
                        isHighlighted = note in highlightedNotes,
                        highlightColor = highlightColor,
                        modifier = Modifier.weight(1f).fillMaxHeight(),
                        onClick = { onKeyTap(note) }
                    )
                }
            }
        }
        // Black keys overlaid — simple proportional placement across the white-key row.
        Row(modifier = Modifier.fillMaxWidth().height(100.dp)) {
            repeat(octaves) {
                WHITE_NOTE_ORDER.forEachIndexed { index, note ->
                    Box(modifier = Modifier.weight(1f)) {
                        val blackNote = BLACK_NOTE_AFTER[note]
                        if (blackNote != null) {
                            BlackKey(
                                label = blackNote,
                                isHighlighted = blackNote in highlightedNotes,
                                highlightColor = highlightColor,
                                modifier = Modifier
                                    .align(Alignment.CenterEnd)
                                    .fillMaxWidth(0.5f)
                                    .fillMaxHeight(),
                                onClick = { onKeyTap(blackNote) }
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun WhiteKey(
    label: String,
    isHighlighted: Boolean,
    highlightColor: Color,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Column(
        modifier = modifier
            .padding(1.dp)
            .background(if (isHighlighted) highlightColor else Color.White, RoundedCornerShape(4.dp))
            .border(1.dp, Color.Black.copy(alpha = 0.3f), RoundedCornerShape(4.dp))
            .clickable(onClick = onClick),
        verticalArrangement = Arrangement.Bottom,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = label,
            color = if (isHighlighted) Color.White else Color.Black,
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.padding(bottom = 6.dp)
        )
    }
}

@Composable
private fun BlackKey(
    label: String,
    isHighlighted: Boolean,
    highlightColor: Color,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Column(
        modifier = modifier
            .background(if (isHighlighted) highlightColor else Color(0xFF111111), RoundedCornerShape(3.dp))
            .clickable(onClick = onClick),
        verticalArrangement = Arrangement.Bottom,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = label,
            color = Color.White,
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.padding(bottom = 4.dp)
        )
    }
}
