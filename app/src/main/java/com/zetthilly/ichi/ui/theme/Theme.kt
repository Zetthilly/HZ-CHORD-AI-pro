package com.zetthilly.ichi.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Palette lifted straight from the HZ Chord AI mockups
val IchiNavyBackground = Color(0xFF0B0E1A)
val IchiCardSurface = Color(0xFF141826)
val IchiGold = Color(0xFFD4AF6A)
val IchiBlue = Color(0xFF3B82F6)
val IchiPurple = Color(0xFF8B5CF6)
val IchiGreen = Color(0xFF22C55E)

private val IchiDarkColorScheme = darkColorScheme(
    primary = IchiBlue,
    secondary = IchiPurple,
    tertiary = IchiGold,
    background = IchiNavyBackground,
    surface = IchiCardSurface
)

@Composable
fun IchiTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = IchiDarkColorScheme,
        content = content
    )
}
