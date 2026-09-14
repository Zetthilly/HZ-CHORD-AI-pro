package com.zetthilly.ichi.ui.moduleLauncher

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

enum class ModuleId { CHORD_DETECTOR, STEM_SPLITTER }

data class ModuleEntry(val id: ModuleId, val label: String, val subtitle: String)

private val modules = listOf(
    ModuleEntry(ModuleId.CHORD_DETECTOR, "Chord Detector", "Chord mode or note-by-note, general or exact voicing"),
    ModuleEntry(ModuleId.STEM_SPLITTER, "Stem Splitter", "Vocals, drums, bass, guitar, keys — with per-stem chords")
)

@Composable
fun ModuleLauncherScreen(onModuleSelected: (ModuleId) -> Unit) {
    Scaffold(topBar = { TopAppBar(title = { Text("Modules") }) }) { padding ->
        LazyVerticalGrid(
            columns = GridCells.Fixed(1),
            modifier = Modifier.padding(padding).padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(modules) { module ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(96.dp)
                        .clickable { onModuleSelected(module.id) }
                ) {
                    Column(
                        modifier = Modifier.fillMaxSize().padding(16.dp),
                        verticalArrangement = Arrangement.Center
                    ) {
                        Text(module.label, style = MaterialTheme.typography.titleMedium)
                        Text(module.subtitle, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}
