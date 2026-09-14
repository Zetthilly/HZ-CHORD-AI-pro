package com.zetthilly.ichi.ui.dashboard

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

data class RecentProject(val name: String, val subtitle: String, val whenLabel: String)

@Composable
fun DashboardScreen(
    onOpenModules: () -> Unit,
    recentProjects: List<RecentProject> = listOf(
        RecentProject("Worship Session.mp3", "E Major · 112 BPM", "Today"),
        RecentProject("Sungura Groove.wav", "A Major · 105 BPM", "Yesterday")
    )
) {
    Scaffold(topBar = { TopAppBar(title = { Text("HZ CHORD AI") }) }) { padding ->
        Column(
            modifier = Modifier.padding(padding).padding(16.dp).fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("NOW PLAYING", style = MaterialTheme.typography.labelSmall)
                    Text("Worship Session.mp3", fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
                        StatColumn("CURRENT CHORD", "E Major")
                        StatColumn("BPM", "112")
                        StatColumn("KEY", "E Major")
                    }
                }
            }

            Text("Quick Actions", style = MaterialTheme.typography.titleMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                FilledTonalButton(onClick = onOpenModules) { Text("Quick Record") }
                FilledTonalButton(onClick = onOpenModules) { Text("Live Analyzer") }
                FilledTonalButton(onClick = onOpenModules) { Text("Import Audio") }
            }

            Text("Recent Projects", style = MaterialTheme.typography.titleMedium)
            recentProjects.forEach { project ->
                Card(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        modifier = Modifier.padding(12.dp).fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column {
                            Text(project.name, fontWeight = FontWeight.Bold)
                            Text(project.subtitle, style = MaterialTheme.typography.bodySmall)
                        }
                        Text(project.whenLabel, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }

            Button(onClick = onOpenModules, modifier = Modifier.fillMaxWidth()) {
                Text("All Modules")
            }
        }
    }
}

@Composable
private fun StatColumn(label: String, value: String) {
    Column {
        Text(label, style = MaterialTheme.typography.labelSmall)
        Text(value, fontWeight = FontWeight.Bold)
    }
}
