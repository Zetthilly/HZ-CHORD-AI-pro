package com.zetthilly.ichi.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.zetthilly.ichi.chord.ChordDetectorScreen
import com.zetthilly.ichi.stems.StemSplitterScreen
import com.zetthilly.ichi.ui.dashboard.DashboardScreen
import com.zetthilly.ichi.ui.moduleLauncher.ModuleId
import com.zetthilly.ichi.ui.moduleLauncher.ModuleLauncherScreen

object Routes {
    const val DASHBOARD = "dashboard"
    const val MODULES = "modules"
    const val CHORD_DETECTOR = "chord_detector"
    const val STEM_SPLITTER = "stem_splitter"
}

@Composable
fun IchiNavGraph() {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = Routes.DASHBOARD) {
        composable(Routes.DASHBOARD) {
            DashboardScreen(onOpenModules = { navController.navigate(Routes.MODULES) })
        }
        composable(Routes.MODULES) {
            ModuleLauncherScreen(onModuleSelected = { moduleId ->
                when (moduleId) {
                    ModuleId.CHORD_DETECTOR -> navController.navigate(Routes.CHORD_DETECTOR)
                    ModuleId.STEM_SPLITTER -> navController.navigate(Routes.STEM_SPLITTER)
                }
            })
        }
        composable(Routes.CHORD_DETECTOR) {
            ChordDetectorScreen(onBack = { navController.popBackStack() })
        }
        composable(Routes.STEM_SPLITTER) {
            StemSplitterScreen(
                onExportedToChordDetector = { navController.navigate(Routes.CHORD_DETECTOR) },
                onBack = { navController.popBackStack() }
            )
        }
    }
}
