package com.zetthilly.ichi

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.zetthilly.ichi.ui.navigation.IchiNavGraph
import com.zetthilly.ichi.ui.theme.IchiTheme

class MainActivity : ComponentActivity() {

    private val requestMicPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* handled reactively by each module's start button */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestMicPermission.launch(Manifest.permission.RECORD_AUDIO)
        }

        setContent {
            IchiTheme {
                IchiNavGraph()
            }
        }
    }
}
