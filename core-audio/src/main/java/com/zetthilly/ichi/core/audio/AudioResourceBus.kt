package com.zetthilly.ichi.core.audio

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class ExportedAudioResource(
    val label: String,        // e.g. "Vocals (Worship Session.mp3)"
    val samples: FloatArray,
    val sampleRate: Int,
    val sourceModule: String  // e.g. "Stem Splitter" — for display only, no code dependency
)

/**
 * A one-slot mailbox for passing an audio buffer between features that
 * otherwise know nothing about each other. Stem Splitter calls [send] when
 * the user taps "Export to Chord Detector"; Chord Detector calls [consume]
 * when it opens to pick up anything waiting. Neither module depends on the
 * other's code — this is the only connection between them, and it's opt-in
 * per export, not automatic.
 *
 * Lives in core-audio (a dependency both features already have) so it adds
 * zero new module coupling.
 */
object AudioResourceBus {
    private val _pending = MutableStateFlow<ExportedAudioResource?>(null)
    val pending: StateFlow<ExportedAudioResource?> = _pending.asStateFlow()

    fun send(resource: ExportedAudioResource) {
        _pending.value = resource
    }

    /** Reads and clears the mailbox in one step, so it's only ever consumed once. */
    fun consume(): ExportedAudioResource? {
        val resource = _pending.value
        _pending.value = null
        return resource
    }
}
