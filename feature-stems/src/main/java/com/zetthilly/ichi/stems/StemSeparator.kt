package com.zetthilly.ichi.stems

import android.content.Context
import android.content.res.AssetFileDescriptor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.ceil
import kotlin.math.min

data class Stems(
    val vocals: FloatArray,
    val drums: FloatArray,
    val bass: FloatArray,
    val guitar: FloatArray,
    val keys: FloatArray,
    val other: FloatArray
)

interface StemSeparator {
    suspend fun separate(inputSamples: FloatArray, sampleRate: Int, onProgress: (Float) -> Unit): Stems
}

/** Thrown when the .tflite model asset hasn't been added yet — see the class doc below for how to get one. */
class StemModelNotFoundException(assetPath: String) : Exception(
    "No separation model found at assets/$assetPath. Convert a Demucs/Spleeter checkpoint to " +
        "TFLite and place it there — see feature-stems/StemSeparator.kt doc, or the project README, " +
        "for the full conversion path."
)

/**
 * Runs a real, trained neural-network model on-device for source separation
 * — chroma/DSP tricks can't get Demucs/Spleeter-quality separation, so this
 * genuinely needs a converted model file, loaded and executed here with
 * TensorFlow Lite's Interpreter. Every inference happens on-device: no
 * network call, no server, nothing leaves the phone.
 *
 * ## Getting a model into this class
 *
 * 1. **Get a pretrained checkpoint.** Demucs v4 (Meta, MIT license) or
 *    Spleeter (Deezer, MIT license) are the standard open options. Demucs
 *    generally separates better; Spleeter is lighter and easier to convert.
 *
 * 2. **Convert it to TFLite**, fully offline, no cloud conversion service
 *    needed:
 *      PyTorch checkpoint --(ai-edge-torch, or PyTorch->ONNX->TF->TFLite)--> .tflite
 *    `ai-edge-torch` (Google's newer converter) can go directly from a
 *    PyTorch `nn.Module` to `.tflite` in one step and is the least fiddly
 *    path as of 2025-2026; the ONNX route is the fallback if a given model's
 *    ops aren't yet supported by ai-edge-torch.
 *
 * 3. **Quantize** (post-training int8, or at least dynamic-range
 *    quantization) — this is what takes a model from "too big/slow for a
 *    phone" to something that actually runs in a reasonable time. Expect a
 *    real quality/size/speed trade-off to tune here.
 *
 * 4. **Drop the resulting file into `app/src/main/assets/models/`** (any
 *    name; update [modelAssetPath] to match) and rebuild. `noCompress +=
 *    "tflite"` is already set in the app module so the file can be mapped
 *    into memory directly rather than decompressed on load.
 *
 * 5. **Match the shape constants below** ([MODEL_CHUNK_SAMPLES],
 *    [MODEL_STEM_COUNT], etc.) to whatever your specific converted model
 *    actually expects/produces — these vary by model and conversion, so
 *    they can't be guessed correctly in advance. Inspect the model with
 *    Python (`interpreter.get_input_details()` /
 *    `get_output_details()` in `tensorflow.lite`) once it's converted, and
 *    set these to match exactly.
 *
 * Most 4-stem models (drums/bass/other/vocals) won't distinguish guitar
 * from keys — this maps both to "other" until/unless you use or train a
 * model with finer stem categories. That mapping lives in [assembleStems].
 */
class TFLiteStemSeparator(
    private val context: Context,
    private val modelAssetPath: String = "models/stem_separator.tflite"
) : StemSeparator {

    // --- Model-shape constants: ADJUST to match your specific converted model ---
    private val MODEL_CHUNK_SAMPLES = 44100 * 8   // e.g. 8-second chunks at 44.1kHz — depends on the model you convert
    private val MODEL_OVERLAP_SAMPLES = 44100 * 1 // 1s overlap, cross-faded between chunks to avoid audible seams
    private val MODEL_STEM_COUNT = 4              // most open 4-stem models: [drums, bass, other, vocals] — verify order against your conversion
    // --- end model-shape constants ---

    private var interpreter: Interpreter? = null

    private fun ensureLoaded(): Interpreter {
        interpreter?.let { return it }
        val modelBuffer = try {
            loadModelFile(context, modelAssetPath)
        } catch (e: java.io.FileNotFoundException) {
            throw StemModelNotFoundException(modelAssetPath)
        }
        val options = Interpreter.Options().apply {
            numThreads = 4
            // setUseNNAPI(true) // uncomment to try hardware acceleration via NNAPI once your model is verified compatible
        }
        return Interpreter(modelBuffer, options).also { interpreter = it }
    }

    override suspend fun separate(
        inputSamples: FloatArray,
        sampleRate: Int,
        onProgress: (Float) -> Unit
    ): Stems = withContext(Dispatchers.Default) {
        val interp = ensureLoaded()

        val step = (MODEL_CHUNK_SAMPLES - MODEL_OVERLAP_SAMPLES).coerceAtLeast(1)
        val totalChunks = ceil(inputSamples.size.toDouble() / step).toInt().coerceAtLeast(1)

        val stemAccumulators = Array(MODEL_STEM_COUNT) { FloatArray(inputSamples.size) }
        val weightAccumulator = FloatArray(inputSamples.size)

        var offset = 0
        var chunkIndex = 0

        while (offset < inputSamples.size) {
            val chunkEnd = min(offset + MODEL_CHUNK_SAMPLES, inputSamples.size)
            val chunkLength = chunkEnd - offset

            val inputChunk = FloatArray(MODEL_CHUNK_SAMPLES) // zero-padded if this is the last, short chunk
            inputSamples.copyInto(inputChunk, destinationOffset = 0, startIndex = offset, endIndex = chunkEnd)

            // Shape here is [1, MODEL_CHUNK_SAMPLES] — the most common convention for a mono
            // waveform-in model. If your converted model expects a different rank/shape
            // (e.g. [1, 1, N] or stereo [1, 2, N]), adjust this input array's nesting to match.
            val input = arrayOf(inputChunk)
            val output = Array(1) { Array(MODEL_STEM_COUNT) { FloatArray(MODEL_CHUNK_SAMPLES) } }

            interp.run(input, output)

            // Overlap-add with a linear cross-fade across the overlap region so chunk
            // boundaries don't produce audible clicks/seams in the reconstructed stems.
            for (i in 0 until chunkLength) {
                val weight = fadeWeight(i, chunkLength, MODEL_OVERLAP_SAMPLES)
                weightAccumulator[offset + i] += weight
                for (s in 0 until MODEL_STEM_COUNT) {
                    stemAccumulators[s][offset + i] += output[0][s][i] * weight
                }
            }

            offset += step
            chunkIndex++
            onProgress((chunkIndex.toFloat() / totalChunks).coerceIn(0f, 1f))
        }

        for (s in 0 until MODEL_STEM_COUNT) {
            for (i in stemAccumulators[s].indices) {
                if (weightAccumulator[i] > 0f) stemAccumulators[s][i] /= weightAccumulator[i]
            }
        }

        assembleStems(stemAccumulators)
    }

    /** Cross-fade weight: ramps up over the leading overlap, flat in the middle, ramps down over the trailing overlap. */
    private fun fadeWeight(indexInChunk: Int, chunkLength: Int, overlapSamples: Int): Float {
        val fadeIn = (indexInChunk.toFloat() / overlapSamples).coerceIn(0f, 1f)
        val fadeOut = ((chunkLength - indexInChunk).toFloat() / overlapSamples).coerceIn(0f, 1f)
        return min(fadeIn, fadeOut).coerceAtLeast(0.0001f)
    }

    /** Maps the model's raw stem outputs onto this app's 6-stem model. See class doc re: guitar/keys. */
    private fun assembleStems(raw: Array<FloatArray>): Stems {
        // Index order assumes the common [drums, bass, other, vocals] convention —
        // verify against your specific model's actual output order and reorder if needed.
        val drums = raw.getOrElse(0) { FloatArray(0) }
        val bass = raw.getOrElse(1) { FloatArray(0) }
        val other = raw.getOrElse(2) { FloatArray(0) }
        val vocals = raw.getOrElse(3) { FloatArray(0) }

        return Stems(
            vocals = vocals,
            drums = drums,
            bass = bass,
            guitar = other,  // 4-stem models don't separate these individually —
            keys = other,    // both point at "other" until a finer-grained model is used.
            other = other
        )
    }

    private fun loadModelFile(context: Context, assetPath: String): MappedByteBuffer {
        val afd: AssetFileDescriptor = context.assets.openFd(assetPath)
        FileInputStream(afd.fileDescriptor).use { input ->
            val channel: FileChannel = input.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    fun release() {
        interpreter?.close()
        interpreter = null
    }
}
