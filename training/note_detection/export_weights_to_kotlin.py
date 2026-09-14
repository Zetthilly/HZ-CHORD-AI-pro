"""
Converts trained_weights.npz into a ready-to-drop-in Kotlin file. The model
is small enough (812 floats total) that embedding the weights as array
literals and hand-writing the two-layer forward pass is simpler and lighter
than pulling in a TFLite/ONNX runtime dependency for a model this size.
"""
import numpy as np

OUT_PATH = "TrainedPitchClassifier.kt"


def format_float_array(name: str, arr: np.ndarray) -> str:
    values = ", ".join(f"{v:.8f}f" for v in arr.flatten())
    return f"private val {name} = floatArrayOf({values})"


def main():
    data = np.load("trained_weights.npz", allow_pickle=True)
    W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]
    out_activation = str(data["out_activation"])

    assert out_activation == "logistic", (
        f"Expected sklearn's multilabel output activation to be 'logistic' (sigmoid), "
        f"got '{out_activation}' — the Kotlin forward pass below assumes sigmoid output; "
        f"update it if you retrain with a different configuration."
    )

    in_dim, hidden_dim = W1.shape
    hidden_dim2, out_dim = W2.shape
    assert hidden_dim == hidden_dim2

    kotlin = f"""package com.zetthilly.ichi.core.theory

import kotlin.math.exp

/**
 * A small (12 -> {hidden_dim} -> {out_dim}) multilabel MLP, trained on synthetic
 * multi-note mixtures to predict which of the 12 pitch classes are actually
 * sounding from a chroma vector — learned weights, not a hand-tuned formula.
 *
 * This is genuinely trained: see /training/note_detection/ at the project
 * root for the full pipeline (synthetic data generator, training script,
 * this export step). It was trained on SYNTHETIC harmonic mixtures, not real
 * recorded instruments — treat it as a real first model and a working
 * scaffold, not a finished, production-accurate one. See that folder's
 * README for how to retrain on real recorded arpeggios (MAESTRO, GuitarSet,
 * or your own recordings) for meaningfully better real-world accuracy.
 *
 * Input: an existing ChromaExtractor.extract(frame) output (12-dim, sums to 1).
 * Output: 12 probabilities, one per pitch class (C, C#, D, ... B) — threshold
 * to get a binary "is this pitch class currently sounding" decision.
 */
object TrainedPitchClassifier {{

    private const val INPUT_DIM = {in_dim}
    private const val HIDDEN_DIM = {hidden_dim}
    private const val OUTPUT_DIM = {out_dim}

    {format_float_array("W1", W1)} // shape [{in_dim}][{hidden_dim}], row-major
    {format_float_array("B1", b1)}
    {format_float_array("W2", W2)} // shape [{hidden_dim}][{out_dim}], row-major
    {format_float_array("B2", b2)}

    /** Returns 12 probabilities (one per pitch class C..B) that each is actively sounding. */
    fun predict(chroma: DoubleArray): FloatArray {{
        require(chroma.size == INPUT_DIM)

        val hidden = FloatArray(HIDDEN_DIM)
        for (h in 0 until HIDDEN_DIM) {{
            var sum = B1[h]
            for (i in 0 until INPUT_DIM) {{
                sum += chroma[i].toFloat() * W1[i * HIDDEN_DIM + h]
            }}
            hidden[h] = if (sum > 0f) sum else 0f // ReLU
        }}

        val output = FloatArray(OUTPUT_DIM)
        for (o in 0 until OUTPUT_DIM) {{
            var sum = B2[o]
            for (h in 0 until HIDDEN_DIM) {{
                sum += hidden[h] * W2[h * OUTPUT_DIM + o]
            }}
            output[o] = sigmoid(sum)
        }}
        return output
    }}

    /** Convenience: pitch classes predicted active above [threshold]. */
    fun activePitchClasses(chroma: DoubleArray, threshold: Float = 0.5f): List<Int> =
        predict(chroma).withIndex().filter {{ it.value >= threshold }}.map {{ it.index }}

    private fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()
}}
"""
    with open(OUT_PATH, "w") as f:
        f.write(kotlin)
    print(f"Wrote {OUT_PATH} ({len(kotlin)} chars, {W1.size + b1.size + W2.size + b2.size} weight values)")


if __name__ == "__main__":
    main()
