"""
Trains a small multilabel MLP: 12-dim chroma vector -> which of the 12 pitch
classes are actually sounding. Small enough to train on a laptop CPU in
seconds, and small enough to embed its learned weights directly as Kotlin
array literals (see export_weights_to_kotlin.py) — no TFLite/ONNX runtime
needed on-device for a model this size.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from synth_dataset import generate_dataset

TRAIN_SIZE = 20000
TEST_SIZE = 4000
HIDDEN_UNITS = 32


def main():
    print(f"Generating {TRAIN_SIZE} training examples...")
    X_train, y_train = generate_dataset(TRAIN_SIZE, seed=42)
    print(f"Generating {TEST_SIZE} held-out test examples...")
    X_test, y_test = generate_dataset(TEST_SIZE, seed=1234)  # disjoint seed -> disjoint examples

    print(f"Training MLP (12 -> {HIDDEN_UNITS} -> 12)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        activation="relu",
        max_iter=300,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.1,
    )
    clf.fit(X_train, y_train)
    print("Output activation used by sklearn:", clf.out_activation_)

    y_pred = (clf.predict_proba(X_test) >= 0.5).astype(int)

    print("\n--- Held-out test set metrics ---")
    print("Exact-match (all pitch classes correct) accuracy:", accuracy_score(y_test, y_pred))
    print("Per-label micro F1:", f1_score(y_test, y_pred, average="micro"))
    print("Per-label macro F1:", f1_score(y_test, y_pred, average="macro"))
    print("Hamming loss (fraction of individual pitch-class labels wrong):", hamming_loss(y_test, y_pred))

    np.savez(
        "trained_weights.npz",
        W1=clf.coefs_[0], b1=clf.intercepts_[0],
        W2=clf.coefs_[1], b2=clf.intercepts_[1],
        out_activation=clf.out_activation_,
    )
    print("\nSaved trained_weights.npz")


if __name__ == "__main__":
    main()
