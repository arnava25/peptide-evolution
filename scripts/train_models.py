#!/usr/bin/env python3
"""
train_models.py
---------------
Trains all three scoring models: AMP, Toxicity, Stability.

Usage (from project root):
    python3 scripts/train_models.py            # train all
    python3 scripts/train_models.py --model amp
    python3 scripts/train_models.py --model toxicity
    python3 scripts/train_models.py --model stability

Outputs:
    models/amp_model.keras
    models/toxicity_cnn_model.keras
    models/stability_cnn_model.keras
    data/model_trainers/training_report.txt
"""

import os, sys, argparse, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

os.makedirs("models", exist_ok=True)
os.makedirs("data/model_trainers", exist_ok=True)

# ── Encoding ──────────────────────────────────────────────────────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INT   = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 = pad
MAX_LEN     = 50

def encode_seq(seq: str) -> np.ndarray:
    arr = np.zeros(MAX_LEN, dtype=np.int32)
    for i, aa in enumerate(str(seq).upper()[:MAX_LEN]):
        arr[i] = AA_TO_INT.get(aa, 0)
    return arr

def encode_batch(sequences) -> np.ndarray:
    return np.stack([encode_seq(s) for s in sequences])

# ── Model architecture ────────────────────────────────────────────────────────
def build_cnn(vocab_size: int = 21, embed_dim: int = 16,
              filters: int = 64, dense_units: int = 64,
              dropout: float = 0.4, name: str = "cnn") -> keras.Model:
    """
    Standard 1-D CNN for sequence classification.
    Input: integer-encoded sequence (MAX_LEN,)
    Output: single sigmoid probability
    """
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32", name="sequence")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    x = layers.Conv1D(filters,     3, activation="relu", padding="same")(x)
    x = layers.Conv1D(filters // 2, 3, activation="relu", padding="same")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs=inp, outputs=out, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_cnn_small(vocab_size: int = 21, embed_dim: int = 8,
                    filters: int = 32, dense_units: int = 32,
                    dropout: float = 0.5, name: str = "cnn_small") -> keras.Model:
    """
    Smaller, more regularized CNN for the stability model (small dataset).
    Extra L2 regularization throughout.
    """
    reg = keras.regularizers.l2(1e-3)
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32", name="sequence")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True,
                         embeddings_regularizer=reg)(inp)
    x = layers.Conv1D(filters, 3, activation="relu", padding="same",
                      kernel_regularizer=reg)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs=inp, outputs=out, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ── Evaluation helpers ────────────────────────────────────────────────────────
def evaluate(model, X, y, threshold=0.5):
    probs = model.predict(X, verbose=0).flatten()
    preds = (probs >= threshold).astype(int)
    return {
        "auc":      roc_auc_score(y, probs),
        "acc":      accuracy_score(y, preds),
        "f1":       f1_score(y, preds, zero_division=0),
    }

def metrics_str(m):
    return f"AUC={m['auc']:.3f}  ACC={m['acc']:.3f}  F1={m['f1']:.3f}"

# ── Dataset loader ────────────────────────────────────────────────────────────
def load_dataset(path):
    df = pd.read_csv(path)
    X  = encode_batch(df["Sequence"].values)
    y  = df["Label"].values.astype(np.float32)
    return X, y, df

# ── Training routines ─────────────────────────────────────────────────────────
def train_amp(report_lines):
    print("\n" + "=" * 55)
    print("Training AMP model")
    print("=" * 55)

    dataset_path = "data/model_trainers/amp_dataset_v2.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "data/model_trainers/amp_dataset.csv"
    print(f"Dataset: {dataset_path}")

    X, y, df = load_dataset(dataset_path)
    print(f"Samples: {len(y)}  (pos={y.sum():.0f}, neg={(1-y).sum():.0f})")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.12, stratify=y_tr, random_state=42
    )

    model = build_cnn(name="amp_cnn")

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, verbose=0),
    ]

    t0 = time.time()
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=60, batch_size=256, callbacks=cb, verbose=1)

    val_m  = evaluate(model, X_val, y_val)
    test_m = evaluate(model, X_te,  y_te)

    print(f"\n  Val  : {metrics_str(val_m)}")
    print(f"  Test : {metrics_str(test_m)}")
    print(f"  Time : {time.time()-t0:.0f}s")

    model.save("models/amp_model.keras")
    print("  Saved: models/amp_model.keras")

    report_lines += [
        "\n=== AMP Model ===",
        f"Dataset: {dataset_path}  ({len(y)} samples)",
        f"Val  : {metrics_str(val_m)}",
        f"Test : {metrics_str(test_m)}",
    ]


def train_toxicity(report_lines):
    print("\n" + "=" * 55)
    print("Training Toxicity model")
    print("=" * 55)

    dataset_path = "data/model_trainers/toxicity_dataset_v2.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "data/model_trainers/toxicity_dataset_balanced.csv"
    print(f"Dataset: {dataset_path}")

    X, y, df = load_dataset(dataset_path)
    print(f"Samples: {len(y)}  (pos={y.sum():.0f}, neg={(1-y).sum():.0f})")
    if "Source" in df.columns:
        print(f"Sources: {df['Source'].value_counts().to_dict()}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.12, stratify=y_tr, random_state=42
    )

    model = build_cnn(filters=64, dense_units=64, dropout=0.4, name="tox_cnn")

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, verbose=0),
    ]

    t0 = time.time()
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=60, batch_size=128, callbacks=cb, verbose=1)

    val_m  = evaluate(model, X_val, y_val)
    test_m = evaluate(model, X_te,  y_te)

    print(f"\n  Val  : {metrics_str(val_m)}")
    print(f"  Test : {metrics_str(test_m)}")
    print(f"  Time : {time.time()-t0:.0f}s")

    model.save("models/toxicity_cnn_model.keras")
    print("  Saved: models/toxicity_cnn_model.keras")

    report_lines += [
        "\n=== Toxicity Model ===",
        f"Dataset: {dataset_path}  ({len(y)} samples)",
        f"Val  : {metrics_str(val_m)}",
        f"Test : {metrics_str(test_m)}",
    ]


def train_stability(report_lines):
    """
    Stability model — small dataset. Uses 5-fold stratified cross-validation
    to get reliable metric estimates, then trains a final model on all data.
    Uses a smaller, heavily regularized architecture.
    """
    print("\n" + "=" * 55)
    print("Training Stability model  (small dataset, 5-fold CV)")
    print("=" * 55)

    dataset_path = "data/model_trainers/stability_dataset_v2.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "data/model_trainers/stability_dataset_balanced.csv"
    print(f"Dataset: {dataset_path}")

    X, y, df = load_dataset(dataset_path)
    n = len(y)
    print(f"Samples: {n}  (pos={y.sum():.0f}, neg={(1-y).sum():.0f})")
    print(f"  NOTE: Small dataset — using 5-fold CV for honest evaluation.")

    # 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs, fold_accs, fold_f1s = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        m = build_cnn_small(name=f"stab_fold{fold}")
        cb = [callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                      restore_best_weights=True, verbose=0)]
        m.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=100, batch_size=16, callbacks=cb, verbose=0)

        metrics = evaluate(m, X_val, y_val)
        fold_aucs.append(metrics["auc"])
        fold_accs.append(metrics["acc"])
        fold_f1s.append(metrics["f1"])
        print(f"  Fold {fold}: {metrics_str(metrics)}")

    mean_auc = np.mean(fold_aucs)
    mean_acc = np.mean(fold_accs)
    mean_f1  = np.mean(fold_f1s)
    print(f"\n  CV mean  : AUC={mean_auc:.3f}±{np.std(fold_aucs):.3f}"
          f"  ACC={mean_acc:.3f}  F1={mean_f1:.3f}")

    # Final model trained on all data
    print("\n  Training final model on full dataset...")
    final_model = build_cnn_small(name="stability_final")
    cb_final = [callbacks.EarlyStopping(monitor="loss", patience=10,
                                        restore_best_weights=True, verbose=0)]
    final_model.fit(X, y, epochs=120, batch_size=16,
                    callbacks=cb_final, verbose=0)

    final_model.save("models/stability_cnn_model.keras")
    print("  Saved: models/stability_cnn_model.keras")

    report_lines += [
        "\n=== Stability Model ===",
        f"Dataset: {dataset_path}  ({n} samples)",
        f"5-fold CV  AUC={mean_auc:.3f}±{np.std(fold_aucs):.3f}"
        f"  ACC={mean_acc:.3f}  F1={mean_f1:.3f}",
        "NOTE: Small dataset — treat CV metrics as approximate upper bound.",
    ]


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["amp", "toxicity", "stability", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    args = parser.parse_args()

    report = ["Training Report", "=" * 55,
              f"Run: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"]

    if args.model in ("amp", "all"):
        train_amp(report)
    if args.model in ("toxicity", "all"):
        train_toxicity(report)
    if args.model in ("stability", "all"):
        train_stability(report)

    report_path = "data/model_trainers/training_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\nReport saved to {report_path}")
    print("\nAll done.")
