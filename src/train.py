"""
train.py - train and evaluate the models.

Usage (from the project folder):
    python src/train.py                      # trains cnn_lstm
    python src/train.py --model cnn          # cnn, lstm or cnn_lstm
    python src/train.py --model all          # trains all three and compares them
    python src/train.py --epochs 40
    python src/train.py --rebuild            # re-extract features (ignore cache)
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # save plots to files, no window
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from keras import callbacks

from dataset import LABELS, SEED, build_dataset, normalize_data
from models import MODELS, get_model

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
CACHE_FILE = OUTPUT_DIR / "features_cache.npz"


def load_data(rebuild=False):
    """Load cached features, or extract them (takes a few minutes) and cache."""
    if CACHE_FILE.exists() and not rebuild:
        print(f"Loading cached features from {CACHE_FILE}")
        d = np.load(CACHE_FILE)
        return d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    X_train, y_train, X_test, y_test = build_dataset()
    np.savez(CACHE_FILE, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return X_train, y_train, X_test, y_test


def split_train_val(n_samples):
    """Validation split. Each original clip and its augmented copy are stored
    next to each other, so they share a group and never end up on both sides."""
    groups = np.arange(n_samples) // 2
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_idx, val_idx = next(splitter.split(np.zeros(n_samples), groups=groups))
    return train_idx, val_idx


def plot_history(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title(f"{name}: accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title(f"{name}: loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name}: confusion matrix (unseen speakers)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def train_one(name, data, epochs):
    X_tr, y_tr, X_val, y_val, X_test, y_test = data
    print(f"\n===== Training {name} =====")

    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = get_model(name, X_tr.shape[1:], len(LABELS))
    model.summary()

    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-5),
        callbacks.ModelCheckpoint(str(OUTPUT_DIR / f"{name}_best.keras"),
                                  monitor="val_accuracy", mode="max",
                                  save_best_only=True),
    ]
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=32,
                        class_weight=class_weight, callbacks=cbs, verbose=2)

    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = float(np.mean(preds == y_test))
    f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))

    report = classification_report(
        y_test, preds, labels=list(range(len(LABELS))),
        target_names=LABELS, zero_division=0)
    print(report)
    print(f"{name}: test accuracy = {acc:.4f}, macro F1 = {f1:.4f}")
    (OUTPUT_DIR / f"{name}_report.txt").write_text(report, encoding="utf-8")

    plot_history(history, name)
    plot_confusion(y_test, preds, name)
    return acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn_lstm",
                        choices=list(MODELS) + ["all"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--rebuild", action="store_true",
                        help="re-extract features instead of using the cache")
    args = parser.parse_args()

    keras.utils.set_random_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    X_train, y_train, X_test, y_test = load_data(args.rebuild)
    X_train, X_test, mean, std = normalize_data(X_train, X_test)
    np.savez(OUTPUT_DIR / "norm_stats.npz", mean=mean, std=std)

    train_idx, val_idx = split_train_val(len(X_train))
    data = (X_train[train_idx], y_train[train_idx],
            X_train[val_idx], y_train[val_idx], X_test, y_test)
    print(f"Train {len(train_idx)} | Validation {len(val_idx)} | Test {len(X_test)}")

    names = list(MODELS) if args.model == "all" else [args.model]
    results = {}
    for name in names:
        results[name] = train_one(name, data, args.epochs)

    with open(OUTPUT_DIR / "model_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_accuracy", "macro_f1"])
        for name, (acc, f1) in results.items():
            writer.writerow([name, f"{acc:.4f}", f"{f1:.4f}"])

    print("\n===== Results on unseen speakers =====")
    for name, (acc, f1) in results.items():
        print(f"{name:10s} accuracy {acc:.4f}   macro F1 {f1:.4f}")
    print(f"Files saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()