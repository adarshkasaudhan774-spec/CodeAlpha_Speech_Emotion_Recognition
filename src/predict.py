"""
predict.py - predict the emotion of one .wav file with a trained model.

Usage:
    python src/predict.py path\to\audio.wav
    python src/predict.py path\to\audio.wav --model cnn
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import LABELS
from features import extract_mfcc, load_audio

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def predict_file(path, model_name="cnn_lstm"):
    """Return a list of (emotion, probability), best first."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    model_path = OUTPUT_DIR / f"{model_name}_best.keras"
    stats_path = OUTPUT_DIR / "norm_stats.npz"
    if not model_path.exists() or not stats_path.exists():
        raise FileNotFoundError("Trained model not found. Run train.py first.")

    model = tf.keras.models.load_model(model_path)
    stats = np.load(stats_path)
    feat = (extract_mfcc(load_audio(path)) - stats["mean"]) / stats["std"]
    probs = model.predict(feat[None, ...], verbose=0)[0]
    order = np.argsort(probs)[::-1]
    return [(LABELS[i], float(probs[i])) for i in order]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="path to a .wav file")
    parser.add_argument("--model", default="cnn_lstm", choices=["cnn", "lstm", "cnn_lstm"])
    args = parser.parse_args()

    results = predict_file(args.audio, args.model)
    print(f"\nPredicted emotion: {results[0][0]}  ({results[0][1] * 100:.1f}%)\n")
    for label, p in results:
        print(f"{label:10s} {p * 100:5.1f}%")