"""
dataset.py - read RAVDESS files, parse labels and build train/test arrays.

RAVDESS filename format (7 fields separated by '-'):
    03-01-06-01-02-01-12.wav
    modality - vocal channel - EMOTION - intensity - statement - repetition - ACTOR
"""
from pathlib import Path

import numpy as np

from features import augment, extract_mfcc, load_audio

# ----------------------------- Settings -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "RAVDESS"
TEST_ACTORS = {21, 22, 23, 24}      # speakers kept out of training
SEED = 42

EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
LABELS = sorted(set(EMOTIONS.values()))
LABEL2ID = {name: i for i, name in enumerate(LABELS)}


def parse_filename(path):
    """Return (label_id, actor_number) for a RAVDESS file, or None if invalid."""
    parts = Path(path).stem.split("-")
    if len(parts) != 7:
        return None
    if parts[0] != "03" or parts[1] != "01":     # keep speech audio only
        return None
    emotion = EMOTIONS.get(parts[2])
    if emotion is None or not parts[6].isdigit():
        return None
    return LABEL2ID[emotion], int(parts[6])


def find_audio_files(data_dir=DATA_DIR):
    """Find all RAVDESS .wav files (also inside nested folders).

    Files with the same name found twice (e.g. a duplicated outer folder in
    the Kaggle zip) are used only once.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    unique = {}
    for path in sorted(data_dir.rglob("*.wav")):
        if parse_filename(path) is not None:
            unique.setdefault(path.name, path)
    if not unique:
        raise FileNotFoundError(
            f"No valid RAVDESS .wav files found under {data_dir}. "
            "Put the Actor_01 ... Actor_24 folders inside data/RAVDESS/."
        )
    return list(unique.values())


def build_dataset(data_dir=DATA_DIR, use_augmentation=True):
    """Extract MFCC features and split by speaker.

    Returns X_train, y_train, X_test, y_test as numpy arrays.
    X shape: (samples, 130, 80). Augmented copies are added to train only.
    """
    rng = np.random.default_rng(SEED)
    files = find_audio_files(data_dir)
    print(f"Found {len(files)} audio files")

    X_train, y_train, X_test, y_test = [], [], [], []
    skipped = 0

    for i, path in enumerate(files, start=1):
        label, actor = parse_filename(path)
        try:
            y = load_audio(path)
            feat = extract_mfcc(y)
            aug_feat = None
            if actor not in TEST_ACTORS and use_augmentation:
                aug_feat = extract_mfcc(augment(y, rng))
        except Exception as err:          # corrupted file: skip, do not crash
            skipped += 1
            print(f"Skipped {path.name}: {err}")
            continue

        if actor in TEST_ACTORS:
            X_test.append(feat)
            y_test.append(label)
        else:
            X_train.append(feat)
            y_train.append(label)
            if aug_feat is not None:
                X_train.append(aug_feat)
                y_train.append(label)

        if i % 100 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)} files")

    if not X_train or not X_test:
        raise RuntimeError("Train or test set is empty. Check the dataset folder.")
    if skipped:
        print(f"Warning: {skipped} files were skipped")

    return (np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int64),
            np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.int64))


def normalize_data(X_train, X_test):
    """Standardize with training statistics only. Returns X_train, X_test, mean, std."""
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1)) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


if __name__ == "__main__":
    # Quick check:  python src/dataset.py
    X_tr, y_tr, X_te, y_te = build_dataset()
    print("Train:", X_tr.shape, y_tr.shape)
    print("Test: ", X_te.shape, y_te.shape)
    print("Labels:", LABELS)
    print("Train class counts:", np.bincount(y_tr, minlength=len(LABELS)))