"""
features.py - audio loading, augmentation and MFCC feature extraction.
"""
import librosa
import numpy as np

# ----------------------------- Settings -------------------------------
SR = 22050                 # sampling rate (Hz)
DURATION = 3               # every clip is fixed to 3 seconds
N_MFCC = 40                # number of MFCC coefficients
HOP_LENGTH = 512
N_SAMPLES = SR * DURATION
MAX_LEN = 1 + N_SAMPLES // HOP_LENGTH   # 130 time frames
N_FEATURES = N_MFCC * 2                 # MFCC + delta = 80


def load_audio(path):
    """Load a .wav file, trim silence and fix its length to 3 seconds."""
    y, _ = librosa.load(path, sr=SR, duration=DURATION)
    if len(y) > 0:
        y, _ = librosa.effects.trim(y, top_db=25)
    y = librosa.util.fix_length(y, size=N_SAMPLES)   # pad with zeros or cut
    return y.astype(np.float32)


def augment(y, rng):
    """Return a randomly augmented copy of the signal (same length)."""
    choice = rng.choice(["noise", "pitch", "stretch"])
    if choice == "noise":
        y_aug = y + 0.005 * rng.standard_normal(len(y))
    elif choice == "pitch":
        steps = float(rng.choice([-2, -1, 1, 2]))
        y_aug = librosa.effects.pitch_shift(y, sr=SR, n_steps=steps)
    else:
        rate = float(rng.uniform(0.9, 1.1))
        y_aug = librosa.effects.time_stretch(y, rate=rate)
    y_aug = librosa.util.fix_length(y_aug, size=N_SAMPLES)
    return y_aug.astype(np.float32)


def extract_mfcc(y):
    """Convert a 3-second signal into a (MAX_LEN, N_FEATURES) matrix."""
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    feat = np.concatenate([mfcc, delta], axis=0).T          # (time, 80)
    if feat.shape[0] < MAX_LEN:                              # safety padding
        feat = np.pad(feat, ((0, MAX_LEN - feat.shape[0]), (0, 0)))
    return feat[:MAX_LEN].astype(np.float32)