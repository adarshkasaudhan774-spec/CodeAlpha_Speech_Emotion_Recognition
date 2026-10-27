"""
app.py - Streamlit demo for speech emotion recognition.

Run from the project folder:
    streamlit run app.py
"""
import os
import sys
import tempfile
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))          # so we can import our modules

from dataset import LABELS                      # noqa: E402
from features import extract_mfcc, load_audio   # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
EMOJI = {
    "angry": "😠", "calm": "😌", "disgust": "🤢", "fearful": "😨",
    "happy": "😄", "neutral": "😐", "sad": "😢", "surprised": "😲",
}


@st.cache_resource
def load_model(name):
    return keras.models.load_model(OUTPUT_DIR / f"{name}_best.keras")


@st.cache_resource
def load_stats():
    stats = np.load(OUTPUT_DIR / "norm_stats.npz")
    return stats["mean"], stats["std"]


def predict(audio_bytes, model_name):
    """Return an array with one probability per emotion (same order as LABELS)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()
        y = load_audio(tmp.name)
    finally:
        tmp.close()
        try:
            os.remove(tmp.name)
        except OSError:
            pass

    mean, std = load_stats()
    feat = (extract_mfcc(y) - mean) / std
    return load_model(model_name).predict(feat[None, ...], verbose=0)[0]


# ------------------------------- Page ---------------------------------
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload or record a short speech clip and the model predicts the emotion.")

available = sorted(p.name[: -len("_best.keras")] for p in OUTPUT_DIR.glob("*_best.keras"))
if not available or not (OUTPUT_DIR / "norm_stats.npz").exists():
    st.error("No trained model found in the 'outputs' folder. "
             "Run `python src/train.py` first.")
    st.stop()

default_index = available.index("cnn") if "cnn" in available else 0
model_name = st.sidebar.selectbox("Model", available, index=default_index)
st.sidebar.info(
    "Trained on RAVDESS (acted English speech, 8 emotions). "
    "Accuracy on unseen speakers is roughly 50-60%, and the confidence "
    "shown is not a guarantee that the prediction is right."
)

audio_bytes = None
tab_upload, tab_record = st.tabs(["Upload a file", "Record with microphone"])
with tab_upload:
    uploaded = st.file_uploader("WAV file (1-3 seconds works best)", type=["wav"])
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
with tab_record:
    recorded = st.audio_input("Record a short clip")
    if recorded is not None:
        audio_bytes = recorded.getvalue()      # a recording replaces an upload

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    with st.spinner("Analyzing..."):
        probs = predict(audio_bytes, model_name)

    top = int(np.argmax(probs))
    st.subheader(f"{EMOJI[LABELS[top]]} {LABELS[top].capitalize()}  ({probs[top] * 100:.1f}%)")
    chart = pd.Series(probs * 100, index=[l.capitalize() for l in LABELS],
                      name="Probability (%)")
    st.bar_chart(chart)