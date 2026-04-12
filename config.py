"""
Configuração central do projeto Voice Emotion Detector.
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAVDESS_DIR = os.path.join(DATA_DIR, "ravdess")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Criar diretórios se não existem
for d in [DATA_DIR, RAVDESS_DIR, MODELS_DIR, RESULTS_DIR, CACHE_DIR, UPLOAD_DIR]:
    os.makedirs(d, exist_ok=True)

# Audio settings
SAMPLE_RATE = 22050
DURATION = 3.0  # seconds - pad or trim to this
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# RAVDESS emotion mapping (skip 2=calm)
EMOTION_MAP = {
    1: "neutral",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgusted",
    8: "surprised",
}

EMOTION_LABELS = sorted(set(EMOTION_MAP.values()))
NUM_CLASSES = len(EMOTION_LABELS)

# Emotion emojis for UI
EMOTION_EMOJIS = {
    "neutral": "\U0001F610",    # 😐
    "happy": "\U0001F60A",      # 😊
    "sad": "\U0001F622",        # 😢
    "angry": "\U0001F620",      # 😠
    "fearful": "\U0001F628",    # 😨
    "disgusted": "\U0001F922",  # 🤢
    "surprised": "\U0001F632",  # 😲
}

# Training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Model file names
BEST_MODEL_FILE = "best_model.joblib"
SCALER_FILE = "scaler.joblib"
LABEL_ENCODER_FILE = "label_encoder.joblib"

# Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = int(os.environ.get("PORT", 5000))
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
