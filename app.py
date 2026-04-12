"""
Flask web server para Speech Emotion Recognition.
"""
import os
import uuid
import traceback
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from config import (
    MODELS_DIR, UPLOAD_DIR, EMOTION_LABELS, EMOTION_EMOJIS,
    BEST_MODEL_FILE, SCALER_FILE, LABEL_ENCODER_FILE,
    FLASK_HOST, FLASK_PORT, MAX_CONTENT_LENGTH,
)
from audio.processor import extract_features
from audio.utils import load_audio, convert_to_wav


app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Load model on startup
model = None
scaler = None
label_encoder = None


def load_model():
    """Carrega modelo, scaler e label encoder."""
    global model, scaler, label_encoder

    model_path = os.path.join(MODELS_DIR, BEST_MODEL_FILE)
    scaler_path = os.path.join(MODELS_DIR, SCALER_FILE)
    le_path = os.path.join(MODELS_DIR, LABEL_ENCODER_FILE)

    if not all(os.path.exists(p) for p in [model_path, scaler_path, le_path]):
        print("AVISO: Modelo não encontrado. Execute train.py primeiro.")
        return False

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(le_path)
    print(f"Modelo carregado: {model.__class__.__name__}")
    return True


def predict_emotion(file_path: str) -> dict:
    """
    Prediz emoção de um arquivo de áudio.

    Returns:
        Dict com emotion, emoji, confidence, probabilities
    """
    if model is None:
        raise RuntimeError("Modelo não carregado. Execute train.py primeiro.")

    # Load and extract features
    y = load_audio(file_path)
    features = extract_features(y)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    emotion = label_encoder.inverse_transform([prediction])[0]

    # Build probability dict
    prob_dict = {}
    for i, label in enumerate(label_encoder.classes_):
        prob_dict[label] = float(probabilities[i])

    confidence = float(max(probabilities))

    return {
        "emotion": emotion,
        "emoji": EMOTION_EMOJIS.get(emotion, ""),
        "confidence": confidence,
        "probabilities": prob_dict,
    }


@app.route("/")
def index():
    """Serve a página principal."""
    return render_template("index.html")


@app.route("/api/predict/upload", methods=["POST"])
def predict_upload():
    """Endpoint para upload de arquivo de áudio."""
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Arquivo vazio"}), 400

    try:
        # Save temp file
        ext = os.path.splitext(file.filename)[1] or ".wav"
        temp_name = f"{uuid.uuid4()}{ext}"
        temp_path = os.path.join(UPLOAD_DIR, temp_name)
        file.save(temp_path)

        # Convert to WAV if needed
        if ext.lower() not in [".wav"]:
            wav_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
            convert_to_wav(temp_path, wav_path)
            os.remove(temp_path)
            temp_path = wav_path

        result = predict_emotion(temp_path)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erro ao processar áudio: {str(e)}"}), 500


@app.route("/api/predict/record", methods=["POST"])
def predict_record():
    """Endpoint para gravação do microfone (WebM blob)."""
    if "audio" not in request.files:
        return jsonify({"error": "Nenhum áudio enviado"}), 400

    audio = request.files["audio"]

    try:
        # Save WebM blob
        temp_name = f"{uuid.uuid4()}.webm"
        temp_path = os.path.join(UPLOAD_DIR, temp_name)
        audio.save(temp_path)

        # Convert to WAV
        wav_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
        convert_to_wav(temp_path, wav_path)
        os.remove(temp_path)

        result = predict_emotion(wav_path)

        # Cleanup
        if os.path.exists(wav_path):
            os.remove(wav_path)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erro ao processar gravação: {str(e)}"}), 500


@app.route("/api/status")
def status():
    """Health check."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "emotion_labels": EMOTION_LABELS,
    })


if __name__ == "__main__":
    load_model()
    print(f"\nServidor iniciando em http://localhost:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
