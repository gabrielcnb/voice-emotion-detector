# Voice Emotion Detector

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

A **Speech Emotion Recognition (SER)** system that classifies audio into 7 emotions using classical Machine Learning with acoustic features extracted via librosa.

## Architecture

```
Audio → Pre-processing → Feature Extraction (79) → Scaler → ML Model → Emotion
         (mono, 22050Hz,   MFCCs, pitch, ZCR,        Standard   SVM/RF/MLP
          3s, normalized)   spectrum, chroma, tonnetz  Scaler
```

## Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)

- **Source:** [Zenodo](https://zenodo.org/record/1188976)
- **Samples:** ~1260 (after removing the "calm" class)
- **Actors:** 24 (12 male, 12 female)
- **Emotions:** 7 classes (neutral, happy, sad, angry, fearful, disgust, surprised)
- **Citation:** Livingstone SR, Russo FA (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).* PLoS ONE 13(5): e0196391.

## Extracted Features (79 dimensions)

| Feature | Dimensions | Description |
|---------|-----------|-----------|
| MFCCs | 26 | 13 means + 13 standard deviations |
| Delta MFCCs | 13 | Temporal derivative of the MFCCs |
| Pitch (pyin) | 5 | Mean, std, max, min, range |
| ZCR | 2 | Zero-crossing rate |
| RMS | 2 | Signal energy |
| Spectral Centroid | 2 | Spectral center of mass |
| Spectral Bandwidth | 2 | Spectral bandwidth |
| Spectral Rolloff | 2 | Spectral rolloff |
| Spectral Contrast | 7 | Contrast across 7 bands |
| Chroma | 12 | 12 pitch classes |
| Tonnetz | 6 | Tonal relations |

## Models

- **SVM** (RBF kernel): GridSearchCV over C and gamma
- **Random Forest** (200 trees): GridSearchCV over n_estimators and max_depth
- **MLP** (256→128→64): early stopping, adam optimizer

All of them use `class_weight='balanced'` to handle class imbalance.

## Installation and Usage

```bash
# 1. Clone and enter the directory
cd voice-emotion-detector

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the RAVDESS dataset (~200MB)
python download_dataset.py

# 5. Train the models (may take 5-15 min)
python train.py

# 6. Start the web application
python app.py
# Go to http://localhost:5000
```

## Web Interface

- **Upload:** Drag and drop or select an audio file (WAV, MP3, OGG, FLAC)
- **Recording:** Use the browser's microphone to record in real time
- **Result:** Detected emotion with confidence score and probability distribution

## Limitations

SER with 7 classes is a **hard** problem. Expected results:

- **Typical accuracy:** 55-70% (classical models with hand-crafted features)
- **Common confusions:** Neutral↔Sad, Fearful↔Surprised, Happy↔Surprised
- **Dataset bias:** RAVDESS is acted speech, not spontaneous
- **Generalization:** Performance drops significantly on out-of-domain audio (noise, accents, languages other than English)
- **Hand-crafted features vs. deep learning:** End-to-end models (wav2vec2, HuBERT) reach 70-80%+ but require a GPU

## Project Structure

```
voice-emotion-detector/
├── app.py                  # Flask server
├── train.py                # Training + evaluation pipeline
├── download_dataset.py     # RAVDESS download
├── config.py               # Central configuration
├── audio/
│   ├── processor.py        # 79-feature extraction
│   └── utils.py            # Load/convert audio
├── ml/
│   ├── models.py           # SVM, RF, MLP configs
│   ├── evaluate.py         # CV, metrics, plots
│   └── features.py         # StandardScaler wrapper
├── templates/index.html    # Main UI
├── static/                 # CSS + JS
└── results/                # Evaluation plots
```

## References

1. Livingstone SR, Russo FA (2018). The RAVDESS. PLoS ONE 13(5): e0196391.
2. McFee B et al. (2015). librosa: Audio and Music Signal Analysis in Python.
3. Pedregosa F et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.