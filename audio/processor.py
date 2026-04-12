"""
Extração de features de áudio usando librosa.
~79 features por arquivo de áudio.
"""
import os
import hashlib
import numpy as np
import librosa
from config import SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, CACHE_DIR
from .utils import load_audio


def _safe_mean_std(feat: np.ndarray) -> np.ndarray:
    """Calcula mean e std de forma segura para features 2D."""
    if feat.ndim == 1:
        return np.array([np.mean(feat), np.std(feat)])
    return np.concatenate([np.mean(feat, axis=1), np.std(feat, axis=1)])


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extrai ~79 features de um sinal de áudio.

    Features:
        - MFCCs: 13 means + 13 stds = 26
        - Delta MFCCs: 13 means = 13
        - Pitch (pyin): mean, std, max, min, range = 5
        - ZCR: mean, std = 2
        - RMS: mean, std = 2
        - Spectral centroid: mean, std = 2
        - Spectral bandwidth: mean, std = 2
        - Spectral rolloff: mean, std = 2
        - Spectral contrast (7 bands): 7 means = 7
        - Chroma (12 bins): 12 means = 12
        - Tonnetz (6 dims): 6 means = 6
        Total = 79

    Args:
        y: Sinal de áudio (numpy array)
        sr: Taxa de amostragem

    Returns:
        numpy array com 79 features
    """
    features = []

    # 1. MFCCs (26 features: 13 mean + 13 std)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(mfccs, axis=1))       # 13
    features.append(np.std(mfccs, axis=1))         # 13

    # 2. Delta MFCCs (13 features: means only)
    delta_mfccs = librosa.feature.delta(mfccs)
    features.append(np.mean(delta_mfccs, axis=1))  # 13

    # 3. Pitch via pyin (5 features)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, hop_length=HOP_LENGTH
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
    if len(f0_valid) == 0:
        f0_valid = np.array([0.0])
    features.append(np.array([
        np.mean(f0_valid),
        np.std(f0_valid),
        np.max(f0_valid),
        np.min(f0_valid),
        np.max(f0_valid) - np.min(f0_valid),  # range
    ]))  # 5

    # 4. Zero Crossing Rate (2 features)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    features.append(np.array([np.mean(zcr), np.std(zcr)]))  # 2

    # 5. RMS Energy (2 features)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    features.append(np.array([np.mean(rms), np.std(rms)]))  # 2

    # 6. Spectral Centroid (2 features)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    features.append(np.array([np.mean(cent), np.std(cent)]))  # 2

    # 7. Spectral Bandwidth (2 features)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    features.append(np.array([np.mean(bw), np.std(bw)]))  # 2

    # 8. Spectral Rolloff (2 features)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    features.append(np.array([np.mean(rolloff), np.std(rolloff)]))  # 2

    # 9. Spectral Contrast (7 features)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(contrast, axis=1))  # 7

    # 10. Chroma (12 features)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(chroma, axis=1))  # 12

    # 11. Tonnetz (6 features)
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    features.append(np.mean(tonnetz, axis=1))  # 6

    result = np.concatenate(features)
    return result


def extract_features_from_file(
    file_path: str,
    sr: int = SAMPLE_RATE,
    use_cache: bool = True
) -> np.ndarray:
    """
    Extrai features de um arquivo de áudio com suporte a cache.

    Args:
        file_path: Caminho do arquivo de áudio
        sr: Taxa de amostragem
        use_cache: Se True, salva/carrega features de cache .npy

    Returns:
        numpy array com features
    """
    if use_cache:
        # Hash baseado no caminho + tamanho + mtime do arquivo
        stat = os.stat(file_path)
        cache_key = hashlib.md5(
            f"{file_path}_{stat.st_size}_{stat.st_mtime}".encode()
        ).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")

        if os.path.exists(cache_path):
            return np.load(cache_path)

    y = load_audio(file_path, sr=sr)
    features = extract_features(y, sr=sr)

    if use_cache:
        np.save(cache_path, features)

    return features
