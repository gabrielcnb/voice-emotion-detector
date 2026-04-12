"""
Utilitários de áudio: carregamento e conversão.
"""
import os
import numpy as np
import librosa
import soundfile as sf
from config import SAMPLE_RATE, DURATION


def load_audio(file_path: str, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """
    Carrega arquivo de áudio, converte para mono, resample e pad/trim para duração fixa.

    Args:
        file_path: Caminho do arquivo de áudio
        sr: Taxa de amostragem desejada
        duration: Duração em segundos (pad com zeros ou trunca)

    Returns:
        numpy array com o sinal de áudio normalizado
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True, duration=duration)

    # Pad ou trim para duração fixa
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        y = y[:target_length]

    # Normalizar
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y


def convert_to_wav(input_path: str, output_path: str, sr: int = SAMPLE_RATE) -> str:
    """
    Converte qualquer formato de áudio para WAV mono.
    Usado para converter WebM do microfone do browser.

    Args:
        input_path: Caminho do arquivo de entrada
        output_path: Caminho do arquivo WAV de saída
        sr: Taxa de amostragem

    Returns:
        Caminho do arquivo convertido
    """
    try:
        y, orig_sr = librosa.load(input_path, sr=sr, mono=True)
        sf.write(output_path, y, sr)
        return output_path
    except Exception:
        # Fallback com pydub para formatos mais exóticos (WebM, OGG)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_channels(1).set_frame_rate(sr)
            audio.export(output_path, format="wav")
            return output_path
        except Exception as e:
            raise RuntimeError(f"Não foi possível converter o áudio: {e}")
