"""
Audio helpers: loading and conversion.
"""
import os
import numpy as np
import librosa
import soundfile as sf
from config import SAMPLE_RATE, DURATION


def load_audio(file_path: str, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """
    Load an audio file, convert to mono, resample and pad/trim to a fixed duration.

    Args:
        file_path: Path to the audio file
        sr: Taxa de amostragem desejada
        duration: Duration in seconds (zero-padded or truncated)

    Returns:
        numpy array holding the normalised audio signal
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True, duration=duration)

    # Pad or trim to the fixed duration
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
    Convert any audio format to mono WAV.
    Usado para converter WebM do microfone do browser.

    Args:
        input_path: Caminho do arquivo de entrada
        output_path: Path to the output WAV file
        sr: Taxa de amostragem

    Returns:
        Caminho do arquivo convertido
    """
    try:
        y, orig_sr = librosa.load(input_path, sr=sr, mono=True)
        sf.write(output_path, y, sr)
        return output_path
    except Exception:
        # Fall back to pydub for the more exotic formats (WebM, OGG)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_channels(1).set_frame_rate(sr)
            audio.export(output_path, format="wav")
            return output_path
        except Exception as e:
            raise RuntimeError(f"Could not convert the audio: {e}")
