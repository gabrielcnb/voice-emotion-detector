"""
Download do dataset RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).

Fonte: https://zenodo.org/record/1188976
Citação: Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of
Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and
vocal expressions in North American English. PLoS ONE 13(5): e0196391.
"""
import os
import sys
import zipfile
import requests
from tqdm import tqdm
from config import DATA_DIR, RAVDESS_DIR


RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
ZIP_PATH = os.path.join(DATA_DIR, "Audio_Speech_Actors_01-24.zip")


def download_file(url: str, dest: str):
    """Download com progress bar."""
    print(f"Baixando RAVDESS de {url}...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Download"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"Download completo: {dest}")


def extract_zip(zip_path: str, dest_dir: str):
    """Extrai ZIP para o diretório de destino."""
    print(f"Extraindo para {dest_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print("Extração completa.")


def count_wav_files(directory: str) -> int:
    """Conta arquivos .wav recursivamente."""
    count = 0
    for root, _, files in os.walk(directory):
        count += sum(1 for f in files if f.endswith(".wav"))
    return count


def main():
    # Verifica se já existe
    wav_count = count_wav_files(RAVDESS_DIR)
    if wav_count >= 1400:
        print(f"RAVDESS já baixado: {wav_count} arquivos .wav encontrados em {RAVDESS_DIR}")
        return

    # Download
    if not os.path.exists(ZIP_PATH):
        download_file(RAVDESS_URL, ZIP_PATH)
    else:
        print(f"ZIP já existe: {ZIP_PATH}")

    # Extract
    extract_zip(ZIP_PATH, RAVDESS_DIR)

    # Verificar
    wav_count = count_wav_files(RAVDESS_DIR)
    print(f"\nTotal de arquivos .wav: {wav_count}")

    if wav_count < 1400:
        print("AVISO: Esperado ~1440 arquivos. Verifique o download.")
    else:
        print("Dataset RAVDESS pronto!")

    # Limpar ZIP para economizar espaço
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print("ZIP removido para economizar espaço.")


if __name__ == "__main__":
    main()
