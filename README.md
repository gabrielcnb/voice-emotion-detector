# 🎙️ Voice Emotion Detector

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

Sistema de **Reconhecimento de Emoção por Voz** (Speech Emotion Recognition - SER) que classifica áudio em 7 emoções usando Machine Learning clássico com features acústicas extraídas via librosa.

## Arquitetura

```
Áudio → Pré-processamento → Extração de Features (79) → Scaler → Modelo ML → Emoção
         (mono, 22050Hz,      MFCCs, pitch, ZCR,         Standard    SVM/RF/MLP
          3s, normalizado)     espectro, chroma, tonnetz   Scaler
```

## Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)

- **Fonte:** [Zenodo](https://zenodo.org/record/1188976)
- **Amostras:** ~1260 (após remover classe "calm")
- **Atores:** 24 (12 homens, 12 mulheres)
- **Emoções:** 7 classes — neutro, feliz, triste, raiva, medo, nojo, surpresa
- **Citação:** Livingstone SR, Russo FA (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).* PLoS ONE 13(5): e0196391.

## Features Extraídas (79 dimensões)

| Feature | Dimensões | Descrição |
|---------|-----------|-----------|
| MFCCs | 26 | 13 médias + 13 desvios padrão |
| Delta MFCCs | 13 | Derivada temporal dos MFCCs |
| Pitch (pyin) | 5 | Média, std, max, min, range |
| ZCR | 2 | Taxa de cruzamento por zero |
| RMS | 2 | Energia do sinal |
| Spectral Centroid | 2 | Centro de massa espectral |
| Spectral Bandwidth | 2 | Largura de banda espectral |
| Spectral Rolloff | 2 | Rolloff espectral |
| Spectral Contrast | 7 | Contraste em 7 bandas |
| Chroma | 12 | 12 classes de pitch |
| Tonnetz | 6 | Relações tonais |

## Modelos

- **SVM** (RBF kernel) — GridSearchCV sobre C e gamma
- **Random Forest** (200 árvores) — GridSearchCV sobre n_estimators e max_depth
- **MLP** (256→128→64) — Early stopping, adam optimizer

Todos usam `class_weight='balanced'` para lidar com desbalanceamento entre classes.

## Instalação e Uso

```bash
# 1. Clonar e entrar no diretório
cd voice-emotion-detector

# 2. Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar dataset RAVDESS (~200MB)
python download_dataset.py

# 5. Treinar modelos (pode levar 5-15 min)
python train.py

# 6. Iniciar aplicação web
python app.py
# Acesse http://localhost:5000
```

## Interface Web

- **Upload:** Arraste ou selecione um arquivo de áudio (WAV, MP3, OGG, FLAC)
- **Gravação:** Use o microfone do navegador para gravar em tempo real
- **Resultado:** Emoção detectada com confiança e distribuição de probabilidades

## Limitações (Discussão Honesta)

SER com 7 classes é um problema **difícil**. Resultados esperados:

- **Acurácia típica:** 55-70% (modelos clássicos com features manuais)
- **Confusões comuns:** Neutro↔Triste, Medo↔Surpresa, Feliz↔Surpresa
- **Viés do dataset:** RAVDESS é atuado (acted speech), não espontâneo
- **Generalização:** Performance cai significativamente em áudio fora do domínio (ruído, sotaques, idiomas diferentes do inglês)
- **Features manuais vs. deep learning:** Modelos end-to-end (wav2vec2, HuBERT) alcançam 70-80%+ mas requerem GPU

## Estrutura do Projeto

```
voice-emotion-detector/
├── app.py                  # Servidor Flask
├── train.py                # Pipeline de treino + avaliação
├── download_dataset.py     # Download RAVDESS
├── config.py               # Configurações centrais
├── audio/
│   ├── processor.py        # Extração de 79 features
│   └── utils.py            # Load/convert áudio
├── ml/
│   ├── models.py           # SVM, RF, MLP configs
│   ├── evaluate.py         # CV, métricas, plots
│   └── features.py         # StandardScaler wrapper
├── templates/index.html    # UI principal
├── static/                 # CSS + JS
└── results/                # Plots de avaliação
```

## Referências

1. Livingstone SR, Russo FA (2018). The RAVDESS. PLoS ONE 13(5): e0196391.
2. McFee B et al. (2015). librosa: Audio and Music Signal Analysis in Python.
3. Pedregosa F et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.
