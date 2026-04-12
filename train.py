"""
Pipeline de treinamento e avaliação do modelo de reconhecimento de emoção por voz.

Etapas:
1. Scan dos arquivos RAVDESS
2. Extração de features (com cache)
3. Split estratificado 80/20
4. Fit scaler apenas no treino
5. GridSearchCV para SVM e RF
6. Treinar MLP
7. Avaliar todos no test set
8. Salvar melhor modelo + scaler
9. Gerar plots
"""
import os
import sys
import time
import json
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from config import (
    RAVDESS_DIR, MODELS_DIR, RESULTS_DIR,
    EMOTION_MAP, EMOTION_LABELS,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    BEST_MODEL_FILE, SCALER_FILE, LABEL_ENCODER_FILE,
)
from audio.processor import extract_features_from_file
from ml.features import FeatureScaler
from ml.models import get_models, get_param_grids
from ml.evaluate import (
    evaluate_model,
    cross_validate_model,
    plot_confusion_matrix,
    plot_model_comparison,
)


def scan_ravdess(ravdess_dir: str) -> list:
    """
    Escaneia diretório RAVDESS e retorna lista de (filepath, emotion_label).

    Formato do nome: {modality}-{vocal}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
    Emoção é o 3o campo. Pula emoção 2 (calm).
    """
    samples = []
    skipped = 0

    for root, dirs, files in os.walk(ravdess_dir):
        for f in files:
            if not f.endswith(".wav"):
                continue
            parts = f.replace(".wav", "").split("-")
            if len(parts) != 7:
                skipped += 1
                continue

            emotion_id = int(parts[2])
            if emotion_id not in EMOTION_MAP:
                skipped += 1
                continue

            emotion_label = EMOTION_MAP[emotion_id]
            filepath = os.path.join(root, f)
            samples.append((filepath, emotion_label))

    print(f"Arquivos encontrados: {len(samples)} (ignorados: {skipped})")
    return samples


def extract_all_features(samples: list) -> tuple:
    """Extrai features de todos os samples."""
    X = []
    y = []
    errors = 0

    for filepath, label in tqdm(samples, desc="Extraindo features"):
        try:
            feats = extract_features_from_file(filepath)
            X.append(feats)
            y.append(label)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Erro em {os.path.basename(filepath)}: {e}")

    if errors > 0:
        print(f"  Total de erros na extração: {errors}")

    return np.array(X), np.array(y)


def main():
    print("=" * 70)
    print("  SPEECH EMOTION RECOGNITION - Pipeline de Treinamento")
    print("=" * 70)

    # 1. Scan dataset
    print("\n[1/7] Escaneando dataset RAVDESS...")
    samples = scan_ravdess(RAVDESS_DIR)
    if len(samples) == 0:
        print("ERRO: Nenhum arquivo encontrado. Execute download_dataset.py primeiro.")
        sys.exit(1)

    # Distribuição das classes
    from collections import Counter
    dist = Counter(s[1] for s in samples)
    print("\nDistribuição das emoções:")
    for emotion in EMOTION_LABELS:
        count = dist.get(emotion, 0)
        print(f"  {emotion:12s}: {count:4d} ({100*count/len(samples):.1f}%)")

    # 2. Extract features
    print(f"\n[2/7] Extraindo {len(samples)} features de áudio...")
    t0 = time.time()
    X, y = extract_all_features(samples)
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Tempo: {time.time()-t0:.1f}s")

    # Verificar NaN/Inf
    nan_count = np.sum(np.isnan(X))
    inf_count = np.sum(np.isinf(X))
    if nan_count > 0 or inf_count > 0:
        print(f"  AVISO: {nan_count} NaN, {inf_count} Inf encontrados. Substituindo por 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Encode labels
    le = LabelEncoder()
    le.fit(EMOTION_LABELS)
    y_encoded = le.transform(y)
    joblib.dump(le, os.path.join(MODELS_DIR, LABEL_ENCODER_FILE))

    # 4. Stratified split
    print("\n[3/7] Split estratificado 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Teste:  {X_test.shape[0]} amostras")

    # 5. Fit scaler no treino apenas
    print("\n[4/7] Ajustando scaler no conjunto de treino...")
    scaler = FeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaler.save()
    print("  Scaler salvo.")

    # 6. Train models
    models = get_models()
    param_grids = get_param_grids()
    all_results = {}
    best_accuracy = 0
    best_model_name = None
    best_model_obj = None

    # SVM com GridSearch
    print("\n[5/7] Treinando modelos...")
    print("\n--- SVM (GridSearchCV) ---")
    t0 = time.time()
    svm_grid = GridSearchCV(
        models["SVM"],
        param_grids["SVM"],
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    svm_grid.fit(X_train_scaled, y_train)
    print(f"  Melhores parâmetros: {svm_grid.best_params_}")
    print(f"  Melhor F1 (CV): {svm_grid.best_score_:.4f}")
    print(f"  Tempo: {time.time()-t0:.1f}s")
    models["SVM"] = svm_grid.best_estimator_

    # RF com GridSearch
    print("\n--- Random Forest (GridSearchCV) ---")
    t0 = time.time()
    rf_grid = GridSearchCV(
        models["RandomForest"],
        param_grids["RandomForest"],
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    rf_grid.fit(X_train_scaled, y_train)
    print(f"  Melhores parâmetros: {rf_grid.best_params_}")
    print(f"  Melhor F1 (CV): {rf_grid.best_score_:.4f}")
    print(f"  Tempo: {time.time()-t0:.1f}s")
    models["RandomForest"] = rf_grid.best_estimator_

    # MLP (sem grid search - muito caro)
    print("\n--- MLP ---")
    t0 = time.time()
    models["MLP"].fit(X_train_scaled, y_train)
    print(f"  Tempo: {time.time()-t0:.1f}s")

    # 7. Evaluate all on test set
    print("\n[6/7] Avaliando modelos no conjunto de teste...")
    labels = le.classes_.tolist()

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        result = evaluate_model(model, X_test_scaled, y_test, labels)

        print(f"\n  Acurácia: {result['accuracy']:.4f}")
        print(f"  F1 (weighted): {result['f1_weighted']:.4f}")
        print(f"\n{result['classification_report_str']}")

        # Cross-validation no dataset completo (para referência)
        X_all_scaled = np.vstack([X_train_scaled, X_test_scaled])
        y_all = np.concatenate([y_train, y_test])
        cv_result = cross_validate_model(model, X_all_scaled, y_all)
        print(f"  CV({CV_FOLDS}-fold) Acurácia: {cv_result['accuracy_mean']:.4f} +/- {cv_result['accuracy_std']:.4f}")
        print(f"  CV({CV_FOLDS}-fold) F1:       {cv_result['f1_mean']:.4f} +/- {cv_result['f1_std']:.4f}")

        all_results[name] = {
            "test_accuracy": result["accuracy"],
            "test_f1": result["f1_weighted"],
            "cv_accuracy_mean": cv_result["accuracy_mean"],
            "cv_accuracy_std": cv_result["accuracy_std"],
            "cv_f1_mean": cv_result["f1_mean"],
            "cv_f1_std": cv_result["f1_std"],
        }

        # Confusion matrix
        plot_confusion_matrix(result["confusion_matrix"], labels, name)

        # Track best
        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            best_model_name = name
            best_model_obj = model

    # 8. Save best model
    print(f"\n[7/7] Salvando melhor modelo: {best_model_name} (acurácia: {best_accuracy:.4f})")
    model_path = os.path.join(MODELS_DIR, BEST_MODEL_FILE)
    joblib.dump(best_model_obj, model_path)
    print(f"  Modelo salvo em: {model_path}")

    # Save metadata
    metadata = {
        "best_model": best_model_name,
        "best_accuracy": best_accuracy,
        "all_results": all_results,
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "emotion_labels": labels,
    }
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata salvo em: {metadata_path}")

    # Model comparison plot
    plot_model_comparison(all_results)

    # Summary
    print("\n" + "=" * 70)
    print("  RESUMO FINAL")
    print("=" * 70)
    print(f"\n{'Modelo':<20} {'Acurácia':>10} {'F1':>10} {'CV Acc':>12} {'CV F1':>12}")
    print("-" * 64)
    for name, r in all_results.items():
        marker = " <-- MELHOR" if name == best_model_name else ""
        print(
            f"{name:<20} {r['test_accuracy']:>10.4f} {r['test_f1']:>10.4f} "
            f"{r['cv_accuracy_mean']:>6.4f}+/-{r['cv_accuracy_std']:.4f} "
            f"{r['cv_f1_mean']:>6.4f}+/-{r['cv_f1_std']:.4f}{marker}"
        )

    print(f"\nArquivos gerados:")
    print(f"  Modelo:  {os.path.join(MODELS_DIR, BEST_MODEL_FILE)}")
    print(f"  Scaler:  {os.path.join(MODELS_DIR, SCALER_FILE)}")
    print(f"  Plots:   {RESULTS_DIR}/")
    print("\nPipeline completo!")


if __name__ == "__main__":
    main()
