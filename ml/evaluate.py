"""
Avaliação de modelos: cross-validation, métricas, visualizações.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from config import CV_FOLDS, RESULTS_DIR, RANDOM_STATE


def cross_validate_model(model, X, y, n_folds=CV_FOLDS) -> dict:
    """
    Stratified K-Fold cross-validation.

    Returns:
        Dict com accuracy_mean, accuracy_std, f1_mean, f1_std
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    acc_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted", n_jobs=-1)

    return {
        "accuracy_mean": float(np.mean(acc_scores)),
        "accuracy_std": float(np.std(acc_scores)),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "accuracy_scores": acc_scores.tolist(),
        "f1_scores": f1_scores.tolist(),
    }


def evaluate_model(model, X_test, y_test, labels) -> dict:
    """
    Avalia modelo no conjunto de teste.

    Returns:
        Dict com accuracy, f1_weighted, classification_report, confusion_matrix
    """
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=labels)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": report,
        "classification_report_str": report_str,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(cm, labels, model_name, save_path=None):
    """Gera heatmap da matriz de confusão."""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalizar para porcentagens
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        square=True,
        cbar_kws={"label": "% das amostras"},
    )

    ax.set_xlabel("Predito", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(f"Matriz de Confusão - {model_name}\n(valores em %)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Matriz de confusão salva em: {save_path}")


def plot_model_comparison(results: dict, save_path=None):
    """Gera gráfico comparativo entre modelos."""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "model_comparison.png")

    models = list(results.keys())
    accuracies = [results[m]["test_accuracy"] for m in models]
    f1_scores_vals = [results[m]["test_f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Acurácia", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, f1_scores_vals, width, label="F1-Score (weighted)", color="#DD8452")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Comparação de Modelos - Speech Emotion Recognition", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Adicionar valores nas barras
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico de comparação salvo em: {save_path}")
