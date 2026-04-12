"""
Configuração dos modelos de Machine Learning.
"""
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from config import RANDOM_STATE


def get_models() -> dict:
    """
    Retorna dicionário com os modelos configurados.
    Todos usam class_weight='balanced' para lidar com desbalanceamento.
    """
    return {
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        ),
    }


def get_param_grids() -> dict:
    """
    Retorna grids de hiperparâmetros para GridSearchCV.
    Só SVM e RF têm grid search (MLP é caro demais).
    """
    return {
        "SVM": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto"],
        },
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
        },
    }
