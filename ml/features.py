"""
Feature scaling com persistência.
"""
import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from config import MODELS_DIR, SCALER_FILE


class FeatureScaler:
    """Wrapper do StandardScaler com save/load."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """Fit no conjunto de treino."""
        self.scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforma features."""
        if not self._fitted:
            raise RuntimeError("Scaler não foi ajustado. Chame fit() primeiro.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit + transform."""
        self.fit(X)
        return self.transform(X)

    def save(self, path: str = None):
        """Salva o scaler em disco."""
        if path is None:
            path = os.path.join(MODELS_DIR, SCALER_FILE)
        joblib.dump(self.scaler, path)

    @classmethod
    def load(cls, path: str = None) -> "FeatureScaler":
        """Carrega scaler do disco."""
        if path is None:
            path = os.path.join(MODELS_DIR, SCALER_FILE)
        instance = cls()
        instance.scaler = joblib.load(path)
        instance._fitted = True
        return instance
