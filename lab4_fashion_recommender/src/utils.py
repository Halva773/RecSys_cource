"""Небольшие независимые от CV-библиотек утилиты."""
from __future__ import annotations

import numpy as np


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """Нормирует каждую строку; нулевые векторы остаются нулевыми."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)
