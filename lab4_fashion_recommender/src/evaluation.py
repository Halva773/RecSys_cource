"""Offline-оценка image retrieval через согласованность с известными ярлыками товара."""
from __future__ import annotations

import numpy as np

from .index import EmbeddingIndex


def evaluate_category_coherence(index: EmbeddingIndex, top_k: int = 10, n_queries: int = 500, random_state: int = 42) -> dict[str, float]:
    """Считает Precision@K по articleType и subCategory.

    В датасете нет пользовательских кликов, поэтому одинаковый тип товара служит
    прокси-сигналом релевантности. Это не персонализация, а прозрачная offline
    метрика для сравнения visual-retrieval моделей на одних и тех же запросах.
    """
    if len(index.catalog) < 2:
        return {"article_type_precision_at_k": 0.0, "subcategory_precision_at_k": 0.0, "queries": 0.0}

    rng = np.random.default_rng(random_state)
    positions = rng.choice(len(index.catalog), size=min(n_queries, len(index.catalog)), replace=False)
    article_scores: list[float] = []
    subcategory_scores: list[float] = []
    for position in positions:
        query = index.catalog.iloc[int(position)]
        recommendations = index.recommend(int(query["id"]), top_k=top_k)
        if recommendations.empty:
            continue
        article_scores.append(float(recommendations["articleType"].eq(query["articleType"]).mean()))
        subcategory_scores.append(float(recommendations["subCategory"].eq(query["subCategory"]).mean()))
    return {
        "article_type_precision_at_k": float(np.mean(article_scores)) if article_scores else 0.0,
        "subcategory_precision_at_k": float(np.mean(subcategory_scores)) if subcategory_scores else 0.0,
        "queries": float(len(article_scores)),
    }
