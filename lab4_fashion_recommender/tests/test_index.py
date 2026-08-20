from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import evaluate_category_coherence
from src.index import EmbeddingIndex


def make_index() -> EmbeddingIndex:
    catalog = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "articleType": ["Tshirts", "Tshirts", "Shoes", "Shoes"],
            "subCategory": ["Topwear", "Topwear", "Shoes", "Shoes"],
            "gender": ["Men", "Men", "Men", "Women"],
            "season": ["Summer", "Summer", "Winter", "Winter"],
            "image_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        }
    )
    embeddings = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype="float32")
    return EmbeddingIndex(catalog=catalog, embeddings=embeddings, method="test")


def test_recommend_excludes_query_and_applies_filters():
    result = make_index().recommend(1, top_k=3, filters={"gender": "Men"})
    assert result["id"].tolist() == [2, 3]


def test_index_roundtrip(tmp_path: Path):
    original = make_index()
    path = tmp_path / "index.npz"
    original.save(path)
    restored = EmbeddingIndex.load(path)
    assert restored.method == "test"
    assert restored.recommend(1, top_k=1)["id"].item() == 2


def test_category_coherence_detects_relevant_neighbours():
    metrics = evaluate_category_coherence(make_index(), top_k=1, n_queries=4)
    assert metrics["article_type_precision_at_k"] == 1.0
