"""Сжатый индекс эмбеддингов и поиск по косинусной близости."""
from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import l2_normalize


@dataclass
class EmbeddingIndex:
    catalog: pd.DataFrame
    embeddings: np.ndarray
    method: str

    def __post_init__(self) -> None:
        if len(self.catalog) != len(self.embeddings):
            raise ValueError("Количество строк каталога и эмбеддингов должно совпадать.")
        self.catalog = self.catalog.reset_index(drop=True).copy()
        self.embeddings = l2_normalize(np.asarray(self.embeddings, dtype="float32"))
        self._positions = {int(item_id): position for position, item_id in enumerate(self.catalog["id"])}

    def recommend(self, item_id: int, top_k: int = 8, filters: dict[str, str] | None = None) -> pd.DataFrame:
        """Возвращает похожие товары, исключая сам запрос и учитывая UI-фильтры."""
        if item_id not in self._positions:
            raise KeyError(f"Товар id={item_id} отсутствует в индексе.")
        query_position = self._positions[item_id]
        scores = self.embeddings @ self.embeddings[query_position]
        allowed = np.ones(len(self.catalog), dtype=bool)
        for column, value in (filters or {}).items():
            if value and value != "Все" and column in self.catalog:
                allowed &= self.catalog[column].eq(value).to_numpy()
        allowed[query_position] = False
        scores[~allowed] = -np.inf
        positions = np.argsort(scores)[::-1][:top_k]
        positions = positions[np.isfinite(scores[positions])]
        result = self.catalog.iloc[positions].copy()
        result["similarity"] = scores[positions]
        return result.reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        """Сохраняет компактно: float32-эмбеддинги сжимаются внутри NPZ."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self.catalog.to_json(orient="table", force_ascii=False)
        np.savez_compressed(path, embeddings=self.embeddings, metadata=np.array(metadata), method=np.array(self.method))

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingIndex":
        with np.load(path, allow_pickle=False) as payload:
            catalog = pd.read_json(StringIO(str(payload["metadata"].item())), orient="table")
            return cls(catalog=catalog, embeddings=payload["embeddings"], method=str(payload["method"].item()))

    def write_manifest(self, path: str | Path, metrics: dict[str, float]) -> None:
        Path(path).write_text(
            json.dumps({"method": self.method, "items": len(self.catalog), "metrics": metrics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
