"""Загрузка и валидация Fashion Product Images Dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "usage",
    "productDisplayName",
}


def load_catalog(data_dir: str | Path) -> pd.DataFrame:
    """Возвращает только товары с существующей картинкой.

    Ожидаемая структура соответствует Kaggle-датасету:
    ``data_dir/styles.csv`` и ``data_dir/images/<id>.jpg``.
    """
    data_dir = Path(data_dir)
    styles_path = data_dir / "styles.csv"
    images_dir = data_dir / "images"
    if not styles_path.exists() or not images_dir.exists():
        raise FileNotFoundError(
            "Не найден Fashion Product Images Dataset. Ожидаются "
            f"{styles_path} и {images_dir}. См. README проекта."
        )

    catalog = pd.read_csv(styles_path, on_bad_lines="skip")
    missing = REQUIRED_COLUMNS.difference(catalog.columns)
    if missing:
        raise ValueError(f"В styles.csv нет обязательных колонок: {sorted(missing)}")

    catalog = catalog.copy()
    catalog["id"] = pd.to_numeric(catalog["id"], errors="coerce")
    catalog = catalog.dropna(subset=["id"]).copy()
    catalog["id"] = catalog["id"].astype("int64")
    catalog["image_path"] = catalog["id"].map(lambda item_id: str(images_dir / f"{item_id}.jpg"))
    catalog = catalog[catalog["image_path"].map(lambda path: Path(path).exists())].copy()

    for column in REQUIRED_COLUMNS - {"id"}:
        catalog[column] = catalog[column].fillna("Unknown").astype(str).str.strip()
    return catalog.reset_index(drop=True)


def stratified_sample(catalog: pd.DataFrame, max_items: int | None, random_state: int = 42) -> pd.DataFrame:
    """Ограничивает каталог, сохраняя представленность крупных категорий."""
    if max_items is None or max_items <= 0 or len(catalog) <= max_items:
        return catalog.reset_index(drop=True)

    fractions = catalog["masterCategory"].value_counts(normalize=True)
    chunks: list[pd.DataFrame] = []
    remaining = max_items
    for category, fraction in fractions.items():
        group = catalog[catalog["masterCategory"] == category]
        n = min(len(group), max(1, round(max_items * fraction)))
        chunks.append(group.sample(n=n, random_state=random_state))
        remaining -= n

    result = pd.concat(chunks, ignore_index=True)
    if remaining > 0:
        unused = catalog[~catalog["id"].isin(result["id"])]
        result = pd.concat(
            [result, unused.sample(n=min(remaining, len(unused)), random_state=random_state)],
            ignore_index=True,
        )
    return result.sample(frac=1, random_state=random_state).head(max_items).reset_index(drop=True)
