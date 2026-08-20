"""Воспроизводимый EDA Fashion Product Images Dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.dataset import load_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сформировать EDA-графики и краткое резюме.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/fashion"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    return parser.parse_args()


def save_countplot(catalog: pd.DataFrame, column: str, output_path: Path, top_n: int = 15) -> None:
    counts = catalog[column].value_counts().head(top_n).sort_values()
    plt.figure(figsize=(10, max(4, 0.35 * len(counts))))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, legend=False, palette="viridis")
    plt.title(f"Распределение: {column}")
    plt.xlabel("Количество товаров")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    catalog = load_catalog(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for column in ("masterCategory", "gender", "season", "articleType", "baseColour"):
        save_countplot(catalog, column, args.output_dir / f"{column}.png")

    summary = {
        "products_with_images": int(len(catalog)),
        "master_categories": int(catalog["masterCategory"].nunique()),
        "article_types": int(catalog["articleType"].nunique()),
        "genders": catalog["gender"].value_counts().to_dict(),
        "top_article_types": catalog["articleType"].value_counts().head(10).to_dict(),
    }
    summary_path = args.output_dir.parent / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"EDA готов: {summary_path}")


if __name__ == "__main__":
    main()
