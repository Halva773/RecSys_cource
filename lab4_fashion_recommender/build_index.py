"""Построение, сравнение и сохранение индексов для лабораторной №4.

Пример:
python build_index.py --data-dir data/fashion --sample-size 5000
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.dataset import load_catalog, stratified_sample
from src.evaluation import evaluate_category_coherence
from src.features import make_extractor
from src.index import EmbeddingIndex


METHODS = ("hog_pca", "resnet18", "mobilenet_v3_small")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Построить visual-search индекс Fashion Product Images.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/fashion"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--sample-size", type=int, default=None, help="0 или отсутствие флага — все изображения.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-queries", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu или cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = stratified_sample(load_catalog(args.data_dir), args.sample_size)
    if len(catalog) < 3:
        raise ValueError("Для построения индекса нужно минимум три товара с изображениями.")
    print(f"Каталог: {len(catalog):,} изображений")
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    for method in args.methods:
        print(f"\n[{method}] извлечение признаков...")
        extractor = make_extractor(method, batch_size=args.batch_size, device=args.device)
        embeddings = extractor.transform(catalog["image_path"].tolist())
        index = EmbeddingIndex(catalog=catalog, embeddings=embeddings, method=method)
        metrics = evaluate_category_coherence(index, top_k=args.top_k, n_queries=args.eval_queries)
        path = args.artifacts_dir / f"{method}.npz"
        index.save(path)
        index.write_manifest(args.artifacts_dir / f"{method}.json", metrics)
        rows.append({"method": method, **metrics})
        print(json.dumps(metrics, indent=2))

    results = pd.DataFrame(rows).sort_values("article_type_precision_at_k", ascending=False)
    results.to_csv(args.artifacts_dir / "benchmark.csv", index=False)
    best = results.iloc[0]
    best_method = str(best["method"])
    shutil.copy2(args.artifacts_dir / f"{best_method}.npz", args.artifacts_dir / "best_index.npz")
    manifest = {"best_method": best_method, "selection_metric": "article_type_precision_at_k", "results": rows}
    (args.artifacts_dir / "best_model.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nЛучшая модель: {best_method}. Сжатый индекс: {args.artifacts_dir / 'best_index.npz'}")


if __name__ == "__main__":
    main()
