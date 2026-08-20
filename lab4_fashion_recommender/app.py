"""Streamlit-интерфейс визуального поиска похожих fashion-товаров."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.index import EmbeddingIndex


ROOT = Path(__file__).parent
ARTIFACT_PATH = ROOT / "artifacts" / "best_index.npz"
MANIFEST_PATH = ROOT / "artifacts" / "best_model.json"

st.set_page_config(page_title="Fashion Visual Search", page_icon="👗", layout="wide")


@st.cache_resource
def load_index() -> EmbeddingIndex:
    return EmbeddingIndex.load(ARTIFACT_PATH)


def image_caption(row: pd.Series) -> str:
    return f"{row['productDisplayName']} · {row['articleType']} · similarity: {row.get('similarity', 0):.3f}"


if not ARTIFACT_PATH.exists():
    st.title("Fashion Visual Search")
    st.warning("Индекс ещё не создан.")
    st.code("python eda.py --data-dir data/fashion\npython build_index.py --data-dir data/fashion --sample-size 5000\nstreamlit run app.py")
    st.info("Скачайте Fashion Product Images (Small) из Kaggle и распакуйте в data/fashion/.")
    st.stop()

index = load_index()
catalog = index.catalog
st.title("Fashion Visual Search")
st.caption(f"Модель: {index.method}. Поиск по косинусной близости image embeddings.")

with st.sidebar:
    st.header("Фильтры каталога")
    filters: dict[str, str] = {}
    for column, label in (("gender", "Пол"), ("masterCategory", "Категория"), ("season", "Сезон"), ("usage", "Назначение")):
        values = sorted(catalog[column].dropna().unique().tolist())
        filters[column] = st.selectbox(label, ["Все", *values])
    top_k = st.slider("Количество рекомендаций", min_value=4, max_value=20, value=8)

filtered = catalog.copy()
for column, value in filters.items():
    if value != "Все":
        filtered = filtered[filtered[column].eq(value)]

if filtered.empty:
    st.warning("По выбранным фильтрам товаров нет. Ослабьте условия.")
    st.stop()

options = {f"{row.productDisplayName} — id {row.id}": int(row.id) for row in filtered.itertuples(index=False)}
selection = st.selectbox("Выберите товар-образец", options.keys())
item_id = options[selection]
query = catalog[catalog["id"].eq(item_id)].iloc[0]

left, right = st.columns((1, 3))
with left:
    st.subheader("Образец")
    st.image(query["image_path"], caption=query["productDisplayName"], use_container_width=True)
    st.write(f"**{query['gender']} · {query['season']}**")
    st.caption(f"{query['masterCategory']} / {query['subCategory']} / {query['articleType']}")
with right:
    st.subheader("Похожие товары")
    recommendations = index.recommend(item_id, top_k=top_k, filters=filters)
    if recommendations.empty:
        st.info("С такими фильтрами не найдено других товаров. Ослабьте условия.")
    else:
        columns = st.columns(4)
        for position, (_, row) in enumerate(recommendations.iterrows()):
            with columns[position % 4]:
                st.image(row["image_path"], caption=image_caption(row), use_container_width=True)
                st.caption(f"{row['gender']} · {row['season']} · {row['baseColour']}")

if MANIFEST_PATH.exists():
    with st.expander("Как выбрана модель"):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        st.write(f"Лучшая модель: **{manifest['best_method']}** по `articleType Precision@K`.")
        st.dataframe(pd.DataFrame(manifest["results"]), use_container_width=True, hide_index=True)
