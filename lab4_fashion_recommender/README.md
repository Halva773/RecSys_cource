# Fashion Visual Search — лабораторная работа №4

Рекомендательная система одежды по визуальной похожести. Она строит эмбеддинги изображений товаров, сравнивает три подхода и показывает похожие позиции в Streamlit-интерфейсе с фильтрами по полу, категории, сезону и назначению.

## Что реализовано

| Подход | Признаки | Роль в сравнении |
| --- | --- | --- |
| `hog_pca` | HOG-дескрипторы → PCA → cosine KNN | Классический CV/ML baseline. |
| `resnet18` | Embeddings предобученной ResNet-18 | Transfer learning с сильными общими visual features. |
| `mobilenet_v3_small` | Embeddings MobileNetV3 Small | Более компактная CNN для быстрого inference. |

Модели оцениваются одинаково: для случайных товаров считается `Precision@K` по совпадению `articleType` и `subCategory`. В датасете нет кликов или пользовательских оценок, поэтому совпадение известной товарной категории — прозрачный proxy-сигнал релевантности. Лучшая модель выбирается по `article_type_precision_at_k`.

## Датасет

Используется [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small): около 44 тысяч изображений и `styles.csv` с полом, категорией, сезоном, цветом и названием товара. Исходные изображения не добавляются в Git; см. [инструкцию по данным](data/README.md).

## Запуск

```powershell
cd lab4_fashion_recommender
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Распаковать Kaggle-датасет в data/fashion/, затем:
python eda.py --data-dir data/fashion
python build_index.py --data-dir data/fashion --sample-size 5000
streamlit run app.py
```

`--sample-size 5000` — быстрый воспроизводимый прогон для ноутбука без GPU. Чтобы обработать весь датасет, не указывайте этот флаг. При первом запуске PyTorch автоматически загрузит веса предобученных CNN.

## Результаты и артефакты

Команда EDA создаёт графики по категориям, полу, сезону, типам товаров и цветам в `reports/figures/` и сводку `reports/eda_summary.json`.

`build_index.py` сохраняет для каждого метода сжатый `NPZ`-индекс эмбеддингов, его метрики и таблицу `benchmark.csv`. Победитель дополнительно записывается как `artifacts/best_index.npz` вместе с `artifacts/best_model.json`. Такой формат хранит только `float32`-векторы и метаданные — быстро загружается в UI и не требует сериализовать нейросеть.

## Структура

```text
lab4_fashion_recommender/
├── src/                # загрузка каталога, модели, индекс и offline-оценка
├── eda.py               # воспроизводимый EDA
├── build_index.py       # обучение/сравнение/сохранение индексов
├── app.py               # Streamlit UI
├── tests/               # unit-тесты индекса и метрик
├── data/README.md       # загрузка официального датасета
└── REPORT.md            # методология для отчёта и защиты
```
