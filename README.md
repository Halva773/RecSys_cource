# Рекомендательные системы

Учебный репозиторий с последовательными работами по рекомендательным системам: от исследования данных и базовых подходов до полноценного веб-приложения с несколькими алгоритмами рекомендаций.

## Содержание

| Работа | Формат | О чём |
| --- | --- | --- |
| 1 | [`01_simple_system.ipynb`](01_simple_system.ipynb) | EDA кино-датасета, популярностные, content-based, Word2Vec и SVD-подходы. |
| 2 | [`lab2.ipynb`](lab2.ipynb) | Сравнение collaborative filtering: KNN, SVD и SVD++. |
| 3–4 | — | Будут добавлены отдельным модулем. |
| 5–6 | [`lab5_recommender/`](lab5_recommender/) | Приложение Streamlit: ETL, SQLite, content-based, item-based collaborative, hybrid и cold start. |

## Быстрый старт

### Работы 1–2

Тетрадки используют файлы из датасета **The Movies Dataset**. Скачайте его с [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) и поместите нужные CSV в корневую папку `data/` (она намеренно не отслеживается в Git). Минимально потребуются:

- для работы 1: `movies_metadata.csv`, `ratings.csv`, `ratings_small.csv`, `keywords.csv`, `credits.csv`, `links_small.csv`;
- для работы 2: `ratings_small.csv`, `links_small.csv`, `movies_metadata.csv`.

После этого откройте нужный `.ipynb` в Jupyter Notebook или VS Code. Зависимости: Python 3.10+, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `gensim`, `wordcloud`, `scipy`, `scikit-surprise`.

### Работы 5–6

Инструкции по запуску, устройству приложения и структуре данных находятся в [README проекта](lab5_recommender/README.md).

## Репозиторий и данные

В репозитории сохраняются исходный код, тетрадки, отчёты и компактные CSV для работ 5–6. Локальные окружения, кэши, скачиваемые данные работ 1–2 и SQLite-базы исключены через `.gitignore`: они либо пересоздаются, либо содержат пользовательское состояние.
