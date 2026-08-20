"""Рекомендательные системы: content-based, collaborative, hybrid, cold start."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = Path(__file__).parent / "movies.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_movies() -> pd.DataFrame:
    """Фильмы + жанры (через запятую) + теги (через пробел)."""
    conn = get_conn()
    movies = pd.read_sql(
        """
        SELECT m.movie_id, m.title, m.year,
               COALESCE(GROUP_CONCAT(g.name, ' '), '') AS genres
        FROM movies m
        LEFT JOIN movie_genres mg ON mg.movie_id = m.movie_id
        LEFT JOIN genres g        ON g.genre_id  = mg.genre_id
        GROUP BY m.movie_id
        """,
        conn,
    )
    tags = pd.read_sql(
        "SELECT movie_id, GROUP_CONCAT(tag, ' ') AS tags FROM tags GROUP BY movie_id",
        conn,
    )
    conn.close()
    df = movies.merge(tags, on="movie_id", how="left").fillna({"tags": ""})
    df["features"] = (df["genres"] + " " + df["tags"]).str.lower()
    return df


def load_ratings() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", conn)
    conn.close()
    return df


# ---------- Content-Based ----------

class ContentRecommender:
    """TF-IDF по жанрам и тегам + cosine similarity."""

    def __init__(self):
        self.movies = load_movies()
        vec = TfidfVectorizer(token_pattern=r"[a-zA-Z0-9\-]+")
        self.matrix = vec.fit_transform(self.movies["features"])
        self.idx_by_id = {mid: i for i, mid in enumerate(self.movies["movie_id"])}

    def similar(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        if movie_id not in self.idx_by_id:
            return pd.DataFrame()
        i = self.idx_by_id[movie_id]
        sims = cosine_similarity(self.matrix[i], self.matrix).ravel()
        order = sims.argsort()[::-1]
        order = [j for j in order if j != i][:top_n]
        out = self.movies.iloc[order][["movie_id", "title", "year", "genres"]].copy()
        out["score"] = sims[order]
        return out.reset_index(drop=True)

    def for_user(self, ratings: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Профиль пользователя — взвешенная сумма tf-idf векторов оценённых фильмов."""
        if ratings.empty:
            return pd.DataFrame()
        seen = set(ratings["movie_id"])
        rows, weights = [], []
        for _, r in ratings.iterrows():
            if r["movie_id"] in self.idx_by_id:
                rows.append(self.idx_by_id[r["movie_id"]])
                weights.append(r["rating"])
        if not rows:
            return pd.DataFrame()
        weights = np.array(weights)
        # Если оценки разные — центрируем (учёт «нелюбимых» фильмов).
        # Если все одинаковые (или одна оценка) — берём просто как веса.
        if len(weights) > 1 and weights.std() > 0:
            weights = weights - weights.mean()
        profile = (self.matrix[rows].T @ weights)
        if hasattr(profile, "toarray"):
            profile = np.asarray(profile.todense()).ravel()
        else:
            profile = np.asarray(profile).ravel()
        sims = cosine_similarity(profile.reshape(1, -1), self.matrix).ravel()
        order = sims.argsort()[::-1]
        out = []
        for j in order:
            mid = int(self.movies.iloc[j]["movie_id"])
            if mid in seen:
                continue
            out.append(j)
            if len(out) >= top_n:
                break
        res = self.movies.iloc[out][["movie_id", "title", "year", "genres"]].copy()
        res["score"] = sims[out]
        return res.reset_index(drop=True)


# ---------- Collaborative (item-item) ----------

class CollaborativeRecommender:
    """Item-based KNN на матрице рейтингов с центрированием по фильму."""

    def __init__(self, min_ratings: int = 5, ratings_df: pd.DataFrame | None = None):
        ratings = load_ratings() if ratings_df is None else ratings_df.copy()
        counts = ratings.groupby("movie_id").size()
        keep = counts[counts >= min_ratings].index
        ratings = ratings[ratings["movie_id"].isin(keep)]
        self.pivot = ratings.pivot_table(
            index="movie_id", columns="user_id", values="rating"
        )
        # Центрируем по фильму — убираем bias популярности.
        centered = self.pivot.sub(self.pivot.mean(axis=1), axis=0).fillna(0)
        self.matrix = centered.values
        self.movie_ids = self.pivot.index.to_numpy()
        self.idx_by_id = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.sims = cosine_similarity(self.matrix)
        np.fill_diagonal(self.sims, 0)

        meta = load_movies()[["movie_id", "title", "year", "genres"]]
        self.meta = meta.set_index("movie_id")

    def similar(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        if movie_id not in self.idx_by_id:
            return pd.DataFrame()
        i = self.idx_by_id[movie_id]
        order = self.sims[i].argsort()[::-1][:top_n]
        ids = self.movie_ids[order]
        out = self.meta.loc[ids].reset_index()
        out["score"] = self.sims[i][order]
        return out

    def for_user(self, ratings: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Скор кандидата = сумма sim * (rating − 2.5) по оцененным фильмам.
        Сдвиг на 2.5 даёт штраф за низкие оценки и не вырождается при одинаковых rating."""
        if ratings.empty:
            return pd.DataFrame()
        seen = set(ratings["movie_id"])
        pred = np.zeros(len(self.movie_ids))
        for _, r in ratings.iterrows():
            if r["movie_id"] not in self.idx_by_id:
                continue
            i = self.idx_by_id[r["movie_id"]]
            pred += self.sims[i] * (r["rating"] - 2.5)
        order = pred.argsort()[::-1]
        out_ids = []
        for j in order:
            mid = int(self.movie_ids[j])
            if mid in seen:
                continue
            out_ids.append((mid, pred[j]))
            if len(out_ids) >= top_n:
                break
        if not out_ids:
            return pd.DataFrame()
        df = self.meta.loc[[m for m, _ in out_ids]].reset_index()
        df["score"] = [s for _, s in out_ids]
        return df


# ---------- Hybrid ----------

class HybridRecommender:
    """Линейная комбинация content-based и collaborative скоров."""

    def __init__(self, content: ContentRecommender, collab: CollaborativeRecommender):
        self.content = content
        self.collab = collab

    @staticmethod
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        s = df["score"]
        rng = s.max() - s.min()
        df = df.copy()
        df["score"] = (s - s.min()) / rng if rng > 0 else 0.0
        return df

    def for_user(self, ratings: pd.DataFrame, top_n: int = 10, alpha: float = 0.5) -> pd.DataFrame:
        c = self._norm(self.content.for_user(ratings, top_n=200))
        k = self._norm(self.collab.for_user(ratings, top_n=200))
        if c.empty and k.empty:
            return pd.DataFrame()
        if c.empty:
            return k.head(top_n)
        if k.empty:
            return c.head(top_n)
        merged = c.merge(k, on=["movie_id", "title", "year", "genres"], how="outer",
                         suffixes=("_c", "_k")).fillna(0)
        merged["score"] = alpha * merged["score_c"] + (1 - alpha) * merged["score_k"]
        return (
            merged.sort_values("score", ascending=False)
            [["movie_id", "title", "year", "genres", "score"]]
            .head(top_n).reset_index(drop=True)
        )


# ---------- Cold start ----------

def popular_movies(top_n: int = 10, min_ratings: int = 50) -> pd.DataFrame:
    """Bayesian-усреднение: даёт хорошие фильмы с надёжной выборкой."""
    conn = get_conn()
    df = pd.read_sql(
        """
        SELECT m.movie_id, m.title, m.year,
               COALESCE(GROUP_CONCAT(DISTINCT g.name), '') AS genres,
               AVG(r.rating) AS avg_rating, COUNT(r.rating) AS n_ratings
        FROM movies m
        LEFT JOIN movie_genres mg ON mg.movie_id = m.movie_id
        LEFT JOIN genres g        ON g.genre_id  = mg.genre_id
        JOIN ratings r            ON r.movie_id  = m.movie_id
        GROUP BY m.movie_id
        HAVING n_ratings >= ?
        """,
        conn, params=(min_ratings,),
    )
    conn.close()
    C = df["avg_rating"].mean()
    m = min_ratings
    df["score"] = (df["n_ratings"] * df["avg_rating"] + m * C) / (df["n_ratings"] + m)
    return df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def cold_start_by_genres(genres: list[str], top_n: int = 10, min_ratings: int = 30) -> pd.DataFrame:
    """Холодный старт по выбранным жанрам: популярные фильмы с пересечением жанров."""
    if not genres:
        return popular_movies(top_n)
    conn = get_conn()
    placeholders = ",".join("?" * len(genres))
    df = pd.read_sql(
        f"""
        WITH movie_stats AS (
            SELECT movie_id, AVG(rating) AS avg_rating, COUNT(*) AS n_ratings
            FROM ratings GROUP BY movie_id
        ),
        movie_match AS (
            SELECT mg.movie_id,
                   SUM(CASE WHEN g.name IN ({placeholders}) THEN 1 ELSE 0 END) AS match_count
            FROM movie_genres mg JOIN genres g ON g.genre_id = mg.genre_id
            GROUP BY mg.movie_id
        )
        SELECT m.movie_id, m.title, m.year,
               (SELECT GROUP_CONCAT(g2.name)
                  FROM movie_genres mg2 JOIN genres g2 ON g2.genre_id = mg2.genre_id
                 WHERE mg2.movie_id = m.movie_id) AS genres,
               s.avg_rating, s.n_ratings, mm.match_count
        FROM movies m
        JOIN movie_stats s  ON s.movie_id  = m.movie_id
        JOIN movie_match mm ON mm.movie_id = m.movie_id
        WHERE mm.match_count > 0 AND s.n_ratings >= ?
        """,
        conn, params=(*genres, min_ratings),
    )
    conn.close()
    if df.empty:
        return pd.DataFrame()
    C = df["avg_rating"].mean()
    m = min_ratings
    bayes = (df["n_ratings"] * df["avg_rating"] + m * C) / (df["n_ratings"] + m)
    df["score"] = bayes * (df["match_count"] / len(genres))
    return df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def list_genres() -> list[str]:
    conn = get_conn()
    rows = conn.execute("SELECT name FROM genres ORDER BY name").fetchall()
    conn.close()
    return [r[0] for r in rows]


# ---------- Оценка моделей (leave-last-K-out) ----------

def evaluate_models(top_n: int = 10, n_users: int = 100, k: int = 5,
                    min_user_ratings: int = 30, alpha: float = 0.5,
                    random_state: int = 42) -> dict:
    """
    Leave-last-K-out: для каждого тестового пользователя сортируем его оценки
    по timestamp и прячем последние K в тест, остальные считаем train.
    Collaborative модель ПЕРЕОБУЧАЕТСЯ на train (без test-оценок) — без leakage.
    Релевант: rating >= 4. Метрики: Precision@N, Recall@N, F1@N.
    """
    rng = np.random.default_rng(random_state)
    conn = get_conn()
    all_ratings = pd.read_sql(
        "SELECT user_id, movie_id, rating, timestamp FROM ratings", conn
    )
    conn.close()

    counts = all_ratings.groupby("user_id").size()
    eligible = counts[counts >= min_user_ratings].index.to_numpy()
    if len(eligible) == 0:
        return {}
    sample = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)

    # Сплит: последние k оценок каждого sample-пользователя — в test.
    sample_ratings = all_ratings[all_ratings["user_id"].isin(sample)]
    sample_ratings = sample_ratings.sort_values(["user_id", "timestamp"])
    test_df = sample_ratings.groupby("user_id").tail(k)
    train_df = all_ratings.merge(
        test_df[["user_id", "movie_id"]].assign(_test=1),
        on=["user_id", "movie_id"], how="left",
    )
    train_df = train_df[train_df["_test"].isna()].drop(columns="_test")

    # Content от ratings не зависит, collaborative — переобучаем на train.
    content = ContentRecommender()
    collab = CollaborativeRecommender(
        ratings_df=train_df[["user_id", "movie_id", "rating"]]
    )
    hybrid = HybridRecommender(content, collab)

    metrics = {"content": [], "collaborative": [], "hybrid": []}
    for uid in sample:
        seed = train_df[train_df["user_id"] == uid][["movie_id", "rating"]]
        test_user = test_df[test_df["user_id"] == uid]
        relevant = set(test_user[test_user["rating"] >= 4.0]["movie_id"])
        if seed.empty or not relevant:
            continue

        recs = {
            "content":       content.for_user(seed, top_n=top_n),
            "collaborative": collab.for_user(seed, top_n=top_n),
            "hybrid":        hybrid.for_user(seed, top_n=top_n, alpha=alpha),
        }
        for name, df in recs.items():
            ids = set(df["movie_id"]) if not df.empty else set()
            tp = len(ids & relevant)
            p = tp / len(ids) if ids else 0
            r = tp / len(relevant)
            f1 = (2 * p * r / (p + r)) if (p + r) else 0
            metrics[name].append((p, r, f1))

    out = {}
    for name, vals in metrics.items():
        if not vals:
            out[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
            continue
        arr = np.array(vals)
        out[name] = {
            "precision": float(arr[:, 0].mean()),
            "recall":    float(arr[:, 1].mean()),
            "f1":        float(arr[:, 2].mean()),
            "n":         len(vals),
        }
    return out
