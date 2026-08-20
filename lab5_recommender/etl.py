"""ETL: загрузка CSV в SQLite. Запуск: python etl.py"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DB_PATH = ROOT / "movies.db"
SCHEMA = ROOT / "schema.sql"


def extract_year(title: str):
    m = re.search(r"\((\d{4})\)\s*$", str(title).strip())
    return int(m.group(1)) if m else None


def clean_title(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title).strip())


def load_csv():
    movies = pd.read_csv(DATA / "movies.csv")
    ratings = pd.read_csv(DATA / "ratings.csv")
    tags = pd.read_csv(DATA / "tags.csv")
    links = pd.read_csv(DATA / "links.csv")

    # Очистка: дубликаты, пропуски.
    movies = movies.drop_duplicates(subset=["movieId"])
    ratings = ratings.drop_duplicates(subset=["userId", "movieId"])
    tags = tags.dropna(subset=["tag"]).drop_duplicates()
    links = links.drop_duplicates(subset=["movieId"])

    movies["year"] = movies["title"].apply(extract_year)
    movies["clean_title"] = movies["title"].apply(clean_title)
    return movies, ratings, tags, links


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA.read_text())

    movies, ratings, tags, links = load_csv()

    # users — уникальные id из ratings и tags.
    users = pd.concat([ratings["userId"], tags["userId"]]).drop_duplicates()
    conn.executemany(
        "INSERT INTO users(user_id) VALUES (?)",
        [(int(u),) for u in users],
    )

    conn.executemany(
        "INSERT INTO movies(movie_id, title, year) VALUES (?, ?, ?)",
        [
            (int(r.movieId), r.clean_title, None if pd.isna(r.year) else int(r.year))
            for r in movies.itertuples()
        ],
    )

    # Жанры в нормальной форме.
    all_genres = set()
    for g in movies["genres"].fillna(""):
        all_genres.update(x for x in g.split("|") if x and x != "(no genres listed)")
    conn.executemany(
        "INSERT INTO genres(name) VALUES (?)",
        [(g,) for g in sorted(all_genres)],
    )
    genre_id = dict(conn.execute("SELECT name, genre_id FROM genres").fetchall())

    movie_genres_rows = []
    for r in movies.itertuples():
        for g in str(r.genres).split("|"):
            if g in genre_id:
                movie_genres_rows.append((int(r.movieId), genre_id[g]))
    conn.executemany(
        "INSERT INTO movie_genres(movie_id, genre_id) VALUES (?, ?)",
        movie_genres_rows,
    )

    conn.executemany(
        "INSERT INTO ratings(user_id, movie_id, rating, timestamp) VALUES (?, ?, ?, ?)",
        ratings[["userId", "movieId", "rating", "timestamp"]].itertuples(index=False, name=None),
    )

    conn.executemany(
        "INSERT INTO tags(user_id, movie_id, tag, timestamp) VALUES (?, ?, ?, ?)",
        tags[["userId", "movieId", "tag", "timestamp"]].itertuples(index=False, name=None),
    )

    conn.executemany(
        "INSERT INTO links(movie_id, imdb_id, tmdb_id) VALUES (?, ?, ?)",
        [
            (int(r.movieId),
             None if pd.isna(r.imdbId) else str(int(r.imdbId)).zfill(7),
             None if pd.isna(r.tmdbId) else str(int(r.tmdbId)))
            for r in links.itertuples()
        ],
    )

    conn.commit()
    print(f"OK. БД: {DB_PATH}")
    for t in ["users", "movies", "genres", "movie_genres", "ratings", "tags", "links"]:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {n}")
    conn.close()


if __name__ == "__main__":
    main()
