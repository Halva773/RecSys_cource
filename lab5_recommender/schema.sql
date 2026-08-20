-- Архитектура БД для рекомендательной системы фильмов.
-- СУБД: SQLite (легко переносится на PostgreSQL/MySQL).

DROP TABLE IF EXISTS ratings;
DROP TABLE IF EXISTS tags;
DROP TABLE IF EXISTS links;
DROP TABLE IF EXISTS movie_genres;
DROP TABLE IF EXISTS genres;
DROP TABLE IF EXISTS movies;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    user_id INTEGER PRIMARY KEY
);

CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    title    TEXT NOT NULL,
    year     INTEGER
);

CREATE TABLE genres (
    genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT NOT NULL UNIQUE
);

-- Связь многие-ко-многим: фильм может иметь несколько жанров.
CREATE TABLE movie_genres (
    movie_id INTEGER NOT NULL,
    genre_id INTEGER NOT NULL,
    PRIMARY KEY (movie_id, genre_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id),
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id)
);

CREATE TABLE ratings (
    user_id   INTEGER NOT NULL,
    movie_id  INTEGER NOT NULL,
    rating    REAL NOT NULL,
    timestamp INTEGER,
    PRIMARY KEY (user_id, movie_id),
    FOREIGN KEY (user_id)  REFERENCES users(user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);

CREATE TABLE tags (
    user_id   INTEGER NOT NULL,
    movie_id  INTEGER NOT NULL,
    tag       TEXT NOT NULL,
    timestamp INTEGER,
    FOREIGN KEY (user_id)  REFERENCES users(user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);

CREATE TABLE links (
    movie_id INTEGER PRIMARY KEY,
    imdb_id  TEXT,
    tmdb_id  TEXT,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);

CREATE INDEX idx_ratings_movie ON ratings(movie_id);
CREATE INDEX idx_ratings_user  ON ratings(user_id);
CREATE INDEX idx_tags_movie    ON tags(movie_id);
