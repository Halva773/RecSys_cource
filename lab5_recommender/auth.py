"""Регистрация/авторизация. Аккаунты хранятся в той же SQLite, что и MovieLens.

Минимально: одна таблица `accounts` + переиспользование таблицы `ratings`.
Каждому аккаунту выдаётся свой user_id из пространства MovieLens, чтобы
foreign-keys и существующие модели работали без изменений.
"""
import hashlib
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "movies.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def _hash(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _next_ml_user_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(user_id), 0) FROM users").fetchone()
    return int(row[0]) + 1


def _create_account(conn: sqlite3.Connection, username: str, password: str,
                    is_admin: bool = False) -> int:
    salt = os.urandom(8).hex()
    ml_user_id = _next_ml_user_id(conn)
    conn.execute("INSERT INTO users(user_id) VALUES (?)", (ml_user_id,))
    conn.execute(
        "INSERT INTO accounts(username, password_hash, salt, is_admin, ml_user_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (username, _hash(password, salt), salt, int(is_admin), ml_user_id),
    )
    conn.commit()
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def init_db() -> None:
    conn = get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                account_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt          TEXT NOT NULL,
                is_admin      INTEGER NOT NULL DEFAULT 0,
                ml_user_id    INTEGER NOT NULL UNIQUE
            )
            """
        )
        conn.commit()
        has_admin = conn.execute(
            "SELECT COUNT(*) FROM accounts WHERE is_admin = 1"
        ).fetchone()[0]
        if not has_admin:
            _create_account(conn, "admin", "admin", is_admin=True)
    finally:
        conn.close()


def register(username: str, password: str) -> tuple[bool, str]:
    username = (username or "").strip()
    if not username or not password:
        return False, "Введите имя пользователя и пароль."
    if len(password) < 3:
        return False, "Пароль слишком короткий."
    conn = get_conn()
    try:
        if conn.execute("SELECT 1 FROM accounts WHERE username = ?", (username,)).fetchone():
            return False, "Имя пользователя уже занято."
        _create_account(conn, username, password, is_admin=False)
        return True, "Регистрация успешна. Войдите в систему."
    finally:
        conn.close()


def login(username: str, password: str) -> dict | None:
    username = (username or "").strip()
    if not username or not password:
        return None
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT account_id, password_hash, salt, is_admin, ml_user_id "
            "FROM accounts WHERE username = ?",
            (username,),
        ).fetchone()
        if not row:
            return None
        account_id, ph, salt, is_admin, ml_user_id = row
        if _hash(password, salt) != ph:
            return None
        return {
            "account_id": int(account_id),
            "username": username,
            "is_admin": bool(is_admin),
            "ml_user_id": int(ml_user_id),
        }
    finally:
        conn.close()


def get_user_ratings(ml_user_id: int) -> pd.DataFrame:
    conn = get_conn()
    try:
        return pd.read_sql(
            """
            SELECT r.movie_id, m.title, m.year, r.rating
            FROM ratings r JOIN movies m ON m.movie_id = r.movie_id
            WHERE r.user_id = ?
            ORDER BY r.timestamp DESC
            """,
            conn, params=(ml_user_id,),
        )
    finally:
        conn.close()


def save_rating(ml_user_id: int, movie_id: int, rating: float) -> None:
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO ratings(user_id, movie_id, rating, timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, movie_id) DO UPDATE SET
                rating = excluded.rating,
                timestamp = excluded.timestamp
            """,
            (int(ml_user_id), int(movie_id), float(rating), int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()
