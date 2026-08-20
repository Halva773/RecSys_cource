"""Streamlit-интерфейс рекомендательной системы.
Запуск: streamlit run app.py
"""
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

import auth
from recommender import (
    ContentRecommender,
    CollaborativeRecommender,
    HybridRecommender,
    cold_start_by_genres,
    evaluate_models,
    list_genres,
    popular_movies,
)

DB_PATH = Path(__file__).parent / "movies.db"

st.set_page_config(page_title="Movie Recommender", layout="wide")
auth.init_db()

if "user" not in st.session_state:
    st.session_state.user = None

with st.sidebar:
    st.header("Аккаунт")
    if st.session_state.user:
        u = st.session_state.user
        role = "админ" if u["is_admin"] else "пользователь"
        st.write(f"**{u['username']}** ({role})")
        if st.button("Выйти"):
            st.session_state.user = None
            st.rerun()
    else:
        tab_login, tab_register = st.tabs(["Войти", "Регистрация"])
        with tab_login:
            li_user = st.text_input("Имя", key="li_user")
            li_pass = st.text_input("Пароль", type="password", key="li_pass")
            if st.button("Войти", key="li_btn"):
                user = auth.login(li_user, li_pass)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Неверный логин или пароль.")
        with tab_register:
            re_user = st.text_input("Имя", key="re_user")
            re_pass = st.text_input("Пароль", type="password", key="re_pass")
            if st.button("Зарегистрироваться", key="re_btn"):
                ok, msg = auth.register(re_user, re_pass)
                (st.success if ok else st.error)(msg)

if not st.session_state.user:
    st.title("Рекомендательная система фильмов")
    st.info(
        "Войдите или зарегистрируйтесь в боковой панели, чтобы получить персональные "
        "рекомендации. Доступ к админке — `admin` / `admin`."
    )
    st.stop()

if st.session_state.user is None:
    st.error("Ошибка: не удалось инициализировать пользователя. Используйте: streamlit run app.py")
    st.stop()


@st.cache_resource
def load_models():
    content = ContentRecommender()
    collab = CollaborativeRecommender()
    hybrid = HybridRecommender(content, collab)
    return content, collab, hybrid


@st.cache_data
def all_movies() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT m.movie_id, m.title, m.year,
               COALESCE((SELECT GROUP_CONCAT(g.name, ', ')
                         FROM movie_genres mg JOIN genres g ON g.genre_id = mg.genre_id
                         WHERE mg.movie_id = m.movie_id), '') AS genres
        FROM movies m ORDER BY m.title
        """,
        conn,
    )
    conn.close()
    return df


@st.cache_data
def all_user_ids() -> list[int]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT user_id FROM users ORDER BY user_id").fetchall()
    conn.close()
    return [r[0] for r in rows]


@st.cache_data
def user_history(user_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT r.movie_id, m.title, r.rating
        FROM ratings r JOIN movies m ON m.movie_id = r.movie_id
        WHERE r.user_id = ? ORDER BY r.rating DESC, r.timestamp DESC
        """,
        conn, params=(user_id,),
    )
    conn.close()
    return df


def show_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("Нет рекомендаций — попробуйте изменить параметры.")
        return
    cols = [c for c in ["title", "year", "genres", "avg_rating", "n_ratings", "score"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)


content, collab, hybrid = load_models()
movies_df = all_movies()
title_to_id = dict(zip(movies_df["title"] + " (" + movies_df["year"].astype("Int64").astype(str) + ")", movies_df["movie_id"]))


def _cold_start(ml_uid: int):
    st.title("Холодный старт")
    st.caption("Выберите любимые жанры и оцените несколько фильмов — далее система начнёт давать персональные рекомендации.")

    st.subheader("1. Любимые жанры")
    genres = st.multiselect(
        "Жанры", list_genres(),
        default=st.session_state.get("user_genres", ["Comedy", "Adventure"]),
    )
    st.session_state.user_genres = genres
    n_seed = st.slider("Сколько фильмов предложить для оценки", 5, 25, 10)

    st.subheader("2. Оцените фильмы (0 — не оценивать / не смотрел)")
    if not genres:
        st.info("Выберите хотя бы один жанр.")
        return
    suggested = cold_start_by_genres(genres, top_n=n_seed, min_ratings=30)
    if suggested.empty:
        st.warning("По выбранным жанрам ничего не нашлось.")
        return
    new_ratings: dict[int, float] = {}
    cols = st.columns(2)
    for i, row in suggested.reset_index(drop=True).iterrows():
        mid = int(row["movie_id"])
        year = int(row["year"]) if pd.notna(row["year"]) else None
        label = f"{row['title']}" + (f" ({year})" if year else "")
        if "genres" in row and pd.notna(row["genres"]):
            label += f" — _{row['genres']}_"
        with cols[i % 2]:
            rating = st.slider(label, 0.0, 5.0, 0.0, 0.5,
                               key=f"rate_{mid}", help="0 = пропустить")
            if rating > 0:
                new_ratings[mid] = rating
    if st.button("Сохранить оценки"):
        if not new_ratings:
            st.warning("Поставьте хотя бы одну оценку.")
            return
        for mid, r in new_ratings.items():
            auth.save_rating(ml_uid, mid, r)
        st.success(f"Сохранено оценок: {len(new_ratings)}.")
        st.rerun()


def _personal_feed(ml_uid: int, hist: pd.DataFrame):
    st.subheader("Ваши персональные рекомендации")
    st.caption("Кликните по строке, чтобы оценить фильм.")
    seed = hist[["movie_id", "rating"]].drop_duplicates(subset="movie_id")
    recs = hybrid.for_user(seed, top_n=15, alpha=0.5)
    if recs is None or recs.empty:
        st.info("Нет рекомендаций — оцените ещё несколько фильмов.")
    else:
        cols_show = [c for c in ["title", "year", "genres", "score"] if c in recs.columns]
        event = st.dataframe(
            recs[cols_show],
            use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
            key="recs_table",
        )
        sel = event.selection.rows if event and event.selection else []
        if sel:
            row = recs.iloc[sel[0]]
            mid = int(row["movie_id"])
            st.markdown(f"**Оценить:** {row['title']}")
            rating = st.slider("Оценка", 0.5, 5.0, 4.0, 0.5, key=f"rate_rec_{mid}")
            if st.button("Сохранить оценку", key=f"save_rec_{mid}"):
                auth.save_rating(ml_uid, mid, rating)
                st.success(f"Сохранено: {row['title']} → {rating}")
                st.rerun()

    with st.expander(f"Мои оценки ({len(hist)})", expanded=False):
        st.dataframe(hist, use_container_width=True, hide_index=True)

    st.subheader("Найти и оценить любой фильм")
    rated = set(hist["movie_id"])
    avail = [t for t, mid in title_to_id.items() if int(mid) not in rated]
    if not avail:
        st.info("Вы оценили все фильмы из базы.")
        return
    pick = st.selectbox("Фильм", avail, key="user_new_pick")
    rating = st.slider("Оценка", 0.5, 5.0, 4.0, 0.5, key="user_new_r")
    if st.button("Сохранить", key="user_new_btn"):
        auth.save_rating(ml_uid, int(title_to_id[pick]), rating)
        st.success(f"Сохранено: {pick} → {rating}")
        st.rerun()


def user_mode(u: dict):
    ml_uid = u["ml_user_id"]
    hist = auth.get_user_ratings(ml_uid)
    if hist.empty:
        _cold_start(ml_uid)
    else:
        st.title(f"Привет, {u['username']}!")
        _personal_feed(ml_uid, hist)


if st.session_state.user is None or not st.session_state.user.get("is_admin", False):
    if st.session_state.user:
        user_mode(st.session_state.user)
    st.stop()

st.title("Рекомендательная система фильмов — админка")
st.caption("Лабораторная работа №5. MovieLens dataset → SQLite → Streamlit.")

tab_content, tab_collab, tab_hybrid, tab_cold, tab_eval, tab_db = st.tabs([
    "Content-based",
    "Collaborative",
    "Гибридная",
    "Холодный старт",
    "Оценка",
    "БД",
])

with tab_content:
    st.header("Похожие фильмы (по жанрам и тегам)")
    pick = st.selectbox("Выберите фильм", list(title_to_id.keys()), key="cb_pick")
    top_n = st.slider("Сколько показать", 5, 30, 10, key="cb_n")
    if pick:
        show_table(content.similar(int(title_to_id[pick]), top_n=top_n))

with tab_collab:
    st.header("Похожие фильмы по поведению пользователей")
    pick = st.selectbox("Выберите фильм", list(title_to_id.keys()), key="kb_pick")
    top_n = st.slider("Сколько показать", 5, 30, 10, key="kb_n")
    if pick:
        show_table(collab.similar(int(title_to_id[pick]), top_n=top_n))

    st.divider()
    st.subheader("Рекомендации существующему пользователю")
    user_id = st.selectbox("user_id", all_user_ids(), key="kb_user")
    if user_id:
        hist = user_history(int(user_id))
        st.write(f"История пользователя ({len(hist)} оценок). Топ-10 любимых:")
        st.dataframe(hist.head(10), use_container_width=True, hide_index=True)
        recs = collab.for_user(hist[["movie_id", "rating"]], top_n=10)
        st.write("Рекомендации:")
        show_table(recs)

with tab_hybrid:
    st.header("Гибридные рекомендации")
    st.write(
        "Оцените несколько фильмов от 0.5 до 5.0 — система объединит "
        "content-based и collaborative-подходы."
    )
    alpha = st.slider("Вес content-based (1 - вес collaborative)", 0.0, 1.0, 0.5, 0.05)
    n = st.slider("Сколько фильмов оценить", 2, 10, 3, key="hyb_n_seed")

    if "seed_picks" not in st.session_state:
        st.session_state.seed_picks = []

    seed_rows = []
    cols = st.columns(2)
    for i in range(n):
        with cols[i % 2]:
            t = st.selectbox(f"Фильм {i+1}", list(title_to_id.keys()), key=f"hyb_t_{i}")
            r = st.slider(f"Оценка {i+1}", 0.5, 5.0, 4.0, 0.5, key=f"hyb_r_{i}")
            seed_rows.append({"movie_id": title_to_id[t], "rating": r})

    if st.button("Получить рекомендации", key="hyb_run"):
        seed = pd.DataFrame(seed_rows).drop_duplicates(subset="movie_id")
        recs = hybrid.for_user(seed, top_n=15, alpha=alpha)
        show_table(recs)

with tab_cold:
    st.header("Решение проблемы холодного старта")
    st.write(
        "У нового пользователя нет оценок. Стратегия: спросить любимые жанры "
        "и рекомендовать фильмы с высоким Bayesian-усреднённым рейтингом, "
        "пересекающиеся по жанрам."
    )
    chosen = st.multiselect("Любимые жанры", list_genres(), default=["Comedy", "Adventure"])
    top_n = st.slider("Сколько показать", 5, 30, 10, key="cs_n")
    min_ratings = st.slider("Минимум оценок у фильма", 5, 200, 30, key="cs_m")
    if chosen:
        show_table(cold_start_by_genres(chosen, top_n=top_n, min_ratings=min_ratings))
    else:
        st.info("Выберите хотя бы один жанр или см. вкладку «Популярные».")
        show_table(popular_movies(top_n=top_n, min_ratings=min_ratings))

with tab_eval:
    st.header("Сравнение моделей (leave-last-K-out)")
    st.write(
        "Для каждого случайного пользователя берём его оценки по времени, "
        "последние K — тест, остальные — train. Collaborative-модель "
        "переобучается на train (без leakage). Релевант: rating ≥ 4. "
        "Метрики: Precision@N, Recall@N, F1@N."
    )
    n_users = st.slider("Сколько пользователей", 20, 300, 100, step=10)
    k = st.slider("Прячем последних оценок (K)", 1, 20, 5)
    top_n = st.slider("top-N рекомендаций", 5, 30, 10, key="ev_n")
    alpha = st.slider("alpha для hybrid", 0.0, 1.0, 0.5, 0.05, key="ev_alpha")
    if st.button("Запустить оценку"):
        with st.spinner("Переобучаем collaborative на train и считаем метрики..."):
            res = evaluate_models(top_n=top_n, n_users=n_users, k=k, alpha=alpha)
        df = pd.DataFrame(res).T
        df.index.name = "model"
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df[["precision", "recall", "f1"]])

with tab_db:
    st.header("Архитектура БД")
    conn = sqlite3.connect(DB_PATH)
    schema = pd.read_sql(
        """
        SELECT name AS table_name, sql AS create_sql
        FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """,
        conn,
    )
    schema["rows"] = schema["table_name"].apply(
        lambda t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    )
    conn.close()
    st.dataframe(schema[["table_name", "rows"]], use_container_width=True, hide_index=True)
    st.code(";\n\n".join(s for s in schema["create_sql"] if s) + ";", language="sql")
