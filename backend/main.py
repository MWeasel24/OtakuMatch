from __future__ import annotations
import os, hashlib
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from functools import lru_cache

from .schemas import (
    RatingIn,
    RecommendQuery,
    EvaluateQuery,
    SignupIn,
    LoginIn,
    UpdateUserIn,
    WipeIn,
)

DATA_DIR = os.environ.get("ANIME_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
ANIME_CSV = os.path.join(DATA_DIR, "Anime.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "Ratings.csv")
USERS_CSV = os.path.join(DATA_DIR, "Users.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")
PLACEHOLDER = "/images/placeholder.jpg"

def deep_clean(obj):
    import math
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): deep_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj

def safe_json(data):
    return JSONResponse(content=deep_clean(data))

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def load_catalog() -> pd.DataFrame:
    anime = pd.read_csv(ANIME_CSV, low_memory=False)
    anime.columns = [str(c).strip().lower() for c in anime.columns]
    if "animeid" in anime.columns and "anime_id" not in anime.columns:
        anime = anime.rename(columns={"animeid":"anime_id"})
    for col in ["anime_id","name","genre","type","episodes","rating","members"]:
        if col not in anime.columns: anime[col] = None
    anime["anime_id"] = pd.to_numeric(anime["anime_id"], errors="coerce").astype("Int64")
    anime["episodes"] = pd.to_numeric(anime["episodes"], errors="coerce").astype("Int64")
    anime["rating"] = pd.to_numeric(anime["rating"], errors="coerce").astype(float)
    anime["members"] = pd.to_numeric(anime["members"], errors="coerce").astype("Int64")
    anime["name"] = anime["name"].astype(str)
    anime["genre"] = anime["genre"].astype(str)
    anime["type"] = anime["type"].astype(str)
    return anime

def load_ratings() -> pd.DataFrame:
    if not os.path.exists(RATINGS_CSV):
        return pd.DataFrame(columns=["user_id","anime_id","rating"])
    df = pd.read_csv(RATINGS_CSV)
    if df.empty: return pd.DataFrame(columns=["user_id","anime_id","rating"])
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").fillna(-1).astype(int)
    df["anime_id"] = pd.to_numeric(df["anime_id"], errors="coerce").fillna(-1).astype(int)
    df["rating"]   = pd.to_numeric(df["rating"], errors="coerce").fillna(-1).astype(int)
    return df

def load_users() -> pd.DataFrame:
    cols = ["user_id", "username", "password_hash"]
    if not os.path.exists(USERS_CSV):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(USERS_CSV)
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Garante colunas
    for c in cols:
        if c not in df.columns:
            df[c] = None

    # Tipagem/sanitização
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").fillna(-1).astype(int)
    df["username"] = df["username"].astype(str).fillna("")
    df["password_hash"] = df["password_hash"].astype(str).fillna("")

    # Remove linhas inválidas (sem username)
    df = df[df["username"].str.strip() != ""].copy()

    return df[cols]

anime_df = load_catalog()
ratings_df = load_ratings()
users_df = load_users()

app = FastAPI(title="Anime Recommender API", version="2025.09")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
os.makedirs(IMG_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMG_DIR), name="images")

def to_jsonable(df: pd.DataFrame) -> List[dict]:
    if df is None or df.empty:
        return []
    safe = df.copy()
    keep = ["anime_id","name","genre","type","episodes","rating","members"]
    cols = [c for c in keep if c in safe.columns]
    safe = safe[cols]
    for col in safe.columns:
        if col in ("name","genre","type"):
            safe[col] = safe[col].astype(str)
        if col in ("anime_id","episodes","members"):
            safe[col] = pd.to_numeric(safe[col], errors="coerce").astype("Int64")
        if col == "rating":
            safe[col] = pd.to_numeric(safe[col], errors="coerce").astype(float)
    safe = safe.replace([np.inf, -np.inf], np.nan)
    safe = safe.where(pd.notna(safe), None)
    for col in safe.columns:
        if str(safe[col].dtype) == "Int64":
            safe[col] = safe[col].astype(object)
    return safe.to_dict(orient="records")

def local_image_path(anime_id: int) -> str:
    p1 = os.path.join(IMG_DIR, f"{anime_id}.jpg")
    p2 = os.path.join(IMG_DIR, f"{anime_id}.png")
    if os.path.exists(p1): return f"/images/{anime_id}.jpg"
    if os.path.exists(p2): return f"/images/{anime_id}.png"
    return PLACEHOLDER

@lru_cache(maxsize=1)
def cached_genres() -> List[str]:
    g = anime_df["genre"].dropna().astype(str).str.split(",").explode().str.strip()
    return sorted({x for x in g if x})

@lru_cache(maxsize=1)
def cached_popular_ids() -> List[int]:
    df = anime_df.copy()
    key = "members" if "members" in df.columns else "rating"
    df[key] = pd.to_numeric(df[key], errors="coerce")
    df = df.sort_values(key, ascending=False, na_position="last")
    return [int(x) for x in df["anime_id"].dropna().astype(int).tolist()]

@lru_cache(maxsize=1)
def cached_stats() -> dict:
    total = int(anime_df.shape[0])
    types = anime_df["type"].astype(str).value_counts().to_dict()
    g = anime_df["genre"].dropna().astype(str).str.split(",").explode().str.strip()
    genres_top = g.value_counts().head(100).to_dict()
    r = pd.to_numeric(anime_df["rating"], errors="coerce").dropna()
    hist = r.round().value_counts().sort_index().astype(int).to_dict()
    mean_rating = float(r.mean()) if len(r) else None
    return {"total": total, "types": types, "genres_top": genres_top, "rating_hist": hist, "mean_rating": mean_rating}

from .recommender import (
    recommend_item_based_sparse,
    evaluate_accuracy_for_user,
    build_matrix,
    column_l2_norms,
)

@app.get("/health")
def health():
    return {"status":"ok", "anime_rows": int(len(anime_df)), "ratings_rows": int(len(ratings_df)), "users": int(len(users_df))}

@app.get("/genres")
def genres():
    return cached_genres()

@app.get("/stats")
def stats():
    return cached_stats()

@app.get("/user_stats")
def user_stats(user_id: int = Query(...)):
    dfu = ratings_df[(ratings_df["user_id"] == int(user_id)) & (ratings_df["rating"] >= 0)]
    return {"ratings_count": int(len(dfu))}

@app.post("/signup")
def signup(s: SignupIn):
    global users_df
    if s.username.strip() == "" or s.password.strip() == "":
        raise HTTPException(status_code=400, detail="Username e senha são obrigatórios.")
    if not users_df.empty and (users_df["username"].astype(str).str.lower() == s.username.strip().lower()).any():
        raise HTTPException(status_code=409, detail="Usuário já existe.")
    # Evita colisão com histórico de Ratings.csv
    max_users = int(users_df["user_id"].max()) if not users_df.empty else 0
    max_ratings = int(ratings_df["user_id"].max()) if not ratings_df.empty else 0
    new_id = max(max_users, max_ratings) + 1

    row = {"user_id": new_id, "username": s.username, "password_hash": _hash(s.password)}
    users_df = pd.concat([users_df, pd.DataFrame([row])], ignore_index=True)
    users_df.to_csv(USERS_CSV, index=False)
    return {"ok": True, "user_id": new_id, "username": s.username}

@app.post("/login")
def login(l: LoginIn):
    if users_df.empty:
        raise HTTPException(status_code=404, detail="Nenhum usuário cadastrado.")
    uname = (l.username or "").strip().lower()
    if uname == "":
        raise HTTPException(status_code=400, detail="Usuário é obrigatório.")
    # usa astype(str) para evitar erro do .str com tipos não-string
    mask = users_df["username"].astype(str).str.lower() == uname
    if not mask.any():
        raise HTTPException(status_code=401, detail="Usuário/senha inválidos.")
    rec = users_df[mask].iloc[0]
    if str(rec["password_hash"]) != _hash(l.password):
        raise HTTPException(status_code=401, detail="Usuário/senha inválidos.")
    return {"ok": True, "user_id": int(rec["user_id"]), "username": str(rec["username"])}

@app.get("/catalog")
def catalog(q: Optional[str] = None, limit: int = 50000):
    df = anime_df
    if q:
        ql = str(q).lower()
        df = df[df["name"].str.lower().str.contains(ql, na=False, regex=False)]
    if limit and limit > 0: df = df.head(int(limit))
    return safe_json(to_jsonable(df))

@app.get("/anime")
def anime(id: int = Query(...)):
    df = anime_df[anime_df["anime_id"] == int(id)]
    if df.empty: raise HTTPException(status_code=404, detail="Anime não encontrado.")
    return safe_json(to_jsonable(df.head(1))[0])

@app.get("/popular")
def popular(limit: int = 25):
    ids = cached_popular_ids()
    sel = set(ids[:max(0,int(limit))])
    df = anime_df[anime_df["anime_id"].isin(list(sel))]
    order = {a:i for i,a in enumerate(ids)}
    df = df.sort_values(by="anime_id", key=lambda s: s.map(order))
    return safe_json(to_jsonable(df))

@app.get("/random")
def random_anime(limit: int = 15):
    sample = anime_df.sample(n=min(int(limit), len(anime_df)), random_state=None)
    return safe_json(to_jsonable(sample))

@app.get("/image/{anime_id}")
def image_url(anime_id: int):
    path = local_image_path(int(anime_id))
    return {"url": path}

@app.post("/rate")
def rate(r: RatingIn):
    global ratings_df
    if r.rating < 0 or r.rating > 10:
        raise HTTPException(status_code=400, detail="A avaliação deve ser entre 0 e 10.")
    mask = (ratings_df["user_id"]==r.user_id) & (ratings_df["anime_id"]==r.anime_id)
    if mask.any():
        ratings_df.loc[mask, "rating"] = int(r.rating)
    else:
        ratings_df = pd.concat([ratings_df, pd.DataFrame([{"user_id": int(r.user_id), "anime_id": int(r.anime_id), "rating": int(r.rating)}])], ignore_index=True)
    ratings_df.to_csv(RATINGS_CSV, index=False)
    return {"ok": True}

def apply_filters(df: pd.DataFrame, filters: Dict[str, str]) -> pd.DataFrame:
    out = df
    if not filters: return out
    t = filters.get("type")
    if t: out = out[out["type"].astype(str).str.lower() == t.lower()]
    g = filters.get("genre")
    if g: out = out[out["genre"].astype(str).str.contains(g, case=False, na=False)]
    max_ep = filters.get("max_episodes")
    if max_ep: out = out[pd.to_numeric(out["episodes"], errors="coerce").fillna(0) <= float(max_ep)]
    return out

@app.post("/recommend")
def recommend(query: RecommendQuery):
    uid = int(query.user_id)

    # Verifica se o usuário tem avaliações válidas
    has_user = (
        (not ratings_df.empty)
        and (ratings_df["user_id"] == uid).any()
        and (ratings_df[(ratings_df["user_id"] == uid) & (ratings_df["rating"] >= 0)].shape[0] > 0)
    )
    if not has_user:
        return []

    # Item-based
    recs = recommend_item_based_sparse(ratings_df, user_id=uid, n_recs=query.n_recs)

    df = anime_df[anime_df["anime_id"].isin(list(recs.keys()))].copy()
    if df.empty:
        return []

    # Aplica filtros (se houver)
    if query.filters:
        df = apply_filters(df, query.filters)

    # Ordena por score previsto (se existir) — recomendação já vem “ordenada por relevância”
    df["pred_score"] = df["anime_id"].map(recs).astype(float)
    df = df.sort_values("pred_score", ascending=False, na_position="last")

    return safe_json(to_jsonable(df.head(int(query.n_recs))))

@app.get("/similar")
def similar(anime_id: int = Query(...), n: int = 12):
    X, uid2ix, iid2ix, ix2uid, ix2iid = build_matrix(ratings_df)
    if anime_id not in iid2ix: return []
    col_idx = iid2ix[anime_id]
    Xc = X.tocsc()
    norms = column_l2_norms(X)
    base_col = Xc.getcol(col_idx).toarray().ravel()
    if norms[col_idx] > 0:
        base_col = base_col / norms[col_idx]
    sims = np.zeros(Xc.shape[1], dtype=float)
    for i in range(Xc.shape[1]):
        col = Xc.getcol(i).toarray().ravel()
        dn = norms[i]
        sims[i] = float(col.dot(base_col) / dn) if dn > 0 else 0.0
    sims[col_idx] = -np.inf
    top_ix = np.argpartition(-sims, kth=min(n, len(sims)-1))[:n*2]
    top_ix = top_ix[np.argsort(-sims[top_ix])][:n]
    ids = [int(ix2iid[i]) for i in top_ix if np.isfinite(sims[i])]
    df = anime_df[anime_df["anime_id"].isin(ids)].copy()
    order = {a:i for i,a in enumerate(ids)}
    df = df.sort_values(by="anime_id", key=lambda s: s.map(order))
    return safe_json(to_jsonable(df))

@app.post("/evaluate")
def evaluate(q: EvaluateQuery):
    # Item-based na avaliação
    res = evaluate_accuracy_for_user(
        ratings_df,
        q.user_id,
        algo="item",
        test_ratio=q.test_ratio,
        top_k=q.top_k,
        relevant_threshold=q.relevant_threshold,
    )
    return res

@app.get("/user_ratings")
def user_ratings(user_id: int):
    dfu = ratings_df[(ratings_df["user_id"] == int(user_id)) & (ratings_df["rating"] >= 0)].copy()
    if dfu.empty: return []
    dfj = dfu.merge(anime_df[["anime_id","name"]], on="anime_id", how="left")
    dfj = dfj[["anime_id","name","rating"]].sort_values(["rating","name"], ascending=[False, True])
    dfj["rating"] = pd.to_numeric(dfj["rating"], errors="coerce").fillna(0).astype(int)
    return dfj.to_dict(orient="records")

# endpoint para “limpar minhas avaliações”
@app.post("/wipe_user_ratings")
def wipe_user_ratings(w: WipeIn):
    global ratings_df
    if ratings_df.empty:
        return {"ok": True, "removed": 0}
    before = ratings_df.shape[0]
    ratings_df = ratings_df[ratings_df["user_id"] != int(w.user_id)].copy()
    ratings_df.to_csv(RATINGS_CSV, index=False)
    removed = before - ratings_df.shape[0]
    return {"ok": True, "removed": int(removed)}

@app.post("/update_user")
def update_user(u: UpdateUserIn):
    global users_df
    # Recarrega do disco para garantir estado mais atual
    users_df = load_users()

    if users_df.empty:
        raise HTTPException(status_code=404, detail="Nenhum usuário cadastrado.")

    mask = users_df["user_id"].astype(int) == int(u.user_id)
    if not mask.any():
        raise HTTPException(status_code=404, detail="Usuário não encontrado.")

    new_username = (u.new_username or "").strip()
    new_password = (u.new_password or "").strip()

    # Troca de username (garante unicidade case-insensitive)
    if new_username:
        if users_df[~mask]["username"].astype(str).str.lower().eq(new_username.lower()).any():
            raise HTTPException(status_code=409, detail="Nome de usuário já existe.")
        users_df.loc[mask, "username"] = new_username

    # Troca de senha
    if new_password:
        users_df.loc[mask, "password_hash"] = _hash(new_password)

    users_df.to_csv(USERS_CSV, index=False)

    # Retorna username atualizado
    current_username = str(users_df.loc[mask, "username"].iloc[0])
    return {"ok": True, "user_id": int(u.user_id), "username": current_username}
