from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
from typing import Dict, Tuple

# ---------- Mapeamentos utilitários ----------
def build_indexers(ratings: pd.DataFrame) -> Tuple[Dict[int,int], Dict[int,int], np.ndarray, np.ndarray]:
    """Cria mapas id->idx (usuários/itens) e vetores idx->id."""
    users = ratings["user_id"].astype(int).unique()
    items = ratings["anime_id"].astype(int).unique()
    uid2ix = {u:i for i,u in enumerate(np.sort(users))}
    iid2ix = {a:i for i,a in enumerate(np.sort(items))}
    ix2uid = np.array(sorted(users), dtype=int)
    ix2iid = np.array(sorted(items), dtype=int)
    return uid2ix, iid2ix, ix2uid, ix2iid

def build_matrix(ratings: pd.DataFrame) -> Tuple[csr_matrix, Dict[int,int], Dict[int,int], np.ndarray, np.ndarray]:
    """Matriz esparsa usuários x itens (ratings >=0)."""
    df = ratings[ratings["rating"] >= 0].copy()
    if df.empty:
        return csr_matrix((0,0)), {}, {}, np.array([],dtype=int), np.array([],dtype=int)
    uid2ix, iid2ix, ix2uid, ix2iid = build_indexers(df)
    rows = df["user_id"].map(uid2ix).values
    cols = df["anime_id"].map(iid2ix).values
    vals = df["rating"].astype(float).values
    m = csr_matrix((vals, (rows, cols)), shape=(len(uid2ix), len(iid2ix)))
    return m, uid2ix, iid2ix, ix2uid, ix2iid

# ---------- Item-based com cosseno rápido ----------
def column_l2_norms(X: csr_matrix) -> np.ndarray:
    # norma L2 por coluna (para cosine); X é csr -> melhor converter para csc p/ coluna
    Xc = X.tocsc()
    sq = np.array(Xc.power(2).sum(axis=0)).ravel()
    norm = np.sqrt(np.maximum(sq, 1e-12))
    return norm

def recommend_item_based_sparse(
    ratings: pd.DataFrame,
    user_id: int,
    n_recs: int = 10,
    filters: Dict[str,str] | None = None,
) -> Dict[int, float]:
    """
    Recomendação item-based (cosine por coluna) com backfill de popularidade.
    1) Calcula o score por similaridade (sem materializar matriz densa).
    2) Exclui itens já avaliados pelo usuário.
    3) Se faltarem candidatos (scores finitos), completa por popularidade global.
    """
    X, uid2ix, iid2ix, ix2uid, ix2iid = build_matrix(ratings)
    if user_id not in uid2ix or X.shape[1] == 0:
        return {}

    uix = uid2ix[user_id]
    x_u = X.getrow(uix)                   # vetor do usuário (1 x n_items)
    rated_items_ix = x_u.indices
    rated_values   = x_u.data
    if rated_items_ix.size == 0:
        return {}

    # --- Similaridade cosseno por coluna (normalização L2 por coluna) ---
    norms = column_l2_norms(X)            # (n_items,)
    Xc = X.tocsc()

    # v = soma_j ( (Xc[:, j] / ||col_j||) * rating_j )
    v = np.zeros(Xc.shape[0], dtype=float)  # tamanho = n_users
    for j_idx, r in zip(rated_items_ix, rated_values):
        col = Xc.getcol(j_idx)
        if norms[j_idx] > 0:
            v += (col.toarray().ravel() / norms[j_idx]) * float(r)

    # scores_i = (Xc[:, i] / ||col_i||)^T @ v
    scores = np.full(Xc.shape[1], -np.inf, dtype=float)
    for i in range(Xc.shape[1]):
        dn = norms[i]
        if dn > 0:
            col = Xc.getcol(i).toarray().ravel()
            scores[i] = float(col.dot(v) / dn)

    # Nunca recomende itens já avaliados
    if rated_items_ix.size > 0:
        scores[rated_items_ix] = -np.inf

    # --- Backfill de popularidade (quando faltam candidatos bons) ---
    finite_mask = np.isfinite(scores) & (scores > -np.inf)
    have = int(np.sum(finite_mask))
    need = int(max(0, n_recs - have))

    if need > 0:
        # Popularidade ~ contagem de interações por item
        Xc_pop = Xc
        pop = np.array((Xc_pop > 0).sum(axis=0)).ravel().astype(float)
        if rated_items_ix.size > 0:
            pop[rated_items_ix] = -np.inf

        fill_ix = np.where(~finite_mask)[0]
        if fill_ix.size > 0:
            order = np.argsort(-pop[fill_ix])    # mais populares primeiro
            take_ix = fill_ix[order[:need]]
            if take_ix.size > 0:
                bumps = np.linspace(-1e-6, -1e-9, num=take_ix.size)
                scores[take_ix] = bumps

    # Seleciona Top-K final
    k = min(n_recs, len(scores)) if len(scores) > 0 else 0
    if k <= 0:
        return {}

    top_ix = np.argpartition(-scores, kth=k-1)[:k]
    top_ix = top_ix[np.argsort(-scores[top_ix])]

    out = {}
    for i in top_ix:
        s = scores[i]
        if np.isfinite(s) and s > -np.inf:
            out[int(ix2iid[i])] = float(s)
    return out

# ---------- User-based opcional (para análises) ----------
def recommend_user_based_sparse(
    ratings: pd.DataFrame,
    user_id: int,
    n_neighbors: int = 30,
    n_recs: int = 10,
) -> Dict[int, float]:
    """User-based simples (cosine row-row) com compatibilidade numpy/scipy ampla."""

    def _to_1d_array(x) -> np.ndarray:
        """Converte x (sparse/dense/matrix) em np.ndarray 1D float, de forma robusta."""
        if hasattr(x, "toarray"):         # scipy.sparse
            arr = x.toarray()
        elif hasattr(x, "A"):             # numpy.matrix
            arr = x.A
        else:
            arr = np.asarray(x)
        return np.asarray(arr, dtype=float).ravel()

    X, uid2ix, iid2ix, ix2uid, ix2iid = build_matrix(ratings)
    if user_id not in uid2ix or X.shape[1] == 0 or X.shape[0] == 0:
        return {}

    uix = uid2ix[user_id]

    # norma L2 por linha (cosine); evita divisão por zero
    row_norm = _to_1d_array(np.sqrt(X.power(2).sum(axis=1)))
    row_norm[row_norm == 0] = 1.0
    Xr = X.multiply(1.0 / row_norm[:, None])  # csr normalizado por linha

    # similaridade com todos (n_users x 1) -> array 1D
    sim = _to_1d_array(Xr @ Xr.getrow(uix).T)
    sim[uix] = 0.0

    if sim.size <= 1:
        return {}

    k = min(n_neighbors, max(1, sim.size - 1))
    nbr_ix = np.argpartition(-sim, kth=k-1)[:k]
    weights = sim[nbr_ix]

    sub = X[nbr_ix, :]             # (k x n_items) sparse
    sub_dense = sub.toarray()      # densifica uma vez p/ broadcasting simples
    numer = (weights[:, None] * sub_dense).sum(axis=0)

    denom = (np.abs(weights[:, None]) * (sub_dense > 0)).sum(axis=0) + 1e-9
    score = numer / denom

    # remove itens já avaliados pelo usuário
    rated = X.getrow(uix).indices
    if rated.size > 0:
        score[rated] = -np.inf

    if score.size == 0:
        return {}
    k_items = min(n_recs, score.size)
    top_ix = np.argpartition(-score, kth=k_items-1)[:k_items]
    top_ix = top_ix[np.argsort(-score[top_ix])]

    out: Dict[int, float] = {}
    for i in top_ix:
        s = float(score[i])
        if np.isfinite(s) and s > -np.inf:
            out[int(ix2iid[i])] = s
    return out

# ---------- Avaliação (acurácia) ----------
def evaluate_accuracy_for_user(
    ratings: pd.DataFrame,
    user_id: int,
    algo: str = "item",
    test_ratio: float = 0.2,
    top_k: int = 10,
    relevant_threshold: float = 7.0,
) -> Dict[str, float]:
    """
    Avalia acurácia Top-K com split 80/20 estratificado por relevância (>= threshold).
    - Garante que o conjunto de teste contenha positivos quando houver.
    - Preserva índices originais ao amostrar para evitar KeyError no drop.
    """
    # Dados do usuário (somente ratings válidos >= 0)
    dfu = ratings[(ratings["user_id"] == user_id) & (ratings["rating"] >= 0)].copy()
    if dfu.shape[0] < 3:
        return {"hits": 0, "recommended": 0, "accuracy": 0.0}

    # Estratificação por relevância
    pos = dfu[dfu["rating"] >= relevant_threshold]
    neg = dfu[dfu["rating"] < relevant_threshold]

    n_pos_test = max(1, int(len(pos) * test_ratio)) if len(pos) > 0 else 0
    n_neg_test = max(0, int(len(neg) * test_ratio))

    # >>> preserva índices originais (sem ignore_index)
    pos_sample = pos.sample(n=min(n_pos_test, len(pos)), random_state=42) if len(pos) else pos.iloc[0:0]
    neg_sample = neg.sample(n=min(n_neg_test, len(neg)), random_state=42) if len(neg) else neg.iloc[0:0]
    test = pd.concat([pos_sample, neg_sample], ignore_index=False)

    # Treino = dfu - teste (pelos índices originais)
    train = dfu.drop(index=test.index, errors="ignore")

    # Junta com demais usuários (mantém o dataset global para a recomendação)
    others = ratings[(ratings["user_id"] != user_id) & (ratings["rating"] >= 0)]
    train_all = pd.concat([train, others], ignore_index=True)

    # Gera recomendações usando apenas os dados de treino
    if algo == "user":
        recs = recommend_user_based_sparse(train_all, user_id=user_id, n_recs=top_k)
    else:
        recs = recommend_item_based_sparse(train_all, user_id=user_id, n_recs=top_k)

    recommended_ids = set(int(k) for k in recs.keys())
    truth_positive = set(test[test["rating"] >= relevant_threshold]["anime_id"].astype(int).tolist())

    hits = len(recommended_ids & truth_positive)
    recommended = len(recommended_ids)
    acc = (hits / recommended) if recommended > 0 else 0.0

    return {"hits": int(hits), "recommended": int(recommended), "accuracy": float(acc)}
