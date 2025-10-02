from __future__ import annotations
import pandas as pd
import streamlit as st
from typing import Dict, Any, List
from utils import api_get, api_post, image_url_or_placeholder, APIError, wipe_user_ratings

# ------------------------------------------------------------
# CONFIGURAÇÃO DE PÁGINA E ESTILOS
# ------------------------------------------------------------
st.set_page_config(page_title="OtakuMatch", page_icon="🍙", layout="wide")

PRIMARY = "#ff7a1a"
BG = "#0d0f14"
CARD = "#141827"
TEXT = "#e9ecff"
MUTED = "#a1a9b8"

st.markdown(f"""
<style>
.stApp {{
  background: radial-gradient(1200px 700px at 20% -10%, #1b2238 0%, transparent 60%),
              radial-gradient(1200px 700px at 80% 0%, transparent 60%),
              {BG};
  color:{TEXT};
}}
.site-title {{
  font-size:56px; font-weight:900;
  background:linear-gradient(90deg, {PRIMARY}, #ffd2b3);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.card {{
  width:100%; height:330px; background:{CARD}; border:1px solid #222739; border-radius:14px;
  box-shadow:0 12px 28px rgba(0,0,0,.35); padding:10px; display:flex; flex-direction:column; gap:8px;
  position:relative; overflow:hidden; transition: transform .12s ease, box-shadow .12s ease;
}}
.card:hover {{ transform: translateY(-2px); box-shadow:0 16px 32px rgba(0,0,0,.45); }}
.card-img {{ width:100%; height:170px; border-radius:10px; overflow:hidden; background:#0a0e1a; flex:0 0 auto; }}
.card-img img {{ width:100%; height:100%; object-fit:cover; display:block; }}
.card-title {{ margin:.15rem 0 0; font-size:.98rem; line-height:1.2; color:{TEXT}; }}
.card-genres {{ color:{MUTED}; font-size:.82rem; min-height:2.1em; }}
.card-meta {{ display:flex; justify-content:space-between; font-size:.82rem; color:{MUTED}; }}

</style>
""", unsafe_allow_html=True)

# Estilos específicos para a página de detalhes
st.markdown("""
<style>
/* Capa só da página de detalhes */
.detail-cover {
  width: 100%;
  height: 420px;            /* ajuste aqui o tamanho desejado */
  border-radius: 14px;
  overflow: hidden;
  background: #0a0e1a;
}
.detail-cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

/* Responsivo opcional */
@media (max-width: 768px){
  .detail-cover { height: 300px; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# ESTADO GLOBAL
# ------------------------------------------------------------
ss = st.session_state
if "user" not in ss: ss.user = None
if "tab" not in ss: ss.tab = st.query_params.get("tab", "inicio")
if "anime" not in ss: ss.anime = st.query_params.get("anime")
if "search_query" not in ss: ss.search_query = st.query_params.get("q", "") or ""
if "last_recs" not in ss: ss.last_recs = []
if "genres_all" not in ss: ss.genres_all = []
if "backend_ok" not in ss: ss.backend_ok = False

# >>> NOVO: restaura login a partir dos query params (evita "perder login" ao navegar)
try:
    qp_uid = st.query_params.get("uid")
    qp_uname = st.query_params.get("uname")
    if ss.user is None and qp_uid and qp_uname:
        ss.user = {"user_id": int(qp_uid), "username": str(qp_uname)}
except Exception:
    pass

# ------------------------------------------------------------
# NAVEGAÇÃO (query params, mesma aba)
# ------------------------------------------------------------
def goto(tab: str, anime: int | None = None, q: str | None = None):
    ss.tab = tab
    st.query_params["tab"] = tab
    if anime is not None:
        ss.anime = str(anime)
        st.query_params["anime"] = str(anime)
    else:
        if "anime" in st.query_params: del st.query_params["anime"]
    if q is not None:
        ss.search_query = q
        st.query_params["q"] = q
    else:
        if "q" in st.query_params: del st.query_params["q"]

# ------------------------------------------------------------
# CHECAGEM DE BACKEND
# ------------------------------------------------------------
try:
    api_get("/health", timeout=5)
    ss.backend_ok = True
except Exception:
    ss.backend_ok = False

# ------------------------------------------------------------
# SIDEBAR (Login / Cadastro)
# ------------------------------------------------------------
with st.sidebar:
    st.header("Conta")
    if ss.user:
        st.write(f"Usuário: {ss.user['username']} (ID {ss.user['user_id']})")
        if st.button("Sair", use_container_width=True):
            ss.user = None
            # limpa query params relacionados ao user
            for k in ("uid", "uname", "anime"):
                if k in st.query_params:
                    del st.query_params[k]
            st.query_params["tab"] = "inicio"
            st.rerun()
    else:
        t1, t2 = st.tabs(["Entrar", "Criar conta"])
        with t1:
            u = st.text_input("Usuário")
            p = st.text_input("Senha", type="password")
            if st.button("Entrar", use_container_width=True):
                try:
                    data = api_post("/login", {"username": u, "password": p}, timeout=20)
                    ss.user = {"user_id": data["user_id"], "username": data["username"]}
                    # persiste nos query params p/ não "perder login" ao navegar
                    st.query_params["uid"] = str(data["user_id"])
                    st.query_params["uname"] = data["username"]
                    st.query_params["tab"] = "inicio"
                    st.rerun()
                except APIError as e:
                    st.error(e.args[0] if e.args else "Usuário ou senha inválidos.")
                except Exception:
                    st.error("Não foi possível entrar.")
        with t2:
            u2 = st.text_input("Novo usuário")
            p2 = st.text_input("Nova senha", type="password")
            if st.button("Criar conta", use_container_width=True):
                try:
                    data = api_post("/signup", {"username": u2, "password": p2}, timeout=20)
                    ss.user = {"user_id": data["user_id"], "username": data.get("username", u2)}
                    # persiste nos query params p/ não "perder login" ao navegar
                    st.query_params["uid"] = str(data["user_id"])
                    st.query_params["uname"] = ss.user["username"]
                    st.query_params["tab"] = "inicio"
                    st.rerun()
                except APIError as e:
                    st.error(e.args[0] if e.args else "Não foi possível criar a conta.")
                except Exception:
                    st.error("Não foi possível criar.")

# ------------------------------------------------------------
# HEADER + NAV + BUSCA (Enter)
# ------------------------------------------------------------
st.markdown('<h1 class="site-title">OtakuMatch</h1>', unsafe_allow_html=True)
if ss.user:
    nav1, nav2, nav3, nav4, searchc = st.columns([1,1,1,1,2])
else:
    nav1, nav2, nav3, searchc = st.columns([1,1,1,3])

with nav1:
    if st.button("Início", use_container_width=True): goto("inicio")
with nav2:
    if st.button("Recomendações", use_container_width=True): goto("recs")
with nav3:
    if st.button("Análise", use_container_width=True): goto("analise")
if ss.user:
    with nav4:
        if st.button("Perfil", use_container_width=True): goto("perfil")

def _submit_search():
    q = st.session_state.get("header_q", "").strip()
    if q:
        goto("busca", q=q)

with searchc:
    st.text_input(
        label="Pesquisar",
        value=ss.search_query,
        key="header_q",
        placeholder="Buscar…",
        on_change=_submit_search,
        label_visibility="collapsed",
    )

# ------------------------------------------------------------
# COMPONENTES DE UI
# ------------------------------------------------------------
def render_card(item: Dict[str, Any]):
    mal_id = int(item.get("anime_id", -1))
    name = item.get("name", "(sem nome)")
    genres = (item.get("genre") or "")[:100]
    typ = item.get("type", "-")
    eps = item.get("episodes", "-")
    rt = item.get("rating", "-")
    img = image_url_or_placeholder(mal_id)

    # monta href preservando a sessão via query params
    if ss.get("user"):
        href = f"?tab=detalhe&anime={mal_id}&uid={ss.user['user_id']}&uname={ss.user['username']}"
    else:
        href = f"?tab=detalhe&anime={mal_id}"

    # torna TODO o card clicável com um <a> envolvendo o card
    st.markdown(f"""
<a href="{href}" target="_self" style="text-decoration:none; color:inherit; display:block;">
  <div class="card">
    <div class="card-img"><img src="{img}" alt="Capa"></div>
    <div class="card-title">{name}</div>
    <div class="card-genres">{genres}</div>
    <div class="card-meta">
      <span>📺 {typ}</span><span>🎬 {eps} ep.</span><span>⭐ {rt}</span>
    </div>
  </div>
</a>
""", unsafe_allow_html=True)

def render_grid_5(items: List[Dict[str, Any]]):
    if not items: return
    ncols = 5
    rows = (len(items) + ncols - 1) // ncols
    idx = 0
    for _ in range(rows):
        cols = st.columns(ncols, gap="large")
        for c in range(ncols):
            if idx >= len(items): break
            with cols[c]:
                render_card(items[idx])
                idx += 1
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# PÁGINAS
# ------------------------------------------------------------
def page_inicio():
    st.markdown("# Boas-vindas ao OtakuMatch")
    st.caption("Descubra animes que combinam com você.")
    if not ss.backend_ok:
        st.warning("Backend indisponível.")
        return
    try:
        hot = api_get("/popular", params={"limit": 10}, timeout=30)
        st.markdown("#### Em alta")
        render_grid_5(hot)
    except Exception:
        st.error("Erro ao carregar 'Em alta'.")
    try:
        rnd = api_get("/random", params={"limit": 15}, timeout=30)
        st.markdown("#### Descubra novos animes")
        render_grid_5(rnd)
    except Exception:
        st.info("Não foi possível carregar aleatórios.")

def page_busca():
    st.markdown(f"## Busca: {ss.search_query or ''}")
    if not ss.backend_ok:
        st.warning("Backend indisponível.")
        return

    cA, cB = st.columns([2,2])
    with cA:
        letra = st.selectbox(
            "Filtrar por letra:",
            ["(todas)"] + [chr(x) for x in range(65, 91)] + ["#"],
            key="busca_letra"
        )
    with cB:
        tipo = st.selectbox(
            "Tipo:",
            ["(todos)", "TV","Movie","OVA","Special","Music","ONA"],
            key="busca_tipo"
        )

    qterm = (ss.search_query or "").strip()
    try:
        res = api_get("/catalog", params={"q": qterm, "limit": 2000}, timeout=60) if qterm else api_get("/catalog", params={"limit": 2000}, timeout=60)
        df = pd.DataFrame(res)
        if df.empty:
            st.info("Nenhum anime encontrado.")
            return

        if letra and letra != "(todas)":
            if letra == "#":
                mask = ~df["name"].str[0].str.upper().str.match(r"[A-Z]", na=False)
            else:
                mask = df["name"].str.upper().str.startswith(letra, na=False)
            df = df[mask]
        if tipo and tipo != "(todos)":
            df = df[df["type"].str.lower() == tipo.lower()]

        show = df.copy()
        show["Capa"] = show["anime_id"].apply(lambda x: image_url_or_placeholder(int(x)))

        # >>> AJUSTADO: se logado, inclui uid/uname no link para não perder login
        def _mk_open_link(anime_id: int) -> str:
            if ss.get("user"):
                return f"?tab=detalhe&anime={int(anime_id)}&uid={ss.user['user_id']}&uname={ss.user['username']}"
            else:
                return f"?tab=detalhe&anime={int(anime_id)}"

        show["Abrir"] = show["anime_id"].apply(_mk_open_link)

        show = show.rename(columns={
            "name": "Título",
            "type": "Tipo",
            "episodes": "Ep.",
            "rating": "Média",
        })[["Capa","Título","Tipo","Ep.","Média","Abrir"]]

        config = {
            "Capa": st.column_config.ImageColumn("Capa"),
            "Média": st.column_config.NumberColumn(format="%.2f"),
            "Ep.": st.column_config.NumberColumn(),
            "Abrir": st.column_config.LinkColumn("Abrir", display_text="Ver detalhes"),
        }

        st.dataframe(show, column_config=config, use_container_width=True, hide_index=True)

    except Exception:
        st.error("Falha ao buscar.")

def page_detalhe():
    if not ss.backend_ok:
        st.warning("Backend indisponível.")
        return
    if not ss.anime:
        st.info("Escolha um anime para ver os detalhes.")
        return
    try:
        anime_id = int(ss.anime)
        item = api_get("/anime", params={"id": anime_id}, timeout=15)
    except Exception:
        st.error("Anime não encontrado.")
        return

    st.markdown("## Detalhes do Anime")
    colL, colR = st.columns([1,2], gap="large")
    img = image_url_or_placeholder(int(item["anime_id"]))
    with colL:
        st.markdown(f'<div class="detail-cover"><img src="{img}"></div>', unsafe_allow_html=True)
    with colR:
        st.subheader(item.get("name","(sem nome)"))
        st.write(f"**Gêneros:** {item.get('genre','-')}")
        st.write(f"**Tipo:** {item.get('type','-')} | **Episódios:** {item.get('episodes','-')} | **Média:** {item.get('rating','-')}")
        if ss.user:
            rv = st.slider("Sua avaliação (0–10)", min_value=0, max_value=10, value=8, step=1, key=f"rate_{anime_id}")
            if st.button("Salvar avaliação", key=f"rate_btn_{anime_id}"):
                try:
                    api_post("/rate", {"user_id": int(ss.user["user_id"]), "anime_id": int(anime_id), "rating": int(rv)}, timeout=15)
                    st.success("Avaliação salva!")
                except APIError as e:
                    st.error(e.args[0] if e.args else "Falha ao salvar.")
                except Exception:
                    st.error("Não foi possível salvar.")
        else:
            st.info("Entre na sua conta para avaliar.")

    if ss.user:
        st.markdown("#### Pessoas também gostaram de")
        try:
            recs = api_get("/similar", params={"anime_id": int(anime_id), "n": 15}, timeout=20)
            if recs:
                render_grid_5(recs)
            else:
                st.info("Sem itens parecidos suficientes.")
        except Exception:
            st.info("Não foi possível carregar recomendados.")

def page_recs():
    st.markdown("## Recomendações")
    if not ss.backend_ok:
        st.warning("Backend indisponível.")
        return
    if not ss.user:
        st.warning("Entre e avalie pelo menos um anime.")
        return
    try:
        stats = api_get("/user_stats", params={"user_id": int(ss.user["user_id"])}, timeout=10)
        if stats.get("ratings_count", 0) <= 0:
            st.warning("Parece que você ainda não avaliou nada.")
            return
    except Exception:
        st.error("Não foi possível verificar suas avaliações.")
        return

    if not ss.genres_all:
        try:
            ss.genres_all = api_get("/genres", timeout=10)
        except Exception:
            ss.genres_all = []

    # ---------- Controles (com keys únicas) ----------
    c1, c2, c3 = st.columns(3)
    type_filter = c1.selectbox(
        "Tipo", ["(todos)", "TV", "Movie", "OVA", "Special", "Music", "ONA"],
        key="recs_tipo"
    )
    genre_filter = c2.selectbox(
        "Gênero", ["(todos)"] + ss.genres_all,
        key="recs_genero"
    )
    max_ep = c3.slider(
        "Máx. episódios", 1, 2000, 500,
        key="recs_max_ep"
    )

    filters = {"max_episodes": str(max_ep)}
    if type_filter != "(todos)":
        filters["type"] = type_filter
    if genre_filter != "(todos)":
        filters["genre"] = genre_filter

    # ÚNICO slider "Quantidade" (key obrigatória)
    nshow = st.slider("Quantidade", 5, 60, 25, key="recs_nshow")

    if st.button("Recomendar agora", use_container_width=True, key="recs_btn"):
        ss.last_recs = []
        payload = {
            "user_id": int(ss.user["user_id"]),
            "n_recs": int(nshow),
            "filters": filters or None
        }

        try:
            # 1) tenta com todos os filtros escolhidos
            res = api_post("/recommend", payload, timeout=45) or []
            df = pd.DataFrame(res)

            # 2) fallback: se vazio e havia gênero, tenta sem gênero
            if df.empty and "genre" in filters:
                payload2 = {
                    "user_id": payload["user_id"],
                    "n_recs": payload["n_recs"],
                    "filters": {k: v for k, v in filters.items() if k != "genre"} or None
                }
                res = api_post("/recommend", payload2, timeout=45) or []
                df = pd.DataFrame(res)
                if not df.empty:
                    st.info("Nenhum resultado para esse gênero específico. Mostrando recomendações sem o filtro de gênero.")

            # 3) fallback final: se ainda vazio, sem filtros
            if df.empty and filters:
                payload3 = {
                    "user_id": payload["user_id"],
                    "n_recs": payload["n_recs"],
                    "filters": None
                }
                res = api_post("/recommend", payload3, timeout=45) or []
                df = pd.DataFrame(res)
                if not df.empty:
                    st.info("Sem resultados com os filtros atuais. Mostrando recomendações gerais.")

            if not df.empty:
                # já vem ordenado pelo backend
                ss.last_recs = df.to_dict(orient="records")
            else:
                ss.last_recs = []
                st.warning("Não foi possível encontrar recomendações com base nas suas avaliações.")
        except APIError as e:
            st.error(e.args[0] if e.args else "Erro ao recomendar.")
        except Exception:
            st.error("Não foi possível gerar recomendações.")

    # ---------- Render ----------
    recs = ss.last_recs or []
    if recs:
        render_grid_5(recs)
    else:
        st.info("Clique em “Recomendar agora” para ver sugestões baseadas no que você avaliou.")

def page_analise():
    st.markdown("## Análise e Métricas")
    if not ss.backend_ok:
        st.warning("Backend indisponível.")
        return

    st.caption("Acurácia = acertos / itens recomendados (top-K).")

    # --- Estatísticas globais (podem aparecer mesmo sem login) ---
    try:
        s = api_get("/stats", timeout=20)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de animes", s.get("total", 0))
        c2.metric("Tipos distintos", len(s.get("types", {})))
        mr = s.get("mean_rating")
        c3.metric("Média global", f"{mr:.3f}" if mr is not None else "-")

        st.markdown("### Distribuições")
        colA, colB = st.columns(2)
        if s.get("types"):
            colA.bar_chart(pd.Series(s["types"]).sort_values(ascending=False))
        if s.get("genres_top"):
            colB.bar_chart(pd.Series(s["genres_top"]).head(20))
        if s.get("rating_hist"):
            st.markdown("### Histograma de notas")
            st.line_chart(pd.Series(s["rating_hist"]).sort_index())
    except Exception:
        st.info("Não foi possível carregar estatísticas.")

    # --- Configurações e cálculo de acurácia (somente logado) ---
    with st.expander("Calcular acurácia"):
        if not ss.get("user"):
            st.info("Entre na sua conta para calcular a acurácia do seu perfil.")
            _ = st.button("Calcular acurácia", disabled=True)  # botão desabilitado só pra UX
            return

        # usa SEMPRE o usuário logado
        uid = int(ss.user["user_id"])

        # controles (sem escolher usuário/algo)
        topk = st.slider("Top-K", 5, 50, 20, key="analise_topk")
        thr  = st.slider("Limiar relevante (≥)", 1, 10, 6, key="analise_thr")

        if st.button("Calcular acurácia"):
            try:
                res = api_post("/evaluate", {
                    "user_id": uid,           # sempre o usuário logado
                    "test_ratio": 0.2,        # 80/20
                    "top_k": int(topk),
                    "relevant_threshold": int(thr)
                }, timeout=45)

                hits, recs, acc = int(res["hits"]), int(res["recommended"]), float(res["accuracy"])
                k1, k2, k3 = st.columns(3)
                k1.metric("Acertos", hits)
                k2.metric("Recomendações", recs)
                k3.metric("Acurácia", f"{acc:.3f}")

            except APIError as e:
                st.error(e.args[0] if e.args else "Falha ao calcular.")
            except Exception:
                st.error("Não foi possível calcular.")

def page_perfil():
    if not ss.user:
        st.info("Entre para acessar seu perfil.")
        return

    st.markdown("## Meu perfil")
    st.write(f"Usuário: {ss.user['username']}  |  ID: {ss.user['user_id']}")

    # ----- Editar usuário -----
    with st.expander("Editar usuário"):
        new_u = st.text_input(
            "Novo nome de usuário",
            value=ss.user["username"],
            key="perfil_new_username",
        )
        new_p = st.text_input(
            "Nova senha (opcional)",
            type="password",
            key="perfil_new_password",
        )
        if st.button("Salvar alterações", key="perfil_save_btn"):
            try:
                payload = {"user_id": int(ss.user["user_id"])}
                if new_u and new_u != ss.user["username"]:
                    payload["new_username"] = new_u
                if new_p:
                    payload["new_password"] = new_p
                if len(payload.keys()) > 1:
                    data = api_post("/update_user", payload, timeout=20)
                    ss.user["username"] = data["username"]
                    st.success("Perfil atualizado!")
                    st.query_params["uname"] = ss.user["username"]
                else:
                    st.info("Nada para atualizar.")
            except APIError as e:
                st.error(e.args[0] if e.args else "Falha ao atualizar.")
            except Exception:
                st.error("Não foi possível atualizar.")

    # ----- Minhas avaliações -----
    st.markdown("### Minhas avaliações")
    try:
        lst = api_get("/user_ratings", params={"user_id": int(ss.user["user_id"])}, timeout=20)
        if not lst:
            st.info("Você ainda não avaliou nenhum anime.")
        else:
            df = pd.DataFrame(lst).rename(columns={"name": "Título", "rating": "Nota"})
            df = df[["Título", "Nota"]].sort_values(["Nota", "Título"], ascending=[False, True])
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        st.error("Não foi possível carregar suas avaliações.")

    # ----- Ações -----
    st.markdown("### Ações")
    colx, coly = st.columns([1, 2])
    with colx:
        if st.button("Reset minhas avaliações", type="secondary", key="perfil_reset_btn"):
            try:
                out = wipe_user_ratings(int(ss.user["user_id"]))
                removed = out.get("removed", 0)
                st.success(f"Avaliações removidas: {removed}")
                # limpa recs em cache e atualiza a tela
                ss.last_recs = []
                st.rerun()
            except APIError as e:
                st.error(e.args[0] if e.args else "Falha ao resetar.")
            except Exception:
                st.error("Não foi possível resetar suas avaliações agora.")

# ------------------------------------------------------------
# ROUTER
# ------------------------------------------------------------
if ss.tab == "recs":
    page_recs()
elif ss.tab == "analise":
    page_analise()
elif ss.tab == "detalhe":
    page_detalhe()
elif ss.tab == "busca":
    page_busca()
elif ss.tab == "perfil":
    page_perfil()
else:
    page_inicio()
