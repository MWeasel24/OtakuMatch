# OtakuMatch – Sistema de Recomendação com Filtragem Colaborativa

Equipe: Daniel Nazário, Matheus Martins

Tecnologias: Streamlit (frontend), FastAPI (backend)

Dataset (Renomear os datasets para Anime.csv e Ratings.cvs e colocar na pasta data):

https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database

# OBJETIVO DO SISTEMA
Construir um recomendador funcional com filtragem colaborativa (item-based), arquitetura frontend/backend separadas e avaliação de acurácia.

O sistema permite:
- Registro/login e avaliação de animes (0–10)
- Recomendações personalizadas por similaridade (com filtros por tipo, gênero e máx. de episódios)
- Visualização em UI clara (cards, busca, detalhes)
- Cálculo de acurácia (Precisão@K) com split 80/20

# CENÁRIO DE USO
Recomendações de animes.

Usuários avaliam animes que já assistiram; a partir das preferências coletivas, o sistema sugere novos títulos semelhantes ao gosto do usuário.

Justificativa: catálogos grandes tornam difícil a descoberta; recomendações colaborativas reduzem o esforço e aumentam a satisfação ao indicar títulos relevantes.

# COMO RODAR
1. Criar um ambiente virtual (Opcional)

- python -m venv .venv && source .venv/bin/activate  # Linux/Mac
- python -m venv .venv && .venv\Scripts\activate     # Windows

2. Instalar dependências
- pip install -r requirements.txt

3. Baixar capas (Opcional)
- python tools/fetch_images.py --csv data/Anime.csv --out data/images --workers 4

4. Backend (FastAPI)
- uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

5. Frontend (Streamlit)
- streamlit run frontend/streamlit_app.py

# ALGORITMO DE RECOMENDAÇÃO
Filtragem colaborativa baseada em itens (item-based) com similaridade cosseno em matriz esparsa:

- Calcula itens “parecidos” a partir dos vetores de avaliações dos usuários.
- Justificativa (cosseno): robusto para dados esparsos, independe de escalas individuais de nota, simples e eficiente.

Observação: quando há poucos vizinhos válidos, aplicamos “backfill” por popularidade para garantir lista Top-K.

# ANÁLISE
Acurácia = (nº de acertos) / (nº de itens recomendados = K)

Procedimento (por usuário):
- Dividimos suas avaliações em 80% treino e 20% teste (estratificado para conter positivos).
- Geramos recomendações usando apenas o treino.
- No teste, itens são relevantes se nota ≥ limiar (ex.: 6 ou 7).
- Acertos = itens recomendados que também aparecem como relevantes no teste.
- Acurácia = acertos / K (exibida no app em decimal).

A tela Análise permite ajustar Top-K e limiar e mostra acertos / recomendações / acurácia.

Observação: O objetivo não é focar em um único gênero, e sim em itens que o usuário provavelmente vai gostar com base em seu histórico.

# RESULTADO (Com 30 animes avaliados)

- Usuário: Action
- Senha: 123

| Top-K | Limiar (≥)  | Acertos | Recomendações  | Acurácia    |
|------:|------------:|--------:|---------------:|------------:|
| 20    | 6           | 2       | 20             | 0.100 (10%) |
| 20    | 7           | 3       | 20             | 0.150 (15%) |
| 20    | 6           | 4       | 20             | 0.200 (20%) |
