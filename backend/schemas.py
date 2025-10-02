from pydantic import BaseModel, Field, validator
from typing import Optional, Dict

# ---------- Inputs de autenticação / usuário ----------
class SignupIn(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)

class LoginIn(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)

class UpdateUserIn(BaseModel):
    user_id: int
    new_username: Optional[str] = None
    new_password: Optional[str] = None

class WipeIn(BaseModel):
    user_id: int

# ---------- Rating ----------
class RatingIn(BaseModel):
    user_id: int
    anime_id: int
    rating: int = Field(ge=0, le=10)


# ---------- Recomendação ----------
class RecommendQuery(BaseModel):
    user_id: int
    algo: str = "item"                 # "item" | "user"
    sim_metric: str = "cosine"         # reservado p/ futuras variações
    n_neighbors: int = 20              # usado no user-based
    n_recs: int = 20                   # alinhado ao main/front
    filters: Optional[Dict[str, str]] = None

    @validator("algo")
    def _check_algo(cls, v):
        v = (v or "").lower()
        if v not in {"item", "user"}:
            raise ValueError("algo deve ser 'item' ou 'user'")
        return v

    @validator("n_neighbors")
    def _check_nbrs(cls, v):
        if v <= 0:
            raise ValueError("n_neighbors deve ser > 0")
        return v

    @validator("n_recs")
    def _check_nrecs(cls, v):
        if v <= 0:
            raise ValueError("n_recs deve ser > 0")
        return v


# ---------- Avaliação ----------
class EvaluateQuery(BaseModel):
    user_id: int
    algo: str = "item"                 # "item" | "user"
    sim_metric: str = "cosine"
    test_ratio: float = 0.2            # 80/20
    top_k: int = 10
    relevant_threshold: float = 7.0

    @validator("algo")
    def _check_algo_eval(cls, v):
        v = (v or "").lower()
        if v not in {"item", "user"}:
            raise ValueError("algo deve ser 'item' ou 'user'")
        return v

    @validator("test_ratio")
    def _check_ratio(cls, v):
        if not (0.05 <= v <= 0.5):
            raise ValueError("test_ratio deve estar entre 0.05 e 0.5")
        return v

    @validator("top_k")
    def _check_topk(cls, v):
        if v <= 0:
            raise ValueError("top_k deve ser > 0")
        return v

    @validator("relevant_threshold")
    def _check_threshold(cls, v):
        if not (0 <= v <= 10):
            raise ValueError("relevant_threshold deve estar entre 0 e 10")
        return v
