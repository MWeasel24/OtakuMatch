from __future__ import annotations
import os
import requests
from functools import lru_cache
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_GET_TIMEOUT = int(os.environ.get("API_GET_TIMEOUT", "15"))
DEFAULT_POST_TIMEOUT = int(os.environ.get("API_POST_TIMEOUT", "20"))

_session = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=0.2,
    status_forcelist=(502, 503, 504),
    allowed_methods=("GET", "POST"),
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

class APIError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message); self.status = status

def _full_url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return urljoin(API_BASE + "/", path.lstrip("/"))

def _handle_resp(r: requests.Response):
    try:
        r.raise_for_status()
        return r.json() if r.content else None
    except requests.HTTPError:
        status = r.status_code
        try:
            body = r.json()
        except Exception:
            body = r.text
        if status in (401, 403): raise APIError("Usuário ou senha inválidos.", status)
        if status == 404:       raise APIError("Recurso não encontrado.", status)
        raise APIError(f"Erro {status}: {str(body)[:200]}", status)
    except requests.RequestException as e:
        raise APIError(f"Falha de conexão com o backend: {e.__class__.__name__}", None)

def api_get(path: str, params=None, timeout: int = DEFAULT_GET_TIMEOUT):
    url = _full_url(path)
    r = _session.get(url, params=params, timeout=timeout); return _handle_resp(r)

def api_post(path: str, json_data=None, timeout: int = DEFAULT_POST_TIMEOUT):
    url = _full_url(path)
    r = _session.post(url, json=json_data, timeout=timeout); return _handle_resp(r)

@lru_cache(maxsize=8192)
def backend_image_url(mal_id: int) -> str:
    try:
        data = api_get(f"/image/{int(mal_id)}", timeout=8)
        url = (data or {}).get("url") or "/images/placeholder.jpg"
        return url if url.startswith("http") else _full_url(url)
    except Exception:
        return _full_url("/images/placeholder.jpg")

def image_url_or_placeholder(mal_id: int) -> str:
    return backend_image_url(int(mal_id))

def wipe_user_ratings(user_id: int) -> dict:
    return api_post("/wipe_user_ratings", {"user_id": int(user_id)})
