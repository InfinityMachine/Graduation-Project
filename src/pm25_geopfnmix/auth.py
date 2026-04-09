from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from .settings import ROOT

_TABPFN_CACHE_FILES = (
    Path.home() / ".cache" / "tabpfn" / "auth_token",
    Path.home() / ".tabpfn" / "token",
)


@lru_cache(maxsize=1)
def _load_local_token_module() -> ModuleType | None:
    token_path = ROOT / "token.py"
    if not token_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("pm25_geopfnmix_local_token", token_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        return token

    module = _load_local_token_module()
    if module is None:
        return None

    token = getattr(module, "HF_TOKEN", "")
    if isinstance(token, str) and token.strip():
        token = token.strip()
        os.environ.setdefault("HF_TOKEN", token)
        return token
    return None


def resolve_tabpfn_token() -> str | None:
    token = os.environ.get("TABPFN_TOKEN", "").strip()
    if token:
        os.environ.setdefault("TABPFN_TOKEN", token)
        return token

    module = _load_local_token_module()
    if module is not None:
        for attr_name in ("TABPFN_TOKEN", "PRIORLABS_TOKEN"):
            token = getattr(module, attr_name, "")
            if isinstance(token, str) and token.strip():
                token = token.strip()
                os.environ.setdefault("TABPFN_TOKEN", token)
                return token

    for cache_path in _TABPFN_CACHE_FILES:
        if cache_path.exists():
            token = cache_path.read_text(encoding="utf-8").strip()
            if token:
                os.environ.setdefault("TABPFN_TOKEN", token)
                return token
    return None


def prime_auth_tokens() -> None:
    resolve_hf_token()
    resolve_tabpfn_token()


def has_hf_token() -> bool:
    return resolve_hf_token() is not None


def has_tabpfn_token() -> bool:
    return resolve_tabpfn_token() is not None


def has_tabpfn_headless_access() -> bool:
    prime_auth_tokens()
    return has_tabpfn_token()


def describe_tabpfn_access_state() -> str:
    if has_tabpfn_token():
        return "tabpfn_token_available"
    if has_hf_token():
        return "hf_token_only"
    return "no_token"
