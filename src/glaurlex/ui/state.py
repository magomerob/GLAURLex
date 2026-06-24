from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, TypeVar

import streamlit as st

from glaurlex.config import (
    DATA_ROOT,
    DEFAULT_PROCESSED_DIR,
    REMOTE_USER_HEADER,
    REQUIRE_AUTH,
    user_processed_dir,
)
from glaurlex.core.groups import ALL_GROUP
from glaurlex.core.groups_store import load_groups
from glaurlex.core.variables_store import load_variables

T = TypeVar("T")

_SAFE_USERNAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class AppState:
    dataset_name: Optional[str] = None
    username: Optional[str] = None


def _sanitize_username(raw: str) -> Optional[str]:
    """! Normaliza el header de usuario en un nombre seguro para rutas.

    Reemplaza cualquier carácter fuera de `[A-Za-z0-9._-]` por `_` y
    descarta valores vacíos o reservados (`.`, `..`).
    """
    if not raw:
        return None
    cleaned = _SAFE_USERNAME_RE.sub("_", raw.strip()).strip("._-")
    if not cleaned or cleaned in {".", ".."}:
        return None
    return cleaned[:64]


def _remote_username() -> Optional[str]:
    """! Lee el usuario autenticado de la cabecera inyectada por el proxy.

    Devuelve `None` si la cabecera no está presente (p. ej. acceso local
    sin Authelia).
    """
    try:
        headers = st.context.headers  # type: ignore[attr-defined]
    except Exception:
        return None
    if not headers:
        return None
    raw = headers.get(REMOTE_USER_HEADER) or headers.get(REMOTE_USER_HEADER.lower())
    return _sanitize_username(raw) if raw else None


def get_query_param(name: str) -> Optional[str]:
    value = st.query_params.get(name)
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return str(value[0])
    return str(value)


def set_query_param(name: str, value) -> None:
    if value is None or value == "":
        if name in st.query_params:
            del st.query_params[name]
        return
    st.query_params[name] = str(value)


def sync_query_state(
    *,
    key: str,
    param: str,
    default: T,
    parse: Callable[[str], T] | None = None,
    serialize: Callable[[T], str] | None = None,
    normalize: Callable[[T], T] | None = None,
    allowed_values: Iterable[T] | None = None,
    drop_if_default: bool = False,
) -> T:
    if key not in st.session_state:
        raw = get_query_param(param)
        if raw is None:
            value = default
        else:
            try:
                value = parse(raw) if parse is not None else raw
            except (TypeError, ValueError):
                value = default
        st.session_state[key] = value

    value = st.session_state[key]
    if normalize is not None:
        value = normalize(value)

    if allowed_values is not None:
        allowed = list(allowed_values)
        if value not in allowed:
            value = default if default in allowed else allowed[0]

    st.session_state[key] = value

    if value is None or (drop_if_default and value == default):
        if param in st.query_params:
            del st.query_params[param]
    else:
        st.query_params[param] = serialize(value) if serialize is not None else str(value)

    return value


def ensure_state():
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()
        qp_dataset = get_query_param("dataset")
        if qp_dataset:
            st.session_state["app_state"].dataset_name = qp_dataset

    if "groups" not in st.session_state:
        st.session_state.groups = {"TODOS": ALL_GROUP}

    if "active_group" not in st.session_state:
        st.session_state.active_group = "TODOS"

    if "dataset_loaded" not in st.session_state:
        st.session_state.dataset_loaded = False

    _apply_user_scope(st.session_state["app_state"])

    return st.session_state["app_state"]


def _apply_user_scope(app_state: "AppState") -> None:
    """! Fija `processed_dir` al sandbox del usuario o detiene la página si
    `REQUIRE_AUTH` está activo y no hay cabecera de autenticación."""
    username = _remote_username()

    if username:
        app_state.username = username
        scoped = str(user_processed_dir(username))
        # Pin: las vistas leen primero esta clave y, en modo multiusuario,
        # ignoramos cualquier override que venga de la URL/UI.
        st.session_state["processed_dir"] = scoped
        st.session_state["DatasetService::processed_dir"] = scoped
        st.session_state["load_data::processed_dir"] = scoped
        st.session_state["multi_tenant"] = True
        user_processed_dir(username).mkdir(parents=True, exist_ok=True)
        return

    if REQUIRE_AUTH:
        st.error(
            "Acceso denegado: no se ha recibido la cabecera de autenticación."
            " Inicia sesión a través del proxy."
        )
        st.stop()

    st.session_state.setdefault("multi_tenant", False)
    st.session_state.setdefault("processed_dir", str(DEFAULT_PROCESSED_DIR))


def has_dataset_loaded() -> bool:
    s: AppState = st.session_state.get("app_state")
    return bool(s and s.dataset_name)


def current_username() -> Optional[str]:
    s: AppState = st.session_state.get("app_state")
    return s.username if s else None


def is_multi_tenant() -> bool:
    return bool(st.session_state.get("multi_tenant"))


__all__ = [
    "AppState",
    "DATA_ROOT",
    "current_username",
    "ensure_groups_loaded_for_dataset",
    "ensure_state",
    "ensure_variables_loaded_for_dataset",
    "get_query_param",
    "has_dataset_loaded",
    "is_multi_tenant",
    "set_query_param",
    "sync_query_state",
]


def ensure_groups_loaded_for_dataset(dataset_name: str) -> None:
    processed_dir = st.session_state.get("DatasetService::processed_dir", DEFAULT_PROCESSED_DIR)
    processed_dir = st.session_state.get("processed_dir", processed_dir)

    cache_key = f"groups::loaded_for::{processed_dir}::{dataset_name}"
    if st.session_state.get("groups::loaded_key") == cache_key:
        return

    st.session_state.groups = load_groups(processed_dir, dataset_name)
    st.session_state.active_group = "TODOS"
    st.session_state["groups::loaded_key"] = cache_key

    if "TODOS" not in st.session_state.groups:
        st.session_state.groups["TODOS"] = ALL_GROUP


def ensure_variables_loaded_for_dataset(dataset_name: str) -> None:
    """! Carga la configuración de variables (ordinales, etc.) del dataset activo.

    Mirror del patrón de `ensure_groups_loaded_for_dataset`. Almacena el dict
    en `st.session_state["variables"]`.
    """
    processed_dir = st.session_state.get("DatasetService::processed_dir", DEFAULT_PROCESSED_DIR)
    processed_dir = st.session_state.get("processed_dir", processed_dir)

    cache_key = f"variables::loaded_for::{processed_dir}::{dataset_name}"
    if st.session_state.get("variables::loaded_key") == cache_key:
        return

    st.session_state["variables"] = load_variables(processed_dir, dataset_name)
    st.session_state["variables::loaded_key"] = cache_key
