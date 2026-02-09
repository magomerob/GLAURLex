from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, TypeVar

import streamlit as st

from urlex.config import DEFAULT_PROCESSED_DIR
from urlex.core.groups import ALL_GROUP
from urlex.core.groups_store import load_groups

T = TypeVar("T")


@dataclass
class AppState:
    dataset_name: Optional[str] = None


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

    return st.session_state["app_state"]


def has_dataset_loaded() -> bool:
    s: AppState = st.session_state.get("app_state")
    return bool(s and s.dataset_name)


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
