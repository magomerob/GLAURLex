from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

from urlex.config import DEFAULT_PROCESSED_DIR
from urlex.core.groups import ALL_GROUP
from urlex.core.groups_store import load_groups


@dataclass
class AppState:
    dataset_name: Optional[str] = None


def ensure_state():
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()

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
