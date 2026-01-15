from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from urlex.core import dataset_service as ds
from urlex.ui.state import set_load_result

DATA_PROCESSED_ROOT = os.getenv("DATA_PROCESSED_ROOT", ".data/processed")

data_processsed_path = Path(DATA_PROCESSED_ROOT)


def render() -> None:
    st.header("1) Carga de datos")

    left, right = st.columns(2, gap="large")

    # -------------------------
    # Subir XLSX
    # -------------------------
    with left:
        st.subheader("Subir archivo (XLSX)")
        uploaded = st.file_uploader("Sube un .xlsx", type=["xlsx"], accept_multiple_files=False)

        if uploaded is not None:
            file_bytes = uploaded.getvalue()

            # Intentamos listar hojas (si el core lo implementa)
            sheet_choice = None
            try:
                sheets = ds.list_xlsx_sheets(file_bytes)
                if sheets:
                    sheet_choice = st.selectbox("Hoja a cargar", sheets, index=0)
            except NotImplementedError:
                st.caption(
                    "ℹ️ (list_xlsx_sheets no implementado: se usará la política por defecto del core)"
                )

            try:
                lr = ds.load_from_upload_xlsx(uploaded.name, file_bytes, sheet=sheet_choice)
                set_load_result(lr)
                st.success(f"Cargado: {lr.ref.name}")
                st.dataframe(lr.df.head(30), use_container_width=True)

            except NotImplementedError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Error cargando/procesando XLSX: {e}")

    # -------------------------
    # Elegir dataset procesado (carpeta)
    # -------------------------
    with right:
        st.subheader("Elegir datos ya procesados (carpeta)")

        try:
            processed = ds.list_processed_datasets(data_processsed_path)
        except NotImplementedError as e:
            st.warning(str(e))
            processed = []

        if not processed:
            st.info(
                f"No hay datasets procesados listados (o no está implementado). Root: `{data_processsed_path.as_posix()}`"
            )
            return

        options = ["— Selecciona —"] + [p.name for p in processed]
        choice = st.selectbox("Dataset procesado", options, index=0)

        if choice != "— Selecciona —":
            selected = next(p for p in processed if p.name == choice)

            # Mostrar inventario de ficheros (útil para debug)
            with st.expander("Ver ficheros dentro"):
                st.write([f.as_posix() for f in selected.files])

            # Cargar dataset principal
            try:
                lr = ds.load_from_processed_dir(selected.path)
                set_load_result(lr)
                st.success(f"Cargado: {lr.ref.name}")
                st.dataframe(lr.df.head(30), use_container_width=True)

                # Mostrar artefactos (stats/grafos) si está implementado
                try:
                    artifacts = ds.get_processed_artifacts(selected.path)
                    with st.expander("Artefactos detectados"):
                        st.write({k: v.as_posix() for k, v in artifacts.items()})
                except NotImplementedError:
                    st.caption("ℹ️ (get_processed_artifacts no implementado)")

            except NotImplementedError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Error cargando dataset procesado: {e}")
