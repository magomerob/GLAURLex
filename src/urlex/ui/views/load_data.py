from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from urlex.config import DEFAULT_PROCESSED_DIR
from urlex.core.dataset_service import DatasetService  # ajusta import si tu ruta cambia
from urlex.ui.state import ensure_state


@st.cache_resource
def get_dataset_service(processed_dir: str) -> DatasetService:
    return DatasetService(processed_dir)


def render_load_data():
    s = ensure_state()

    st.header("1) Carga de datos")

    processed_dir = st.text_input(
        "Directorio de datasets procesados",
        value=str(DEFAULT_PROCESSED_DIR),
        help="Aquí se guardan (y se buscan) los directorios con parquets.",
    )
    service = get_dataset_service(processed_dir)

    col1, col2 = st.columns(2, gap="large")

    # -----------------------
    # A) Subir XLSX y procesar
    # -----------------------
    with col1:
        st.subheader("A) Subir XLSX y procesar")

        uploaded = st.file_uploader(
            "Sube un archivo .xlsx",
            type=["xlsx"],
            accept_multiple_files=False,
        )

        dataset_name = st.text_input(
            "Nombre del dataset (opcional)",
            value="",
            help="Si lo dejas vacío, se usa el nombre del fichero.",
        )

        overwrite = st.checkbox("Sobrescribir si ya existe", value=False)

        if st.button("Procesar XLSX", disabled=(uploaded is None)):
            # Usamos un tmp real en el sistema
            # Streamlit no da un tmp global directamente, así que creamos uno en processed_dir/.tmp_uploads
            tmp_base = Path(processed_dir) / ".tmp_uploads"
            tmp_base.mkdir(parents=True, exist_ok=True)

            xlsx_path = tmp_base / (uploaded.name or "uploaded.xlsx")
            xlsx_path.write_bytes(uploaded.getvalue())

            try:
                name = service.process_xlsx(
                    xlsx_path=xlsx_path,
                    dataset_name=(dataset_name.strip() or None),
                    overwrite=overwrite,
                )
                s.dataset_name = name
                st.success(f"Procesado OK. Dataset activo: **{name}**")
                st.rerun()

            except Exception as e:
                st.error(f"Error procesando XLSX: {e}")

            shutil.rmtree(tmp_base)
    # -----------------------
    # B) Cargar ya procesado
    # -----------------------
    with col2:
        st.subheader("B) Elegir dataset ya procesado")

        names = service.list_processed()
        if not names:
            st.info("No hay datasets procesados aún en ese directorio.")
        else:
            choice = st.selectbox("Datasets disponibles", options=names, index=0)
            if st.button("Usar este dataset"):
                s.dataset_name = choice
                st.success(f"Dataset activo: **{choice}**")
                st.rerun()
    st.divider()

    # Estado actual
    st.subheader("Estado")
    if s.dataset_name:
        st.write(f"Dataset activo: **{s.dataset_name}**")
        temas = service.list_temas(s.dataset_name)
        st.write(f"Temas detectados: **{len(temas)}**")
        with st.expander("Ver temas"):
            st.write(temas)

        if st.button("Descargar dataset (desactivar)"):
            s.dataset_name = None
            st.info("Dataset desactivado. Visualización y Grafos quedan bloqueados.")
            st.rerun()
    else:
        st.write("No hay dataset activo.")
