from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

import streamlit as st

from glaurlex.config import DEFAULT_PROCESSED_DIR
from glaurlex.core.dataset_service import DatasetService
from glaurlex.ui.state import ensure_state, set_query_param, sync_query_state


@st.cache_resource
def get_dataset_service(processed_dir: str) -> DatasetService:
    return DatasetService(processed_dir)


@st.cache_data(show_spinner=False)
def _dataset_summary(processed_dir: str, name: str) -> Dict[str, int]:
    ds = get_dataset_service(processed_dir).load_processed(name)
    n_informantes = int(len(ds.informantes))
    n_temas = int(len(ds.temas))
    n_tokens = int(sum(len(df) for df in ds.temas.values()))
    types_set: set = set()
    for df in ds.temas.values():
        types_set.update(df["type"].dropna().astype(str).unique().tolist())
    n_types = int(len(types_set))
    return {
        "informantes": n_informantes,
        "temas": n_temas,
        "types": n_types,
        "tokens": n_tokens,
    }


def _parse_salamanca_stimulus_map(raw_text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for i, line in enumerate(raw_text.splitlines(), start=1):
        row = line.strip()
        if not row:
            continue
        if "=" not in row:
            raise ValueError(
                f"Línea {i}: formato inválido. Usa `codigo=estimulo`, por ejemplo `1=CASA`."
            )
        code, stimulus = row.split("=", 1)
        code = code.strip()
        stimulus = stimulus.strip()
        if not code:
            raise ValueError(f"Línea {i}: el código está vacío.")
        if not stimulus:
            raise ValueError(f"Línea {i}: el estímulo está vacío.")
        mapping[code] = stimulus
    return mapping


def render_load_data():
    s = ensure_state()

    st.header("Carga de datos")

    sync_query_state(
        key="load_data::processed_dir",
        param="processed_dir",
        default=str(DEFAULT_PROCESSED_DIR),
    )
    processed_dir = st.text_input(
        "Directorio de datasets procesados",
        key="load_data::processed_dir",
        help="Aquí se guardan (y se buscan) los directorios con parquets.",
    )
    st.session_state["processed_dir"] = processed_dir
    service = get_dataset_service(processed_dir)

    col1, col2 = st.columns(2, gap="large")

    # -----------------------
    # A) Subir archivo y procesar
    # -----------------------
    with col1:
        st.subheader("A) Subir archivo y procesar")

        input_format = st.selectbox(
            "Formato de entrada",
            options=["xlsx", "salamanca"],
            index=0,
            help="Selecciona el formato del archivo que vas a subir.",
        )

        if input_format == "xlsx":
            upload_label = "Sube un archivo .xlsx"
            upload_types = ["xlsx"]
            process_button_label = "Procesar XLSX"
            default_uploaded_name = "uploaded.xlsx"
        else:
            upload_label = "Sube un archivo .txt (Salamanca)"
            upload_types = ["txt"]
            process_button_label = "Procesar Salamanca"
            default_uploaded_name = "uploaded.txt"

        uploaded = st.file_uploader(
            upload_label,
            type=upload_types,
            accept_multiple_files=False,
        )

        dataset_name = st.text_input(
            "Nombre del dataset (opcional)",
            value="",
            help="Si lo dejas vacío, se usa el nombre del fichero.",
        )

        overwrite = st.checkbox("Sobrescribir si ya existe", value=False)
        salamanca_map: Dict[str, str] | None = None

        if input_format == "salamanca":
            with st.expander("Diccionario código -> estímulo (opcional)"):
                st.caption("Una línea por par con formato `codigo=estimulo`.")
                raw_map_text = st.text_area(
                    "Correspondencias",
                    value="",
                    placeholder="1=CASA\n2=PERRO\n3=AGUA",
                    height=140,
                )
                if raw_map_text.strip():
                    try:
                        salamanca_map = _parse_salamanca_stimulus_map(raw_map_text)
                        st.caption(f"Entradas válidas: {len(salamanca_map)}")
                    except ValueError as parse_error:
                        st.error(str(parse_error))
                        salamanca_map = None

        if st.button(process_button_label, disabled=(uploaded is None)):
            # Usamos un tmp real en el sistema
            # Streamlit no da un tmp global directamente, así que creamos uno en processed_dir/.tmp_uploads
            tmp_base = Path(processed_dir) / ".tmp_uploads"
            tmp_base.mkdir(parents=True, exist_ok=True)

            input_path = tmp_base / (uploaded.name or default_uploaded_name)
            input_path.write_bytes(uploaded.getvalue())

            try:
                if input_format == "xlsx":
                    name = service.process_xlsx(
                        xlsx_path=input_path,
                        dataset_name=(dataset_name.strip() or None),
                        overwrite=overwrite,
                    )
                else:
                    if raw_map_text.strip() and salamanca_map is None:
                        raise ValueError(
                            "El diccionario de correspondencias tiene errores de formato."
                        )
                    name = service.process_salamanca(
                        txt_path=input_path,
                        dataset_name=(dataset_name.strip() or None),
                        overwrite=overwrite,
                        stimulus_map=salamanca_map,
                    )

                s.dataset_name = name
                set_query_param("dataset", name)
                st.success(f"Procesado OK. Dataset activo: **{name}**")
                st.rerun()

            except Exception as e:
                st.error(f"Error procesando {input_format}: {e}")

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
                set_query_param("dataset", choice)
                st.success(f"Dataset activo: **{choice}**")
                st.rerun()
    st.divider()

    # Estado actual
    st.subheader("Estado")
    if s.dataset_name:
        st.write(f"Dataset activo: **{s.dataset_name}**")

        try:
            summary = _dataset_summary(processed_dir, s.dataset_name)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Informantes", f"{summary['informantes']}")
            m2.metric("Temas", f"{summary['temas']}")
            m3.metric("Types únicos", f"{summary['types']}")
            m4.metric("Tokens totales", f"{summary['tokens']}")
        except Exception as e:
            st.warning(f"No se pudieron calcular las estadísticas del dataset: {e}")

        temas = service.list_temas(s.dataset_name)
        with st.expander("Ver temas"):
            st.write(temas)

        if st.button("Descargar dataset (desactivar)"):
            s.dataset_name = None
            set_query_param("dataset", None)
            st.info("Dataset desactivado. Visualización y Grafos quedan bloqueados.")
            st.rerun()
    else:
        st.write("No hay dataset activo.")
