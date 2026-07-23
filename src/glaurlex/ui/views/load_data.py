from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

import streamlit as st

from glaurlex.config import DEFAULT_PROCESSED_DIR
from glaurlex.core.dataset_service import DatasetService
from glaurlex.core.example_data import (
    EXAMPLE_SALAMANCA_FILENAME,
    EXAMPLE_XLSX_FILENAME,
    build_example_salamanca,
    build_example_xlsx,
    salamanca_stimulus_map,
)
from glaurlex.ui.state import (
    current_username,
    ensure_state,
    set_query_param,
)


@st.cache_resource
def get_dataset_service(processed_dir: str) -> DatasetService:
    return DatasetService(processed_dir)


@st.cache_data(show_spinner=False)
def _example_xlsx_bytes() -> bytes:
    return build_example_xlsx()


@st.cache_data(show_spinner=False)
def _example_salamanca_bytes() -> bytes:
    return build_example_salamanca().encode("utf-8")


def _render_example_downloads() -> None:
    """! Muestra los datasets de ejemplo descargables (XLSX y Salamanca TXT)."""
    with st.expander("¿No tienes datos? Descarga un ejemplo"):
        st.caption(
            "Dos datasets de ejemplo con la misma estructura que espera la app. "
            "Descárgalos, ábrelos como plantilla y súbelos en el formato "
            "correspondiente."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Ejemplo XLSX",
                data=_example_xlsx_bytes(),
                file_name=EXAMPLE_XLSX_FILENAME,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.caption("Hojas `Informantes`, `Variables` y un tema por hoja.")
        with c2:
            st.download_button(
                "Ejemplo Salamanca (.txt)",
                data=_example_salamanca_bytes(),
                file_name=EXAMPLE_SALAMANCA_FILENAME,
                mime="text/plain",
                use_container_width=True,
            )
            st.caption("Formato `EXPID INDIV TEMA palabra1, palabra2, ...`.")

        map_hint = "\n".join(
            f"{code}={stimulus}" for code, stimulus in salamanca_stimulus_map().items()
        )
        st.caption(
            "Para el ejemplo Salamanca, puedes usar este diccionario "
            "código → estímulo al procesar y así nombrar los temas:"
        )
        st.code(map_hint, language="text")


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

    # El directorio de datos viene de la config (env `GLAURLEX_DATA_DIR`) y,
    # en modo multiusuario, se fija al sandbox del usuario autenticado. La UI
    # solo muestra dónde está; no permite cambiarlo.
    processed_dir = st.session_state.get("processed_dir", str(DEFAULT_PROCESSED_DIR))
    st.session_state["processed_dir"] = processed_dir
    user = current_username()
    if user:
        st.caption(f"Usuario: **{user}** — sandbox: `{processed_dir}`")
    else:
        st.caption(f"Directorio de datos: `{processed_dir}`")
    service = get_dataset_service(processed_dir)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("A) Subir archivo y procesar")

        _render_example_downloads()

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

            with st.expander("Eliminar dataset"):
                st.warning(
                    f"Se eliminará **{choice}** del disco de forma permanente. "
                    "Esta acción no se puede deshacer."
                )
                # La `key` incluye el nombre elegido: al cambiar de dataset en el
                # selectbox, la confirmación se reinicia automáticamente.
                confirm_delete = st.checkbox(
                    "Sí, quiero eliminar este dataset",
                    key=f"confirm_delete::{choice}",
                )
                if st.button("Eliminar definitivamente", disabled=not confirm_delete):
                    try:
                        service.delete_processed(choice)
                        # Invalida el resumen cacheado del dataset borrado.
                        _dataset_summary.clear()
                        if s.dataset_name == choice:
                            s.dataset_name = None
                            set_query_param("dataset", None)
                        st.success(f"Dataset «{choice}» eliminado.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"No se pudo eliminar «{choice}»: {e}")
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

        if st.button(
            "Desactivar dataset",
            help="Quita el dataset de la sesión actual. No borra los archivos del disco.",
        ):
            s.dataset_name = None
            set_query_param("dataset", None)
            st.info("Dataset desactivado. El resto de secciones quedan bloqueadas.")
            st.rerun()
    else:
        st.write("No hay dataset activo.")
