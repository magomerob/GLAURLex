# src/urlex/ui/pages/visualize.py
from __future__ import annotations

import streamlit as st

from urlex.core import dataset_service as ds
from urlex.ui.state import get_load_result


def render() -> None:
    st.header("2) Visualización de datos")

    lr = get_load_result()
    if lr is None:
        st.warning("No hay datos cargados. Ve a **1) Carga de datos**.")
        return

    # Contexto
    st.write(f"**Dataset activo:** `{lr.ref.name}`  \n**Origen:** `{lr.ref.kind}`")
    if lr.ref.path is not None:
        st.caption(f"Ruta: `{lr.ref.path.as_posix()}`")

    df = lr.df

    # Controles básicos (UI-only)
    c1, c2, c3 = st.columns([1, 2, 2], gap="large")
    with c1:
        n_rows = st.number_input("Filas a mostrar", min_value=5, max_value=500, value=50, step=5)
    with c2:
        selected_cols = st.multiselect(
            "Columnas",
            df.columns.tolist(),
            default=df.columns.tolist()[: min(8, df.shape[1])],
        )
    with c3:
        query = st.text_input(
            "Filtro (pandas query) — opcional",
            placeholder="ej: colA > 10 and colB == 'X'",
        )

    view = df
    if selected_cols:
        view = view[selected_cols]

    if query.strip():
        try:
            view = view.query(query)
        except Exception as e:
            st.error(f"Query inválida: {e}")

    # Tabla
    st.subheader("Tabla")
    st.dataframe(view.head(int(n_rows)), use_container_width=True)

    # Resumen (placeholder de core)
    st.subheader("Resumen (placeholder core)")

    try:
        summary = ds.summarize_dataframe(df)  # <- placeholder en core/dataset_service.py
        st.json(summary)
    except NotImplementedError as e:
        st.info("Resumen no disponible aún (placeholder).")
        st.caption(str(e))
    except Exception as e:
        st.error(f"Error generando resumen: {e}")

    # Artefactos del dataset procesado (placeholder core)
    if lr.ref.kind == "processed" and lr.ref.path is not None:
        st.subheader("Artefactos del dataset procesado (placeholder core)")
        try:
            artifacts = ds.get_processed_artifacts(lr.ref.path)  # <- placeholder
            if not artifacts:
                st.info("No se detectaron artefactos (o el core devolvió vacío).")
            else:
                # mostramos en formato simple (clave -> ruta)
                st.write({k: v.as_posix() for k, v in artifacts.items()})
        except NotImplementedError as e:
            st.info("Detección de artefactos no disponible aún (placeholder).")
            st.caption(str(e))
        except Exception as e:
            st.error(f"Error leyendo artefactos: {e}")

    # Lugar para futuras visualizaciones
    with st.expander("Visualizaciones (placeholder)"):
        st.write("Aquí añadiremos gráficas, estadísticas y exploración interactiva.")
        st.button("Generar plot (no hace nada aún)", disabled=True)
