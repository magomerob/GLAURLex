"""! @package urlex.core.xlsx_processing
Procesamiento de archivos XLSX a parquets por tema.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _sanitize_filename(name: str) -> str:
    """! Normaliza un nombre para usarlo como archivo.

    @param name Nombre original.
    @return Nombre seguro para rutas.
    """
    # evita problemas en Linux/Windows y rutas
    name = str(name).strip()
    name = re.sub(r"[^\w\- ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "tema"


def _coerce_code(x):
    """! Intenta normalizar códigos numéricos de Excel.

    Convierte 1, 1.0, "1", " 1 " a int cuando sea posible.
    Si no se puede, devuelve el valor original.

    @param x Valor de entrada.
    @return Valor normalizado o el original.
    """
    if pd.isna(x):
        return x
    if isinstance(x, (int,)):
        return x
    if isinstance(x, float):
        # 1.0 -> 1 (si es entero)
        if x.is_integer():
            return int(x)
        return x
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return x
        # '1.0' o '1'
        try:
            f = float(s)
            if f.is_integer():
                return int(f)
            return x
        except Exception:
            return x
    return x


def pdprocessxlsx(path, respath):
    """! Procesa un XLSX y escribe parquets en el directorio de salida.

    @param path Ruta al XLSX.
    @param respath Directorio de salida para parquets.
    @exception ValueError Si faltan hojas requeridas.
    """
    path = Path(path)
    respath = Path(respath)
    respath.mkdir(parents=True, exist_ok=True)

    xl = pd.read_excel(path, sheet_name=None)

    required = {"Informantes", "Variables"}
    missing = required - set(xl.keys())
    if missing:
        raise ValueError(f"Faltan hojas requeridas: {sorted(missing)}")

    informantes = xl["Informantes"].copy()
    variables = xl["Variables"].copy()

    informantes.columns = [str(c).strip() for c in informantes.columns]
    variables.columns = [str(c).strip() for c in variables.columns]

    informantescompletos = informantes.copy()

    cols = list(informantescompletos.columns)
    for j, c in enumerate(cols):
        if j == 0:
            continue

        if c not in variables.columns:
            continue

        vals = variables[c].tolist()
        mapping = {i + 1: v for i, v in enumerate(vals) if not pd.isna(v)}

        col_raw = informantescompletos[c].map(_coerce_code)

        mapped = col_raw.map(mapping)

        informantescompletos[c] = mapped.where(~mapped.isna(), col_raw)

    informantescompletos.to_parquet(respath / "informantes.parquet", index=False)

    temas = [k for k in xl.keys() if k not in ["Informantes", "Variables"]]

    for tema in temas:
        tabla = xl[tema]

        if tabla is None or tabla.empty:
            continue

        tmp = tabla.reset_index(drop=True).copy()
        tmp.insert(0, "user_id", range(1, len(tmp) + 1))

        col_order = [c for c in tabla.columns]
        pos_map = {c: i for i, c in enumerate(col_order)}

        tema_tokenized = tmp.melt("user_id", var_name="col", value_name="token")

        tema_tokenized = tema_tokenized.dropna(subset=["token"]).copy()
        tema_tokenized["token"] = tema_tokenized["token"].astype("string")
        tema_tokenized["token"] = tema_tokenized["token"].str.strip()

        tema_tokenized = tema_tokenized[
            (tema_tokenized["token"].notna())
            & (tema_tokenized["token"] != "")
            & (tema_tokenized["token"].str.lower() != "nan")
        ].copy()

        tema_tokenized["pos"] = tema_tokenized["col"].map(pos_map)
        tema_tokenized = tema_tokenized.dropna(subset=["pos"]).copy()
        tema_tokenized["pos"] = tema_tokenized["pos"].astype(int)

        tema_tokenized = (
            tema_tokenized[["user_id", "pos", "token"]]
            .sort_values(["user_id", "pos"])
            .reset_index(drop=True)
        )
        out_name = _sanitize_filename(tema) + ".parquet"
        tema_tokenized.to_parquet(respath / out_name, index=False)
