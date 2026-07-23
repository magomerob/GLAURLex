from __future__ import annotations

from pathlib import Path

import pandas as pd

from glaurlex.core.example_data import (
    EXAMPLE_SALAMANCA_FILENAME,
    EXAMPLE_XLSX_FILENAME,
    build_example_salamanca,
    build_example_xlsx,
    salamanca_stimulus_map,
)
from glaurlex.core.salamanca_processing import pdprocesssalamanca
from glaurlex.core.xlsx_processing import pdprocessxlsx


def test_example_xlsx_roundtrips_through_pdprocessxlsx(tmp_path: Path) -> None:
    xlsx = tmp_path / EXAMPLE_XLSX_FILENAME
    out_dir = tmp_path / "out"

    data = build_example_xlsx()
    assert isinstance(data, (bytes, bytearray)) and len(data) > 0
    xlsx.write_bytes(data)

    pdprocessxlsx(str(xlsx), str(out_dir))

    inf_parquet = out_dir / "informantes.parquet"
    assert inf_parquet.exists()
    inf = pd.read_parquet(inf_parquet)
    assert "CODIGO_INFORMANTE" in inf.columns
    # Los códigos de variable se mapean a etiquetas de la hoja "Variables".
    assert set(inf["NIVEL"].unique()) <= {"A1", "A2", "B1", "B2", "C1"}
    assert set(inf["SEXO"].unique()) <= {"HOMBRE", "MUJER"}

    # Al menos un tema con el contrato [user_id, pos, type].
    tema_parquets = [p for p in out_dir.glob("*.parquet") if p.name != "informantes.parquet"]
    assert tema_parquets
    tema = pd.read_parquet(tema_parquets[0])
    assert set(tema.columns) == {"user_id", "pos", "type"}
    assert tema["user_id"].nunique() == len(inf)


def test_example_salamanca_roundtrips_through_pdprocesssalamanca(tmp_path: Path) -> None:
    txt = tmp_path / EXAMPLE_SALAMANCA_FILENAME
    out_dir = tmp_path / "out"

    text = build_example_salamanca()
    assert isinstance(text, str) and text.strip()
    txt.write_text(text, encoding="utf-8")

    pdprocesssalamanca(str(txt), str(out_dir), stimulus_map=salamanca_stimulus_map())

    inf = pd.read_parquet(out_dir / "informantes.parquet")
    assert set(inf["NIVEL"].unique()) == {"B1", "C1"}

    # El diccionario de estímulos nombra los temas (no quedan como tema_01...).
    for tema_name in salamanca_stimulus_map().values():
        assert (out_dir / f"{tema_name}.parquet").exists()
