from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from glaurlex.core.salamanca_processing import pdprocesssalamanca


def test_pdprocesssalamanca_happy_path(tmp_path: Path) -> None:
    txt = tmp_path / "input.txt"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    txt.write_text(
        "\n".join(
            [
                "26012 601 01 IRALCOLEGIO, TRABAJAR, COMER, BEBER",
                "26012 602 01 DORMIR,  LEER",
                "26022 603 02 UNO, DOS",
            ]
        ),
        encoding="utf-8",
    )

    pdprocesssalamanca(str(txt), str(out_dir), stimulus_map={"1": "ACCIONES"})

    inf_parquet = out_dir / "informantes.parquet"
    acciones_parquet = out_dir / "ACCIONES.parquet"
    tema_02_parquet = out_dir / "tema_02.parquet"

    assert inf_parquet.exists()
    assert acciones_parquet.exists()
    assert tema_02_parquet.exists()

    inf = pd.read_parquet(inf_parquet).sort_values("CODIGO_INFORMANTE").reset_index(drop=True)
    assert list(inf["CODIGO_INFORMANTE"]) == [601, 602, 603]
    assert list(inf["NIVEL_CODIGO"]) == ["12", "12", "22"]
    assert list(inf["NIVEL"]) == ["B1", "B1", "C1"]

    acciones = pd.read_parquet(acciones_parquet)
    assert set(acciones.columns) == {"user_id", "pos", "type"}

    tokens_601 = acciones[acciones["user_id"] == 601].sort_values("pos")["type"].tolist()
    assert tokens_601 == ["IRALCOLEGIO", "TRABAJAR", "COMER", "BEBER"]

    tokens_602 = acciones[acciones["user_id"] == 602].sort_values("pos")["type"].tolist()
    assert tokens_602 == ["DORMIR", "LEER"]

    tema_02 = pd.read_parquet(tema_02_parquet)
    tokens_603 = tema_02[tema_02["user_id"] == 603].sort_values("pos")["type"].tolist()
    assert tokens_603 == ["UNO", "DOS"]


def test_pdprocesssalamanca_invalid_line_raises(tmp_path: Path) -> None:
    txt = tmp_path / "bad.txt"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    txt.write_text("linea mal formada", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        pdprocesssalamanca(str(txt), str(out_dir))

    assert "formato inválido" in str(exc.value)
