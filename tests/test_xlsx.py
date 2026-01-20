from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from urlex.core.xlsx_processing import pdprocessxlsx


def _write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    # engine openpyxl suele estar disponible; si no, añade dependencia
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def test_pdprocessxlsx_happy_path(tmp_path: Path) -> None:
    xlsx = tmp_path / "input.xlsx"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Informantes: primera col id, el resto son variables codificadas
    informantes = pd.DataFrame(
        {
            "CODIGO_INFORMANTE": [1, 2, 3],
            "ORIGEN": [1.0, "2", " 1 "],  # mezcla tipos para testear coerción
            "NIVELOPT": [2, 1, 2],
            "SIN_MAPEO": [99, 98, 97],  # NO existe en Variables → no rompe
        }
    )

    # Variables: columnas con el nombre de la variable (mismo nombre que en Informantes)
    # índice 1-based por enumeración: 1 -> first row, 2 -> second row, ...
    variables = pd.DataFrame(
        {
            "ORIGEN": ["LOCAL", "HERENCIA"],
            "NIVELOPT": ["BAJO", "BAJOMEDIO"],
        }
    )

    # Tema con tokens (incluye espacios, vacíos, NaN, "nan")
    tema1 = pd.DataFrame(
        {
            "A": [" gato ", "", None],
            "B": ["nan", " perro", "  "],
            "Unnamed: 0": [111, 222, 333],
        }
    )

    # Tema con nombre problemático
    tema_name_weird = "Tema.1.Prueba"
    tema2 = pd.DataFrame(
        {
            "X": [" hola", "adios"],
            "Y": [None, " mundo "],
        }
    )

    _write_xlsx(
        xlsx,
        {
            "Informantes": informantes,
            "Variables": variables,
            "Animales": tema1,
            tema_name_weird: tema2,
        },
    )

    pdprocessxlsx(str(xlsx), str(out_dir))

    # ---- Comprueba outputs ----
    inf_parquet = out_dir / "informantes.parquet"
    assert inf_parquet.exists()

    animales_parquet = out_dir / "Animales.parquet"
    assert animales_parquet.exists()

    # Tema2 debe estar sanitizado
    # La función sustituye chars raros por "_" y mantiene espacios (normaliza)
    # "Tema/1:Prueba" -> "Tema_1_Prueba.parquet"
    sanitized = out_dir / "Tema_1_Prueba.parquet"
    assert sanitized.exists()

    # ---- Valida informantes mapeados ----
    inf = pd.read_parquet(inf_parquet)

    # ORIGEN: 1 -> LOCAL, 2 -> HERENCIA (incluye 1.0, "2", " 1 ")
    assert inf.loc[0, "ORIGEN"] == "LOCAL"
    assert inf.loc[1, "ORIGEN"] == "HERENCIA"
    assert inf.loc[2, "ORIGEN"] == "LOCAL"

    # NIVELOPT: 2 -> BAJOMEDIO, 1 -> BAJO
    assert inf.loc[0, "NIVELOPT"] == "BAJOMEDIO"
    assert inf.loc[1, "NIVELOPT"] == "BAJO"
    assert inf.loc[2, "NIVELOPT"] == "BAJOMEDIO"

    # SIN_MAPEO no existe en Variables → se conserva tal cual
    assert list(inf["SIN_MAPEO"]) == [99, 98, 97]

    # ---- Valida tokenizado del tema Animales ----
    anim = pd.read_parquet(animales_parquet)

    # Debe tener columnas esperadas
    assert set(anim.columns) == {"user_id", "pos", "token"}

    # user_id debe ser 1..n filas originales
    assert sorted(anim["user_id"].unique().tolist()) == [1, 2, 3]

    # Tokens limpiados:
    # fila1: A="gato" (B era "nan" → se elimina), 111
    # fila2: B="perro" (A era "" → se elimina), 222
    # fila3: 333

    tokens_by_user = anim.groupby("user_id")["token"].apply(list).to_dict()

    assert tokens_by_user.get(1) == ["gato", "111"]
    assert tokens_by_user.get(2) == ["perro", "222"]
    assert tokens_by_user.get(3, []) == ["333"]

    # Posiciones: A=0, B=1 (Unnamed eliminado)
    # Para user 1 token "gato" venía de A -> pos 0
    row_gato = anim[(anim["user_id"] == 1) & (anim["token"] == "gato")].iloc[0]
    assert int(row_gato["pos"]) == 0


def test_pdprocessxlsx_missing_required_sheets(tmp_path: Path) -> None:
    xlsx = tmp_path / "bad.xlsx"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Falta "Informantes" y/o "Variables"
    _write_xlsx(xlsx, {"Algo": pd.DataFrame({"x": [1]})})

    with pytest.raises(ValueError) as e:
        pdprocessxlsx(str(xlsx), str(out_dir))

    msg = str(e.value)
    assert "Faltan hojas requeridas" in msg


def test_pdprocessxlsx_skips_empty_theme_sheet(tmp_path: Path) -> None:
    xlsx = tmp_path / "empty_theme.xlsx"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    informantes = pd.DataFrame({"ID": [1], "ORIGEN": [1]})
    variables = pd.DataFrame({"ORIGEN": ["LOCAL"]})

    # Tema vacío
    empty_theme = pd.DataFrame()

    _write_xlsx(
        xlsx,
        {
            "Informantes": informantes,
            "Variables": variables,
            "Vacio": empty_theme,
        },
    )

    pdprocessxlsx(str(xlsx), str(out_dir))

    # informantes existe
    assert (out_dir / "informantes.parquet").exists()
    # tema vacío no genera parquet
    assert not (out_dir / "Vacio.parquet").exists()
