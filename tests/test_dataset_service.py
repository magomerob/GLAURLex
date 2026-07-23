from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from glaurlex.core.dataset_service import DatasetService


def _make_processed(processed_dir: Path, name: str) -> Path:
    """! Crea un dataset procesado mínimo (informantes + un tema)."""
    root = processed_dir / name
    root.mkdir(parents=True)
    pd.DataFrame({"CODIGO_INFORMANTE": [1, 2]}).to_parquet(root / "informantes.parquet")
    pd.DataFrame({"user_id": [1, 2], "pos": [0, 0], "type": ["gato", "perro"]}).to_parquet(
        root / "Animales.parquet"
    )
    return root


def test_delete_processed_removes_dataset(tmp_path: Path) -> None:
    service = DatasetService(tmp_path)
    root = _make_processed(tmp_path, "demo")
    assert "demo" in service.list_processed()

    service.delete_processed("demo")

    assert not root.exists()
    assert "demo" not in service.list_processed()


def test_delete_processed_missing_dataset_raises(tmp_path: Path) -> None:
    service = DatasetService(tmp_path)
    with pytest.raises(FileNotFoundError):
        service.delete_processed("no_existe")


def test_delete_processed_rejects_path_traversal(tmp_path: Path) -> None:
    base = tmp_path / "processed"
    outsider = tmp_path / "keep_me"
    outsider.mkdir(parents=True)
    (outsider / "informantes.parquet").write_text("data", encoding="utf-8")

    service = DatasetService(base)
    with pytest.raises(ValueError):
        service.delete_processed("../keep_me")

    # El directorio fuera del sandbox no se ha tocado.
    assert outsider.exists()
