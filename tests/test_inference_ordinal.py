"""Tests for ordinal-variable support in compare_groups()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glaurlex.core.inference import compare_groups
from glaurlex.core.variables_store import (
    get_order,
    is_ordinal,
    load_variables,
    save_variables,
)


@pytest.fixture
def trend_df() -> pd.DataFrame:
    """A 3-level ordinal predictor with a strong monotonic trend."""
    rng = np.random.default_rng(42)
    rows = []
    for level, mu in [("a", 1.0), ("b", 2.0), ("c", 3.0)]:
        for v in rng.normal(loc=mu, scale=0.3, size=30):
            rows.append({"metric": float(v), "level": level})
    return pd.DataFrame(rows)


def test_descriptives_respect_declared_order(trend_df: pd.DataFrame):
    res = compare_groups(trend_df, "metric", "level", order=["c", "b", "a"])
    assert res.descriptives["level"].tolist() == ["c", "b", "a"]


def test_descriptives_default_alphabetical_without_order(trend_df: pd.DataFrame):
    res = compare_groups(trend_df, "metric", "level")
    assert res.descriptives["level"].tolist() == ["a", "b", "c"]
    assert res.trend is None


def test_trend_block_present_when_order_given(trend_df: pd.DataFrame):
    res = compare_groups(trend_df, "metric", "level", order=["a", "b", "c"])
    assert res.trend is not None
    assert "Spearman ρ" in res.trend
    assert res.trend["Spearman p_value"] < 0.05
    assert res.trend["Spearman ρ"] > 0.5
    # Reverse order flips the sign.
    res_rev = compare_groups(trend_df, "metric", "level", order=["c", "b", "a"])
    assert res_rev.trend["Spearman ρ"] < -0.5


def test_kgroup_kwargs_unchanged_when_order_omitted(trend_df: pd.DataFrame):
    """Regression: the existing k-group output shape is preserved."""
    res = compare_groups(trend_df, "metric", "level")
    assert res.kind == "k-group"
    assert "F" in res.parametric
    assert "H" in res.non_parametric
    assert res.posthoc is not None


def test_extra_levels_in_order_are_ignored(trend_df: pd.DataFrame):
    """Levels declared in `order` but not in data must not crash."""
    res = compare_groups(
        trend_df, "metric", "level", order=["a", "b", "c", "ghost"]
    )
    # Descriptive table only includes data levels.
    assert set(res.descriptives["level"]) == {"a", "b", "c"}
    assert res.trend is not None


def test_levels_missing_from_order_are_appended(trend_df: pd.DataFrame):
    """Levels present in data but absent from `order` go to the end alphabetically."""
    res = compare_groups(trend_df, "metric", "level", order=["b"])
    levels_out = res.descriptives["level"].tolist()
    assert levels_out[0] == "b"
    assert set(levels_out[1:]) == {"a", "c"}
    assert levels_out[1:] == sorted(levels_out[1:])


# ---------------------------------------------------------------------------
# Matriz de tests para compare_groups (independiente del bloque ordinal)
# ---------------------------------------------------------------------------


def _build_groups_df(
    means: list[float], n_per_group: int = 25, scale: float = 0.4, seed: int = 0
) -> pd.DataFrame:
    """Construye un DataFrame con un nivel por media y N constante por nivel."""
    rng = np.random.default_rng(seed)
    rows = []
    for idx, mu in enumerate(means):
        level = chr(ord("a") + idx)
        for v in rng.normal(loc=mu, scale=scale, size=n_per_group):
            rows.append({"metric": float(v), "level": level})
    return pd.DataFrame(rows)


# Cada caso: (id, medias_por_grupo, kind_esperado, hay_efecto, seed)
_GROUP_MATRIX = [
    ("2g_null", [1.0, 1.0], "two-group", False, 101),
    ("2g_strong", [0.0, 3.0], "two-group", True, 102),
    ("3g_null", [1.0, 1.0, 1.0], "k-group", False, 103),
    ("3g_strong", [0.0, 1.5, 3.0], "k-group", True, 104),
    ("4g_strong", [0.0, 1.0, 2.0, 3.0], "k-group", True, 105),
]


@pytest.mark.parametrize(
    "case_id, means, expected_kind, has_effect, seed",
    _GROUP_MATRIX,
    ids=[c[0] for c in _GROUP_MATRIX],
)
def test_compare_groups_matrix_shape_and_signal(
    case_id, means, expected_kind, has_effect, seed
):
    """kind, n_total y dirección de los p-values según el caso."""
    df = _build_groups_df(means, n_per_group=25, seed=seed)
    res = compare_groups(df, "metric", "level")

    assert res.kind == expected_kind
    assert res.n_total == 25 * len(means)
    assert len(res.descriptives) == len(means)

    p_param = (
        res.parametric.get("p_value")
        if expected_kind == "two-group"
        else res.parametric.get("p_value")
    )
    p_np = res.non_parametric.get("p_value")

    if has_effect:
        assert p_param < 0.05
        assert p_np < 0.05
    else:
        assert p_param > 0.05
        assert p_np > 0.05


@pytest.mark.parametrize(
    "means, expected_kind",
    [([0.0, 3.0], "two-group"), ([0.0, 1.5, 3.0], "k-group")],
)
def test_compare_groups_matrix_effect_size_keys(means, expected_kind):
    """Las claves de tamaño de efecto dependen del kind."""
    df = _build_groups_df(means, n_per_group=20, seed=7)
    res = compare_groups(df, "metric", "level")

    assert res.kind == expected_kind
    if expected_kind == "two-group":
        assert "Cohen's d" in res.effect_size
        assert "rank-biserial r" in res.effect_size
        assert abs(res.effect_size["Cohen's d"]) > 1.0
        assert res.posthoc is None
    else:
        assert "η² (ANOVA)" in res.effect_size
        assert "ε² (Kruskal-Wallis)" in res.effect_size
        assert 0.0 <= res.effect_size["η² (ANOVA)"] <= 1.0
        assert res.posthoc is not None


@pytest.mark.parametrize("posthoc_flag", [True, False])
def test_compare_groups_matrix_posthoc_toggle(posthoc_flag):
    df = _build_groups_df([0.0, 1.5, 3.0], n_per_group=20, seed=3)
    res = compare_groups(df, "metric", "level", posthoc=posthoc_flag)

    if posthoc_flag:
        assert res.posthoc is not None
        # 3 grupos → 3 comparaciones pareadas.
        assert len(res.posthoc) == 3
        # Bonferroni nunca aumenta la significación.
        assert (res.posthoc["p_Bonferroni"] >= res.posthoc["p_value"]).all()
    else:
        assert res.posthoc is None


def test_compare_groups_matrix_insufficient_levels():
    """Un único nivel con datos → kind='insufficient' y nota informativa."""
    df = pd.DataFrame({"metric": [1.0, 2.0, 3.0, 4.0], "level": ["a"] * 4})
    res = compare_groups(df, "metric", "level")
    assert res.kind == "insufficient"
    assert res.parametric == {}
    assert res.non_parametric == {}
    assert any("≥2 niveles" in n for n in res.notes)


def test_compare_groups_matrix_drops_singleton_groups():
    """Niveles con N<2 se ignoran al elegir el test, pero suman en descriptivos."""
    df = pd.concat(
        [
            _build_groups_df([0.0, 3.0], n_per_group=15, seed=11),
            pd.DataFrame({"metric": [5.0], "level": ["c"]}),
        ],
        ignore_index=True,
    )
    res = compare_groups(df, "metric", "level")
    assert res.kind == "two-group"
    # n_total cuenta solo grupos válidos (≥2 obs.), no el singleton.
    assert res.n_total == 30


def test_variables_store_roundtrip(tmp_path):
    name = "demo"
    (tmp_path / name).mkdir()

    cfg = {
        "EDAD": {"ordinal": True, "order": ["joven", "adulto", "mayor"]},
        # No-ordinal entry should not be persisted.
        "SEXO": {"ordinal": False, "order": []},
    }
    save_variables(str(tmp_path), name, cfg)
    loaded = load_variables(str(tmp_path), name)

    assert "EDAD" in loaded
    assert loaded["EDAD"]["ordinal"] is True
    assert loaded["EDAD"]["order"] == ["joven", "adulto", "mayor"]
    assert "SEXO" not in loaded

    assert is_ordinal(loaded, "EDAD")
    assert get_order(loaded, "EDAD") == ["joven", "adulto", "mayor"]
    assert not is_ordinal(loaded, "MISSING")
