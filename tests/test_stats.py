import numpy as np
import pandas as pd
import pytest

from urlex.core.stats import estadisticas_df

# LAS DISPONIBILIDADES SE TOMAN DE LEXPRO, SIN CORREGIR LA ERRATA

# ============================================================
# Ejemplo 1
# U1: a,b,c
# U2: b,a,c
# U3: b,c,a
# Disponibilidades esperadas:
#   b=0.8215, a=0.5601, c=0.2987
# ============================================================


@pytest.fixture
def df_example_1():
    return pd.DataFrame(
        {
            "token": [
                # U1
                "a",
                "b",
                "c",
                # U2
                "b",
                "a",
                "c",
                # U3
                "b",
                "c",
                "a",
            ],
            "pos": [
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
            ],
            "user_id": [
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
            ],
        }
    )


# ============================================================
# Ejemplo 2
# U1: a,b
# U2: a,c
# U3: a,d
# Disponibilidades esperadas:
#   a=1.0, b=c=d=0.1055
# ============================================================


@pytest.fixture
def df_example_2():
    return pd.DataFrame(
        {
            "token": [
                # U1
                "a",
                "b",
                # U2
                "a",
                "c",
                # U3
                "a",
                "d",
            ],
            "pos": [
                0,
                1,
                0,
                1,
                0,
                1,
            ],
            "user_id": [
                1,
                1,
                2,
                2,
                3,
                3,
            ],
        }
    )


@pytest.fixture
def empty_df():
    return pd.DataFrame(columns=["token", "pos", "user_id"])


# ============================================================
# Helpers
# ============================================================


def _assert_no_duplicate_token_per_user(df: pd.DataFrame) -> None:
    assert not df.duplicated(subset=["user_id", "token"]).any()


def _assert_basic_properties(stats: pd.DataFrame) -> None:
    assert set(stats.columns) == {
        "token",
        "freq_rel",
        "disponibilidad",
        "avg_pos",
        "aparición",
        "freq_acum",
    }
    assert np.isclose(stats["freq_rel"].sum(), 1.0)
    assert np.all(stats["aparición"].between(0, 1))

    freq_acum = stats["freq_acum"].to_numpy()
    assert np.all(freq_acum[:-1] <= freq_acum[1:])

    disp = stats["disponibilidad"].to_numpy()
    assert np.all(disp[:-1] >= disp[1:])


# ============================================================
# Tests – Ejemplo 1
# ============================================================


def test_example_1_structure(df_example_1):
    _assert_no_duplicate_token_per_user(df_example_1)
    stats = estadisticas_df(df_example_1)
    _assert_basic_properties(stats)


def test_example_1_freqs(df_example_1):
    # 9 filas, a/b/c aparecen 3 veces -> 1/3
    stats = estadisticas_df(df_example_1).set_index("token")
    assert np.allclose(stats["freq_rel"], 1 / 3)


def test_example_1_avg_pos(df_example_1):
    """
    a: pos {0,1,2} -> 1
    b: pos {1,0,0} -> 1/3
    c: pos {2,2,1} -> 5/3
    """
    stats = estadisticas_df(df_example_1).set_index("token")
    assert np.isclose(stats.loc["a", "avg_pos"], 1.0)
    assert np.isclose(stats.loc["b", "avg_pos"], 1 / 3)
    assert np.isclose(stats.loc["c", "avg_pos"], 5 / 3)


def test_example_1_aparicion_is_one(df_example_1):
    # cada token aparece en los 3 usuarios
    stats = estadisticas_df(df_example_1)
    assert np.allclose(stats["aparición"], 1.0)


def test_example_1_disponibilidades(df_example_1):
    """
    Esperadas (por token):
      b = 0.8215
      a = 0.5601
      c = 0.2987
    """
    stats = estadisticas_df(df_example_1).set_index("token")
    assert np.isclose(stats.loc["b", "disponibilidad"], 0.8215, atol=1e-4)
    assert np.isclose(stats.loc["a", "disponibilidad"], 0.5601, atol=1e-4)
    assert np.isclose(stats.loc["c", "disponibilidad"], 0.2987, atol=1e-4)


# ============================================================
# Tests – Ejemplo 2
# ============================================================


def test_example_2_structure(df_example_2):
    _assert_no_duplicate_token_per_user(df_example_2)
    stats = estadisticas_df(df_example_2)
    _assert_basic_properties(stats)


def test_example_2_freqs(df_example_2):
    # 6 filas: a=3/6, b=1/6, c=1/6, d=1/6
    stats = estadisticas_df(df_example_2).set_index("token")
    assert np.isclose(stats.loc["a", "freq_rel"], 3 / 6)
    assert np.isclose(stats.loc["b", "freq_rel"], 1 / 6)
    assert np.isclose(stats.loc["c", "freq_rel"], 1 / 6)
    assert np.isclose(stats.loc["d", "freq_rel"], 1 / 6)


def test_example_2_aparicion(df_example_2):
    # ninf=3: a en 3/3; b,c,d en 1/3
    stats = estadisticas_df(df_example_2).set_index("token")
    assert np.isclose(stats.loc["a", "aparición"], 1.0)
    assert np.isclose(stats.loc["b", "aparición"], 1 / 3)
    assert np.isclose(stats.loc["c", "aparición"], 1 / 3)
    assert np.isclose(stats.loc["d", "aparición"], 1 / 3)


def test_example_2_avg_pos(df_example_2):
    # a siempre en pos 0 -> 0
    # b,c,d siempre en pos 1 -> 1
    stats = estadisticas_df(df_example_2).set_index("token")
    assert np.isclose(stats.loc["a", "avg_pos"], 0.0)
    assert np.isclose(stats.loc["b", "avg_pos"], 1.0)
    assert np.isclose(stats.loc["c", "avg_pos"], 1.0)
    assert np.isclose(stats.loc["d", "avg_pos"], 1.0)


def test_example_2_disponibilidades(df_example_2):
    """
    Esperadas (por token):
      a = 1.0
      b = 0.1055
      c = 0.1055
      d = 0.1055
    """
    stats = estadisticas_df(df_example_2).set_index("token")
    assert np.isclose(stats.loc["a", "disponibilidad"], 1.0, atol=1e-6)
    assert np.isclose(stats.loc["b", "disponibilidad"], 0.1055, atol=1e-4)
    assert np.isclose(stats.loc["c", "disponibilidad"], 0.1055, atol=1e-4)
    assert np.isclose(stats.loc["d", "disponibilidad"], 0.1055, atol=1e-4)


# ============================================================
# Caso vacío
# ============================================================


def test_empty_df(empty_df):
    stats = estadisticas_df(empty_df)

    assert isinstance(stats, pd.DataFrame)
    assert len(stats) == 0
    assert set(stats.columns) == {
        "token",
        "freq_rel",
        "disponibilidad",
        "avg_pos",
        "aparición",
        "freq_acum",
    }
