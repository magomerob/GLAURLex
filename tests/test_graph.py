import networkx as nx
import pandas as pd

from urlex.core.graph import (
    bigrams_for_tema,
    bigrams_to_dirgraph,
    bigrams_to_undgraph,
    bigrams_to_unordered,
)


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["token_1", "token_2"]).reset_index(drop=True)


def test_bigrams_empty():
    df = pd.DataFrame(columns=["user_id", "pos", "token"])
    out = bigrams_for_tema(df)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0
    assert set(out.columns) == {"token_1", "token_2", "count"}


def test_bigrams_single_token_per_user():
    df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "pos": [0, 0],
            "token": ["a", "b"],
        }
    )
    out = bigrams_for_tema(df)
    assert len(out) == 0


def test_bigrams_counts():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2],
            "pos": [0, 1, 2, 3, 0, 1],
            "token": ["a", "b", "c", "d", "b", "c"],
        }
    )

    out = bigrams_for_tema(df)
    expected = pd.DataFrame(
        {
            "token_1": ["a", "b", "c"],
            "token_2": ["b", "c", "d"],
            "count": [1, 2, 1],
        }
    )

    pd.testing.assert_frame_equal(_sorted(out), _sorted(expected))


def test_bigrams_order():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "pos": [0, 1, 0, 1],
            "token": ["a", "b", "b", "a"],
        }
    )

    out = bigrams_for_tema(df)
    expected = pd.DataFrame(
        {
            "token_1": [
                "a",
                "b",
            ],
            "token_2": ["b", "a"],
            "count": [1, 1],
        }
    )

    pd.testing.assert_frame_equal(_sorted(out), _sorted(expected))


def test_bigrams_to_unordered():
    df = pd.DataFrame(
        {
            "token_1": ["a", "b", "c"],
            "token_2": ["b", "a", "d"],
            "count": [2, 3, 1],
        }
    )

    out = bigrams_to_unordered(df)
    expected = pd.DataFrame(
        {
            "token_1": ["a", "c"],
            "token_2": ["b", "d"],
            "count": [5, 1],
        }
    )

    pd.testing.assert_frame_equal(_sorted(out), _sorted(expected))


def test_bigrams_to_dirgraph_weights_and_nodes():
    df = pd.DataFrame(
        {
            "token_1": ["a", "b", "a"],
            "token_2": ["b", "c", "c"],
            "count": [2, 1, 4],
        }
    )

    out = bigrams_to_dirgraph(df)

    assert isinstance(out, nx.DiGraph)
    assert set(out.nodes()) == {"a", "b", "c"}
    assert out.has_edge("a", "b")
    assert out.has_edge("b", "c")
    assert out.has_edge("a", "c")
    assert len(out.edges()) == 3
    assert out["a"]["b"]["weight"] == 2
    assert out["b"]["c"]["weight"] == 1
    assert out["a"]["c"]["weight"] == 4


def test_bigrams_to_undgraph_edges():
    df = pd.DataFrame(
        {
            "token_1": ["a", "b", "c"],
            "token_2": ["b", "a", "d"],
            "count": [2, 3, 1],
        }
    )

    out = bigrams_to_undgraph(df)

    assert isinstance(out, nx.Graph)
    assert set(out.nodes()) == {"a", "b", "c", "d"}
    assert out.has_edge("a", "b")
    assert out.has_edge("b", "a")
    assert out.has_edge("c", "d")
    assert out.has_edge("d", "c")
    assert len(out.edges()) == 2
    assert out["a"]["b"]["weight"] == 5
    assert out["b"]["a"]["weight"] == 5
    assert out["c"]["d"]["weight"] == 1
