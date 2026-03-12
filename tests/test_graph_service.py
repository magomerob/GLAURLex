import networkx as nx

from glaurlex.core.graph_service import GraphService


def test_graph_service_save_load_gml(tmp_path):
    processed_dir = tmp_path / "processed"
    dataset_dir = processed_dir / "ds1"
    dataset_dir.mkdir(parents=True)

    svc = GraphService(processed_dir)
    g = nx.DiGraph()
    g.add_edge("a", "b", weight=2)

    path = svc.save_graph("ds1", "g1", g)
    assert path.exists()

    loaded = svc.load_graph("ds1", "g1")
    assert isinstance(loaded, nx.DiGraph)
    assert loaded.has_edge("a", "b")
    assert loaded["a"]["b"]["weight"] == 2


def test_graph_service_list_graphs(tmp_path):
    processed_dir = tmp_path / "processed"
    dataset_dir = processed_dir / "ds1"
    dataset_dir.mkdir(parents=True)

    svc = GraphService(processed_dir)
    g = nx.Graph()

    svc.save_graph("ds1", "g1", g)
    svc.save_graph("ds1", "g2", g)

    assert svc.list_graphs("ds1") == ["g1", "g2"]
