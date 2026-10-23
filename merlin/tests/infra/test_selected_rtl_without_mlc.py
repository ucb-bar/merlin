"""Missing optional discovery must not prevent reading an explicit FIRRTL source."""

from merlin.targetgen.rtl import mlc_bridge
from merlin.targetgen.rtl.source_selection import selected_sources


def test_mesh_discovery_stays_unknown_without_mlc(monkeypatch):
    monkeypatch.setattr(mlc_bridge, "mlc_dir", lambda: None)
    with selected_sources({"target": "example"}):
        assert mlc_bridge.discovered_mesh("example") is None
        assert mlc_bridge.discovered_dim("example") is None
