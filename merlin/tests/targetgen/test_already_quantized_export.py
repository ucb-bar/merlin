"""An export's mutation contract determines immutable lifted buffer roles."""
from types import SimpleNamespace

import pytest

from merlin.targetgen._pt2_capture_worker import _read_only_buffer_arguments, main


def test_only_export_declared_readonly_buffers_become_parameters():
    def spec(kind, target):
        return SimpleNamespace(kind=SimpleNamespace(name=kind), target=target)
    signature = SimpleNamespace(
        input_specs=[spec("PARAMETER", "weight"), spec("BUFFER", "codes"), spec("BUFFER", "state"),
                     spec("USER_INPUT", None), spec("BUFFER", "exponents")],
        buffers_to_mutate={"state_update": "state"},
    )
    assert _read_only_buffer_arguments(SimpleNamespace(graph_signature=signature)) == [1, 4]


def test_already_quantized_export_refuses_quantization_before_loading(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        main(["--exported", str(tmp_path / "never-loaded.pt2"), "--out", str(tmp_path / "out"),
              "--already-quantized", "--quant", "some_recipe"])
    assert not (tmp_path / "out").exists()
