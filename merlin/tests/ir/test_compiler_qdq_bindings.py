"""Canonical source-byte mutation guards; no substituted proof or frontend verdict."""

from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.verify import compiler_qdq as qdq
from merlin.verify.gate_errors import GateRejected

OBSERVERS = (
    "src/merlin/integrations/model2mlir.py",
    "src/merlin/integrations/__init__.py",
    "src/merlin/common/provenance.py",
    "src/merlin/common/provenance_lost.py",
    "src/merlin/targetgen/__init__.py",
    "src/merlin/targetgen/_m2m_capture_worker.py",
    "src/merlin/targetgen/_m2m_capture_artifacts.py",
)


@pytest.mark.parametrize("observer", OBSERVERS)
def test_consumed_source_observer_mutation_invalidates_bound_bytes(tmp_path, observer):
    """Use real project bytes in an isolated copy; issue no semantic completion."""
    original = repo_root()
    for name in qdq._SOURCE_BINDING_PATHS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / name).read_bytes())
    refs = qdq._source_refs(tmp_path)
    by_name = {Path(ref["path"]).relative_to(tmp_path).as_posix(): ref for ref in refs}
    assert set(OBSERVERS) <= set(by_name)
    for ref in refs:
        qdq.bindings._bound(ref)
    target = tmp_path / observer
    target.write_bytes(target.read_bytes() + b"\n# isolated observer mutation\n")
    with pytest.raises(GateRejected, match="bound checker/input/report bytes changed"):
        qdq.bindings._bound(by_name[observer])
