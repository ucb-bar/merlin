"""Phase 1's functional evidence is bought once; a launch may reuse it, but only when identical.

Every perf launch rebuilt the public+hidden functional GSIM certificate -- about twelve minutes, and
twice a failure -- for a result that could not differ. The certificate is keyed on the frozen
submission, and the only other thing that moves it is the GSIM build it pins, which the coordinator
already requires to equal the tuning certificate's. So reuse is admissible exactly when those pins
match, and this pins that it is checked rather than assumed: a root left behind by an interrupted
run, or one built against a different engine, must force a rebuild.
"""
from __future__ import annotations

import json
import sys

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts"))

import functional_gsim_qualification as Q  # noqa: E402


class _Source:
    """The minimum of a CertificateRecord that the reuse check reads."""

    def __init__(self, target="gemmini", pins=None):
        self.target = target
        self.pins = pins or {name: {"sha256": f"{i:064x}"}
                             for i, name in enumerate(_required_pins())}


def _required_pins():
    import perf_gsim_gate as GATE
    return list(GATE.REQUIRED_PINS)


def _root(tmp_path, *, pins, completion=True, target="gemmini", name="r"):
    root = tmp_path / name
    root.mkdir()
    certificate = {"schema_version": 1, "target": target, "members": [],
                   "pins": pins, "unresolved": []}
    (root / "functional-certificate.abc.json").write_text(json.dumps(certificate), encoding="utf-8")
    if completion:
        (root / "completion.abc.json").write_text(json.dumps({"status": "complete"}),
                                                  encoding="utf-8")
    return root


def _patch_load(monkeypatch, *, target="gemmini", pins):
    """Stand in for the real loader; this test is about the reuse DECISION, not certificate parsing."""
    import perf_gsim_gate as GATE

    class _Rec:
        def __init__(self):
            self.target = target
            self.pins = pins

    monkeypatch.setattr(Q.GATE, "load_certificate", lambda *a, **k: _Rec())
    return GATE


def test_a_finished_certificate_with_identical_pins_is_reused(tmp_path, monkeypatch):
    source = _Source()
    _patch_load(monkeypatch, pins=source.pins)
    root = _root(tmp_path, pins=source.pins)
    result = Q.reusable_certificate(root, source)
    assert result is not None
    path, digest = result
    assert path.name.startswith("functional-certificate.")
    assert digest


def test_a_different_engine_pin_forces_a_rebuild(tmp_path, monkeypatch):
    """The whole point: evidence taken on another simulator is not this run's evidence."""
    source = _Source()
    other = {name: {"sha256": "f" * 64} for name in source.pins}
    _patch_load(monkeypatch, pins=other)
    root = _root(tmp_path, pins=other)
    assert Q.reusable_certificate(root, source) is None


def test_a_root_without_a_completion_receipt_forces_a_rebuild(tmp_path, monkeypatch):
    """completion is written last, so its absence means the run was interrupted."""
    source = _Source()
    _patch_load(monkeypatch, pins=source.pins)
    root = _root(tmp_path, pins=source.pins, completion=False)
    assert Q.reusable_certificate(root, source) is None


def test_a_different_target_forces_a_rebuild(tmp_path, monkeypatch):
    source = _Source(target="gemmini")
    _patch_load(monkeypatch, target="atlas", pins=source.pins)
    root = _root(tmp_path, pins=source.pins, target="atlas")
    assert Q.reusable_certificate(root, source) is None


def test_an_absent_root_is_not_reuse(tmp_path):
    assert Q.reusable_certificate(tmp_path / "nope", _Source()) is None


def test_an_unloadable_certificate_forces_a_rebuild(tmp_path, monkeypatch):
    source = _Source()

    def _boom(*a, **k):
        raise ValueError("digest mismatch")

    monkeypatch.setattr(Q.GATE, "load_certificate", _boom)
    root = _root(tmp_path, pins=source.pins)
    assert Q.reusable_certificate(root, source) is None
