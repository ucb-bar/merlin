"""The device runs on the operands the GRADE'S REFERENCE used — and only when they are the same thing.

Two failure modes, and the fix has to avoid both:

* embed the recorded operands NOWHERE and a capsule graded against an INDEPENDENT golden (one computed
  off-device on recorded operands) has its device harness materialize each leaf from its NAME instead.
  It then computes the right function of the wrong inputs, and the mismatch is reported as a functional
  failure of the submission. That is what the ten host-lane capsules would have hit the moment their
  results became deliverable;
* embed them EVERYWHERE and a capsule whose golden is RECOMPUTED on the integer engine breaks, because
  that golden is computed from the deterministic name-materialized fill while its ``golden.yaml`` may
  record different operands. ``GS0_matmul_spec`` is exactly that capsule (it passes today) and this
  file pins the distinction against it by name.

What separates the two is read from the buffer itself: a FLOAT-declared output cannot have been
graded against a golden the integer engine recomputed, so for that buffer -- and only that one --
``link_elf`` embeds the recorded operands.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.targetgen.contract import compile as contract_compile


class _Recorder:
    """Stands in for a backend's harness renderer and records the operands it was handed."""

    def __init__(self):
        self.inputs = "<not called>"

    def __call__(self, cb, *, target, inputs=None):
        self.inputs = inputs
        raise _Rendered


class _Rendered(Exception):
    """Raised once the renderer has been observed; the link that follows is not under test."""


def _link(cb, monkeypatch, tmp_path, *, inputs=None):
    rec = _Recorder()
    from merlin.runtime.backends import base
    monkeypatch.setattr(base, "harness_renderer", lambda target: rec)
    monkeypatch.setattr(base, "harness_build_recipe",
                        lambda target: SimpleNamespace(load_address=0, link_script=None,
                                                       support_sources=(), error_cls=RuntimeError))
    with pytest.raises(_Rendered):
        contract_compile.link_elf(cb, tmp_path / "kernel.o", tmp_path, target="synthetic",
                                  inputs=inputs)
    return rec.inputs


VALUES = {"arg0": [1.5, -2.0, 0.25, 3.0]}
RECORDED = {"arg0": {"shape": [2, 2], "values": VALUES["arg0"]}}


def _cb(out_dtype: str, **extra) -> dict:
    return dict({"tensors": {"arg0": {"shape": [2, 2], "dtype": "f32", "role": "input"},
                             "Y0": {"shape": [2, 2], "dtype": out_dtype, "role": "output"}},
                 "commands": []}, **extra)


def test_a_float_graded_buffer_runs_on_the_recorded_operands(monkeypatch, tmp_path):
    """A float compare can only be against an INDEPENDENT golden, so the device has to see the
    operands that golden was computed on."""
    assert _link(_cb("f32", canonical_inputs=RECORDED), monkeypatch, tmp_path) == VALUES


def test_an_integer_output_still_materializes_from_names(monkeypatch, tmp_path):
    """THE GS0 GUARD. An integer-declared output is graded against a golden RECOMPUTED from the
    deterministic fill; a golden.yaml that also records operands records DIFFERENT numbers. Embedding
    those would silently change the stimulus of every capsule that passes today."""
    assert _link(_cb("i32", canonical_inputs=RECORDED), monkeypatch, tmp_path) is None


def test_a_buffer_with_no_recorded_operands_is_unaffected(monkeypatch, tmp_path):
    assert _link(_cb("f32"), monkeypatch, tmp_path) is None


def test_an_explicit_argument_still_wins(monkeypatch, tmp_path):
    explicit = {"arg0": [9.0, 9.0, 9.0, 9.0]}
    got = _link(_cb("f32", canonical_inputs=RECORDED), monkeypatch, tmp_path, inputs=explicit)
    assert got == explicit


def test_a_recorded_operand_that_names_no_declared_tensor_is_dropped(monkeypatch, tmp_path):
    """The runner re-keys recorded operands onto the buffer's own tensor names; one that still does
    not match names nothing the harness can embed, and is not smuggled through under its own name."""
    cb = _cb("f32", canonical_inputs={"X": {"shape": [2, 2], "values": VALUES["arg0"]}})
    assert _link(cb, monkeypatch, tmp_path) is None


def test_gs0_records_operands_that_are_not_its_materialized_ones():
    """The measurement the guard exists for, asserted against the shipped capsule rather than
    described in a comment: GS0's golden is recomputed on the integer engine, and the operands its
    golden.yaml records are NOT the ones that recompute uses."""
    import yaml
    from merlin.common.paths import merlin_dir
    from merlin.targetgen import capsule_golden as cg

    d = merlin_dir() / "contract" / "capsules" / "isa" / "GS0_matmul_spec"
    if not (d / "golden.yaml").exists():
        pytest.skip("GS0_matmul_spec golden not present in this checkout")
    capsule = yaml.safe_load((d / "capsule.yaml").read_text(encoding="utf-8"))
    assert not cg.is_independent_float_golden(capsule, d), "GS0's golden is the recomputed one"
    assert (capsule.get("numeric_policy") or {}).get("compare") == "exact_int"
    recorded = cg.canonical_input_values(capsule, d)
    assert recorded, "GS0 does record operands — which is what makes it the hazard"
    materialized = cg.materialize_capsule_leaves(capsule)
    assert any(list(materialized[name].data) != [int(v) for v in spec["values"]]
               for name, spec in recorded.items() if name in materialized)
