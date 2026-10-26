"""A named-engine self-check grades Core ATen capsules through the grade's own full-call tiers.

Core ATen capsules execute only through ``run_full_call``. The plain simulator adapters a named
``--sim`` builds lack it, so without a candidate policy every such capsule read back
``oracle_unavailable``. The self-check now takes a tier from the grader's ladder, but only when that
adapter runs the engine the caller selected: Spike at L2, the named engine at L3.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from merlin_experiments.phase1.feedback import selfcheck

from merlin.targetgen import capsule_common
from merlin.targetgen import core_aten_capsules as C
from merlin.targetgen.core_aten_capture import case_capture_name

TARGET = "fixture_target"
A = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
B = np.array([[5.0, 6.0], [7.0, 8.0]], np.float32)
Y = A @ B


def _tensor(values):
    return dict(
        kind="tensor",
        dtype="float32",
        shape=list(values.shape),
        stride=[values.shape[1], 1],
        storage_offset=0,
        requires_grad=False,
        values=values.tolist(),
    )


def _meta(tensor):
    return {key: value for key, value in tensor.items() if key not in {"kind", "values"}}


@pytest.fixture
def mm_capsule(tmp_path):
    """The public ``aten.mm.default`` capsule, written by the ordinary capsule writer."""
    a, b, y = _tensor(A), _tensor(B), _tensor(Y)
    arguments = dict(args=dict(kind="tuple", items=[a, b]), kwargs={})
    case = dict(
        overload="aten.mm.default",
        arguments=arguments,
        post_arguments=json.loads(json.dumps(arguments)),
        expected=y,
        comparison="torch_close",
        comparison_parameters={"rtol": 0, "atol": 0},
        mutated_arguments=[],
        output_input_aliases=[],
    )
    source = tmp_path / "captures" / case_capture_name(case)
    source.mkdir(parents=True)
    abi = dict(dtype="f32", shape=[2, 2])
    capture = dict(
        overload=case["overload"],
        status="captured_exact",
        capture_meta=dict(
            ok=True,
            opaque=0,
            input_abi=[abi, abi],
            output_abi=[abi],
            result_contract=dict(
                authority="same_conversion_exported_program",
                inputs=[_meta(a), _meta(b)],
                results=[dict(_meta(y), role="user_output", alias_inputs=[])],
            ),
        ),
    )
    (source / "capture.json").write_text(json.dumps(capture))
    (source / "inputs.json").write_text(json.dumps([A.tolist(), B.tolist()]))
    (source / "golden.json").write_text('{"private_answer": 1}')
    (source / "capsule.pytorch.py").write_text("# input loader fixture\n")
    (source / "capsule.linalg.mlir").write_text(
        "builtin.module {\nfunc.func @forward(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {\n"
        '%y = "tensor.cast"(%a) {prov.aten = "aten.mm.default"} : (tensor<2x2xf32>) -> tensor<2x2xf32>\n'
        "func.return %y : tensor<2x2xf32>\n}\n}\n"
    )
    public, private = tmp_path / "public", tmp_path / "_private" / "public"
    manifest = C.write_capsules({"cases": [case]}, source.parent, public, private)
    [entry] = manifest["capsules"]
    capsule = capsule_common.load_capsule(public / entry["id"])
    capsule["required_oracle_tiers"] = ["L2", "L3"]
    # The boundary reads back the result, then every caller input after the call.
    readback = dict(
        metadata=[_meta(y), _meta(a), _meta(b)],
        aliases=[[], [], []],
        pre_bytes={"0": A.tobytes().hex(), "1": B.tobytes().hex()},
    )
    return entry["id"], capsule, [Y.tobytes(), A.tobytes(), B.tobytes()], readback


def _plain(engine):
    """A simulator adapter as ``CR.simulator_adapter`` builds it: no full-call boundary."""

    def run(*_args):
        raise AssertionError("a command-buffer adapter cannot run a full-call capsule")

    run._merlin_simulator_engine = engine
    return run


def _full_call(engine, calls, output, readback):
    """A grader full-call adapter for ``engine`` that returns the candidate boundary bytes."""

    def execute(**_kwargs):
        calls.append(engine)
        return dict(
            output_bytes=output,
            semantic_readback=readback,
            provenance={"engine": engine},
            engine=engine,
            derived_from_rtl=engine != "spike",
        )

    def run(*_args):
        raise AssertionError("this oracle executes the full-call boundary only")

    run.run_full_call = execute
    run._merlin_simulator_engine = engine
    return run


def _select(monkeypatch, graded):
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    monkeypatch.setattr(selfcheck, "grade_l3_selection", lambda target, sim: None)
    monkeypatch.setattr(selfcheck.CR, "simulator_adapter", lambda engine, target, **kw: _plain(engine))
    requests = []
    monkeypatch.setattr(
        selfcheck.CR, "oracle_adapters", lambda target, sim_via, **kw: requests.append((target, sim_via)) or graded
    )
    return requests


def _grade(capsule, adapters, tmp_path, monkeypatch):
    monkeypatch.setattr(capsule_common, "run_entrypoints", lambda *a, **kw: (None, {}, "submitted LLVM"))
    config = SimpleNamespace(target=TARGET, fourth_output_name="lowered.mlir", rtl_tiers={"L3"})
    paths = SimpleNamespace(run_path=tmp_path / "runs")
    options = dict(paths=paths, config=config, pkg=None, contract=None, timeout=1, no_oracle=False)
    return C.run_capsule(capsule, tmp_path, adapters=adapters, **options)


def test_mm_capsule_reaches_both_tiers_through_same_engine_full_call_adapters(mm_capsule, tmp_path, monkeypatch):
    name, capsule, output, readback = mm_capsule
    assert name == "aten__mm__default"
    calls = []
    graded = {
        "L2": _full_call("spike", calls, output, readback),
        "L3": _full_call("verilator", calls, output, readback),
    }
    requests = _select(monkeypatch, graded)
    adapters, sim = selfcheck._adapters("verilator", TARGET, "chipyard")
    assert sim == "verilator" and requests == [(TARGET, "chipyard")]
    assert adapters == graded
    result = _grade(capsule, adapters, tmp_path, monkeypatch)
    assert (result.get("failure") or {}).get("plane") != "oracle_unavailable", result["failure"]
    assert result["status"] == "pass" and calls == ["spike", "verilator"]
    assert {tier: row["status"] for tier, row in result["tiers"].items()} == {"L2": "pass", "L3": "pass"}


@pytest.mark.parametrize(
    "graded_engines",
    [
        {"L2": "verilator", "L3": "gsim"},  # neither tier runs the selected engine
        {"L2": "gsim", "L3": "spike"},  # swapped onto the wrong tiers
        {"L2": None, "L3": "vcs"},  # a missing L2 and a different RTL engine
    ],
)
def test_a_different_engine_full_call_adapter_is_never_substituted(mm_capsule, tmp_path, monkeypatch, graded_engines):
    _name, capsule, output, readback = mm_capsule
    calls = []
    graded = {tier: _full_call(engine, calls, output, readback) for tier, engine in graded_engines.items() if engine}
    _select(monkeypatch, graded)
    adapters, _ = selfcheck._adapters("verilator", TARGET, "chipyard")
    assert [adapters[tier]._merlin_simulator_engine for tier in ("L2", "L3")] == ["spike", "verilator"]
    assert not any(hasattr(adapter, "run_full_call") for adapter in adapters.values())
    result = _grade(capsule, adapters, tmp_path, monkeypatch)
    assert result["failure"] == {"plane": "oracle_unavailable", "category": "NOT_RUN_IS_NOT_PASS", "tier": "L2"}
    assert calls == []


def test_an_existing_full_call_tier_is_kept_and_a_spike_screen_adds_no_rtl_tier(mm_capsule, monkeypatch):
    _name, _capsule, output, readback = mm_capsule
    calls = []
    graded = {"L2": _full_call("spike", calls, output, readback), "L3": _full_call("spike", calls, output, readback)}
    _select(monkeypatch, graded)
    adapters, sim = selfcheck._adapters("spike", TARGET, "chipyard")
    assert sim == "spike" and set(adapters) == {"L2"} and adapters["L2"] is graded["L2"]
