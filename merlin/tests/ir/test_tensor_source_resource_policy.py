"""Finite source intake policies; tiny host controls, no SDK or model execution.

Set MERLIN_RESOURCE_POLICY_CANDIDATE_ROOT only for reviewing an uninstalled
candidate tree. Ordinary installed execution reads the canonical source files.
Allocation sentinels fire before a parent splat expansion if its guard regresses.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from xdsl.parser import Parser

from merlin.common.paths import repo_root
from merlin.verify import memory_scalar_nan_admission as canonical_admission
from merlin.verify.compiler_profile import SourceOutsideProfile
from merlin.verify.gate_errors import GateUnavailable
from merlin.verify.lean_graph_input_admission import InputAdmissionError

ROOT = Path(os.environ.get("MERLIN_RESOURCE_POLICY_CANDIDATE_ROOT", repo_root()))


def load(name):
    path = ROOT / "src/merlin/verify" / (name + ".py")
    spec = importlib.util.spec_from_file_location("merlin.verify._resource_candidate_" + name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def admission():
    return load("memory_scalar_nan_admission")


def literal(*items):
    attributes = ", ".join("a" + str(i) + " = " + value for i, value in enumerate(items))
    return "builtin.module attributes {" + attributes + "} {}"


def refuse_value_expansion(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dense literal reached value conversion before resource refusal")

    monkeypatch.setattr(Parser._TensorLiteralElement, "to_type", forbidden)


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "8"])
def test_optional_element_limit_is_validated_before_parsing(admission, limit):
    with pytest.raises(ValueError, match="positive integer or None"):
        admission._parse_bounded("not parsed", maximum_bytes=64, memory=False, maximum_dense_elements=limit)


@pytest.mark.parametrize("mode", ["bytes", "elements"])
def test_huge_compact_splat_refuses_before_parent_value_conversion(admission, monkeypatch, mode):
    refuse_value_expansion(monkeypatch)
    with pytest.raises(GateUnavailable, match="decoded-storage" if mode == "bytes" else "decoded-element"):
        admission._parse_bounded(
            literal("dense<0> : tensor<1152921504606846976xi8>"),
            maximum_bytes=8 if mode == "bytes" else 1 << 62,
            maximum_dense_elements=None if mode == "bytes" else 8,
            memory=False,
        )


@pytest.mark.parametrize("mode", ["bytes", "elements"])
def test_product_guard_runs_before_any_dense_allocation(admission, monkeypatch, mode):
    refuse_value_expansion(monkeypatch)
    with pytest.raises(GateUnavailable, match="decoded-storage" if mode == "bytes" else "decoded-element"):
        admission._parse_bounded(
            literal("dense<0> : tensor<3x3xi8>"),
            maximum_bytes=8 if mode == "bytes" else 64,
            maximum_dense_elements=None if mode == "bytes" else 8,
            memory=False,
        )


@pytest.mark.parametrize("mode", ["bytes", "elements"])
def test_multiple_literals_charge_the_aggregate_before_second_expansion(admission, monkeypatch, mode):
    original = Parser._TensorLiteralElement.to_type
    calls = []

    def once(self, *args, **kwargs):
        calls.append(True)
        if len(calls) > 1:
            pytest.fail("the second literal reached value conversion before aggregate refusal")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Parser._TensorLiteralElement, "to_type", once)
    with pytest.raises(GateUnavailable, match="decoded-storage" if mode == "bytes" else "decoded-element"):
        admission._parse_bounded(
            literal("dense<0.0> : tensor<3xf32>", "dense<1.0> : tensor<3xf32>"),
            maximum_bytes=20 if mode == "bytes" else 64,
            maximum_dense_elements=None if mode == "bytes" else 5,
            memory=False,
        )
    assert len(calls) == 1


def test_exact_aggregate_bounds_accept_small_values_and_empty_literals(admission):
    module = admission._parse_bounded(
        literal("dense<0.0> : tensor<3xf32>", "dense<1.0> : tensor<3xf32>", "dense<> : tensor<0xi8>"),
        maximum_bytes=24,
        maximum_dense_elements=6,
        memory=False,
    )
    assert len(module.attributes["a0"].data.data) == len(module.attributes["a1"].data.data) == 12
    assert module.attributes["a2"].data.data == b""


@pytest.mark.parametrize("raw", ["0000000000000080", "0000807F000080FF", "0100807F0100C07F"])
def test_dense_float_raw_bits_are_unchanged_with_optional_policy(admission, raw):
    source = literal('dense<"0x' + raw + '"> : tensor<2xf32>')
    baseline = canonical_admission._parse_bounded(source, maximum_bytes=8, memory=False)
    actual = admission._parse_bounded(source, maximum_bytes=8, maximum_dense_elements=2, memory=False)
    assert actual.attributes["a0"].data.data == baseline.attributes["a0"].data.data == bytes.fromhex(raw)
    admission.admit_loop_scalar_attributes(actual)
    if raw.startswith("0100807F"):
        with pytest.raises(InputAdmissionError, match="dense-nan"):
            admission.admit_memory_scalar_attributes(actual)
    else:
        admission.admit_memory_scalar_attributes(actual)


def test_scalar_nan_handling_and_runtime_tensor_signatures_are_unchanged(admission):
    module = admission._parse_bounded(
        literal("0x7F800001 : f32"), maximum_bytes=8, memory=False, maximum_dense_elements=1
    )
    for guard in (admission.admit_memory_scalar_attributes, admission.admit_loop_scalar_attributes):
        with pytest.raises(InputAdmissionError, match="scalar-nan"):
            guard(module)
    # Ordinary runtime tensor types charge no dense-literal storage/work budget.
    admission._parse_bounded(
        "builtin.module { func.func @f(%x:tensor<1152921504606846976xi8>) { func.return } }",
        maximum_bytes=1,
        maximum_dense_elements=1,
        memory=False,
    )


@pytest.mark.parametrize("family,budget", [("parse_memory_module", 256 * 1024), ("parse_loop_module", 4 * 1024 * 1024)])
def test_existing_family_byte_limits_and_no_optional_element_policy_are_unchanged(
    admission, monkeypatch, family, budget
):
    assert admission.MAX_DENSE_FLOAT_BYTES == 256 * 1024 and admission.MAX_LOOP_DENSE_BYTES == 4 * 1024 * 1024
    calls = []
    monkeypatch.setattr(admission, "_parse_bounded", lambda *args, **kwargs: calls.append((args, kwargs)))
    getattr(admission, family)("source-control")
    assert calls == [(("source-control",), {"maximum_bytes": budget, "memory": True})]


@pytest.mark.parametrize("family,budget", [("parse_memory_module", 256 * 1024), ("parse_loop_module", 4 * 1024 * 1024)])
def test_existing_family_oversize_splats_still_refuse_before_expansion(admission, monkeypatch, family, budget):
    refuse_value_expansion(monkeypatch)
    with pytest.raises(GateUnavailable, match="decoded-storage"):
        getattr(admission, family)(literal(f"dense<0> : tensor<{budget + 1}xi8>"))


@pytest.mark.parametrize("count", [5001, 10134, 32768, 32769])
def test_tensor_syntax_policy_and_forwarded_dense_limits_without_exporter(monkeypatch, tmp_path, count):
    profile = load("compiler_ir_source_profile")
    assert profile.MAX_SOURCE_BYTES == 4 * 1024 * 1024
    assert profile.MAX_SOURCE_OPERATIONS == 32768
    assert profile.MAX_DENSE_DECODED_BYTES == 64 * 1024 * 1024
    assert profile.MAX_DENSE_ELEMENTS == 8 * 1024 * 1024
    operation = SimpleNamespace(successors=[], regions=[])
    calls = []

    def parsed(raw, **kwargs):
        calls.append((raw, kwargs))
        return SimpleNamespace(walk=lambda: iter([operation] * count))

    class BeforeExporter(Exception):
        pass

    def stop():
        raise BeforeExporter("host policy marker only, no source export/proof/tool result")

    monkeypatch.setattr(canonical_admission, "_parse_bounded", parsed)
    monkeypatch.setattr(profile, "uuid", SimpleNamespace(uuid4=stop))
    validation = SimpleNamespace(gate=SimpleNamespace(directory=tmp_path))
    if count <= 32768:
        with pytest.raises(BeforeExporter):
            profile.admit_source("source-control", pass_name="cse", validation=validation)
    else:
        with pytest.raises(SourceOutsideProfile) as failure:
            profile.admit_source("source-control", pass_name="cse", validation=validation)
        assert failure.value.code == "source-syntax-bound" and "32768" in failure.value.reason
    assert len(calls) == 1
    assert {key: value for key, value in calls[0][1].items() if key != "context"} == {
        "maximum_bytes": 64 * 1024 * 1024,
        "maximum_dense_elements": 8 * 1024 * 1024,
        "memory": False,
    }


def test_raw_source_byte_policy_remains_separate_and_runs_before_dense_parse(monkeypatch):
    profile = load("compiler_ir_source_profile")
    monkeypatch.setattr(profile, "MAX_SOURCE_BYTES", 8)
    monkeypatch.setattr(
        canonical_admission, "_parse_bounded", lambda *a, **k: pytest.fail("raw byte limit reached parser")
    )
    with pytest.raises(SourceOutsideProfile) as failure:
        profile.admit_source("123456789", pass_name="cse", validation=None)
    assert failure.value.code == "source-byte-bound"
