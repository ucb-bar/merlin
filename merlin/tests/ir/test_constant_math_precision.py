"""Bounded power preparation is explicit, replayable and independent of host FP."""

import copy
import json
import random
import struct
import subprocess
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from types import SimpleNamespace

import pytest

from merlin.semantic_compiler.constant_math_backend import HELPER_ENV, MathHelper, MathUnavailable, build_helper
from merlin.semantic_compiler.constant_math_precision import prepare_constant_math, verify_constant_math
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from merlin.semantic_compiler.scalar_reference import _round
from merlin.semantic_compiler.source_preparation import prepare_source, verify_source_preparation
from merlin.tests.ir.test_semantic_constant_folding import request


@pytest.fixture(scope="module")
def helper(tmp_path_factory):
    return build_helper(tmp_path_factory.mktemp("math-parent") / "helper")[0]


def bits(value, width):
    return int.from_bytes(struct.pack("<f" if width == 32 else "<d", value), "little")


def test_old_preparation_replays_without_adopting_new_tensor_folds(helper, monkeypatch):
    monkeypatch.setenv(HELPER_ENV, str(helper.path))
    source = power(32, [bits(2.0, 32)], [bits(2.0, 32)])
    identity = SemanticNode("base_view", "identity", (source.nodes[0].id,), source.nodes[0].type)
    exponent = replace(source.nodes[-1], inputs=(identity.id, source.nodes[1].id))
    source = replace(source, nodes=(*source.nodes[:-1], identity, exponent))
    old, legacy = prepare_source(source, constant_math_policy=helper.policy(), include_tensor_constants=False)
    assert legacy["schema"] == "merlin.source_preparation.v1" and old.node(exponent.id).effect == "pure"
    assert verify_source_preparation(old, legacy).digest() == source.digest()
    current, record = prepare_source(source, constant_math_policy=helper.policy())
    assert record["schema"] == "merlin.source_preparation.v2" and current.node(exponent.id).effect == "constant"
    assert verify_source_preparation(current, record).digest() == source.digest()


def power(width, bases, exponents):
    dtype = f"f{width}"
    return request(
        f"source_{dtype}_powf_unqualified", dtype, dtype, f"source-{dtype}-powf-unqualified-v1", [bases, exponents]
    )


@pytest.mark.parametrize("width", [32, 64])
def test_integer_exponents_match_independent_rational_rounding(helper, width):
    rng = random.Random(width)
    bias, fraction = (127, 23) if width == 32 else (1023, 52)
    bases = [1, (1 << fraction) - 1, 1 << fraction, bits(1.0, width), ((2 * bias) << fraction) | ((1 << fraction) - 1)]
    bases += [rng.randrange(1, ((2 * bias + 1) << fraction)) for _ in range(96)]
    for base in bases:
        value = Fraction(struct.unpack("<f" if width == 32 else "<d", base.to_bytes(width // 8, "little"))[0])
        for exponent in (-12, -2, -1, 0, 1, 2, 3, 12):
            actual, _ = helper.power(width, base, bits(exponent, width), 1024)
            assert actual == _round(value**exponent, width)


def test_noninteger_bounds_and_precision_limit(helper):
    rng = random.Random(83)
    for width in (32, 64):
        for _ in range(96):
            base = bits(rng.uniform(0.01, 100), width)
            value = struct.unpack("<f" if width == 32 else "<d", base.to_bytes(width // 8, "little"))[0]
            with localcontext() as context:
                context.prec = 180
                root = Decimal.from_float(value).sqrt()
                # Independent Decimal endpoints must have the same IEEE code.
                lower = _round(Fraction(root.next_minus()), width)
                upper = _round(Fraction(root.next_plus()), width)
                assert lower == upper == helper.power(width, base, bits(0.5, width), 1024)[0]
    a, b = 0x40158FB8EBD434B5, 0xC03BA72CB48FE1C6
    with pytest.raises(ArithmeticError, match="inconclusive"):
        helper.power(64, a, b, 64)
    assert helper.power(64, a, b, 128) == (0x3BBBB741FA1A0ED9, 128)
    source = power(64, [a], [b])
    unchanged, receipt = prepare_constant_math(source, helper.policy(max_precision_bits=64), helper=helper)
    assert unchanged == source and receipt["deferred"][0]["status"] == "resource_limit"


def test_whole_tensor_domain_work_and_no_implicit_selection(helper, monkeypatch):
    one, two = bits(1.0, 32), bits(2.0, 32)
    for base, exponent in [(0, one), (0x80000000, one), (0xBF800000, two), (0x7F800000, one), (one, 0x7FC00000)]:
        source = power(32, [two, base], [two, exponent])
        prepared, receipt = prepare_constant_math(source, helper.policy(), helper=helper)
        assert prepared == source and not receipt["folds"]
        assert receipt["deferred"][0]["status"] == "unsupported_semantics"
    source = power(32, [two, two], [two, two])
    prepared, receipt = prepare_constant_math(source, helper.policy(element_budget=1), helper=helper)
    assert prepared == source and receipt["deferred"][0]["status"] == "resource_limit"
    monkeypatch.delenv(HELPER_ENV, raising=False)
    prepared, receipt = prepare_source(source)
    assert prepared == source and not receipt["folds"]
    with pytest.raises(MathUnavailable):
        prepare_source(source, constant_math_policy=helper.policy())
    monkeypatch.setenv(HELPER_ENV, str(helper.path))
    for field in ("helper_sha256", "dependencies"):
        policy = copy.deepcopy(helper.policy())
        if field == "helper_sha256":
            policy[field] = "0" * 64
        else:
            policy[field][0]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="identity changed"):
            MathHelper.selected(policy)


def test_rounds_replay_multiple_outputs_and_mutations(helper, monkeypatch):
    monkeypatch.setenv(HELPER_ENV, str(helper.path))
    source = power(32, [bits(2.0, 32)], [bits(3.0, 32)])
    type_ = source.node("result").type
    add = SemanticNode(
        "added",
        "f32_add",
        ("result", "c0"),
        type_,
        (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
        index_maps=pointwise_identity_maps(1, 2),
    )
    second = replace(source.node("result"), id="second", inputs=("added", "c0"))
    source = replace(
        source,
        nodes=(*source.nodes, add, second),
        outputs=("second", "result"),
        output_storages=("external", "external"),
    )
    limited, receipt = prepare_source(source, constant_math_policy=helper.policy(max_rounds=1))
    assert receipt["limits"][0]["status"] == "resource_limit"
    assert verify_source_preparation(limited, receipt).digest() == source.digest()
    prepared, receipt = prepare_source(source, constant_math_policy=helper.policy())
    assert not receipt["limits"] and prepared.outputs == source.outputs
    assert verify_source_preparation(prepared, receipt).digest() == source.digest()
    data = {c.node_id: bytes.fromhex(c.data_hex) for c in prepared.constants}
    assert data["result"] == bits(8.0, 32).to_bytes(4, "little")
    assert data["second"] == bits(100.0, 32).to_bytes(4, "little")
    for mutate in (
        lambda r: r["stages"].reverse(),
        lambda r: r.update(before_digest="0" * 64),
        lambda r: r["policy"].update(element_budget=1),
        lambda r: r.update(limits=[{"status": "proved"}]),
    ):
        changed = copy.deepcopy(receipt)
        mutate(changed)
        with pytest.raises(ValueError):
            verify_source_preparation(prepared, changed)


def test_literal_power_checks_signature_and_receipt_bytes(helper, monkeypatch):
    monkeypatch.setenv(HELPER_ENV, str(helper.path))
    source = power(32, [bits(2.0, 32)], [bits(3.0, 32)])
    node = replace(
        source.node("result"),
        op="source_f32_composite_unqualified",
        inputs=("c1",),
        attrs=(
            ("operation", "literal_base_pow_f32"),
            ("literal_bits", "0x40000000"),
            ("numeric_semantics", "source-f32-composite-unqualified-v1"),
        ),
        index_maps=pointwise_identity_maps(1, 1),
    )
    source = replace(source, nodes=(*source.nodes[:-1], node))
    prepared, receipt = prepare_constant_math(source, helper.policy(), helper=helper)
    assert verify_constant_math(prepared, receipt, helper=helper).digest() == source.digest()
    bad = copy.deepcopy(receipt)
    bad["folds"][0]["bound_precision_counts"] = {"64": 2}
    with pytest.raises(ValueError, match="replay"):
        verify_constant_math(prepared, bad, helper=helper)
    for wrong in (
        replace(node, op="source_f32_powf_unqualified"),
        replace(node, index_maps=()),
        replace(node, attrs=(*node.attrs, ("unknown", True))),
    ):
        unchanged, evidence = prepare_constant_math(
            replace(source, nodes=(*source.nodes[:-1], wrong)), helper.policy(), helper=helper
        )
        assert not evidence["folds"] and unchanged.node("result").op == wrong.op


def test_installed_cli_builds_fresh_helper(tmp_path):
    completed = subprocess.run(
        ["merlin-targetgen", "native-math-build", "--out", str(tmp_path / "helper")], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "built"
    again = subprocess.run(
        ["merlin-targetgen", "native-math-build", "--out", str(tmp_path / "helper")], capture_output=True, text=True
    )
    assert again.returncode == 2 and json.loads(again.stdout)["status"] == "compile_error"


def test_application_intake_preserves_missing_helper_status(helper, monkeypatch):
    from merlin.semantic_compiler.target_binding import NativeCompilationError
    from merlin.targetgen import native_application as intake

    monkeypatch.delenv(HELPER_ENV, raising=False)
    source = power(32, [bits(2.0, 32)], [bits(3.0, 32)])
    staged = SimpleNamespace(
        argument_bindings=(),
        program="",
        entry="forward",
        precision_policy={"source_constant_math": helper.policy()},
        original_source_sha256="0" * 64,
    )
    monkeypatch.setattr(intake, "load_compile_inputs", lambda path: staged)
    monkeypatch.setattr(intake, "load_lifted_argument_bytes", lambda path, stage: {})
    monkeypatch.setattr(intake, "translate_linalg_text", lambda *a, **k: SimpleNamespace(request=source))
    args = SimpleNamespace(
        compile_inputs="compiler-only-inputs",
        mode="diagnostic",
        **dict.fromkeys(
            (
                "linalg_entry",
                "physical_layouts",
                "diagnostic_region",
                "diagnostic_graph_policy",
                "diagnostic_tile_index",
                "whole_contraction",
                "row_softmax_tile_index",
                "ordered_k_tile",
                "ordered_m_tile",
                "ordered_n_tile",
                "outer_tile_index",
            )
        ),
    )
    binding = SimpleNamespace(application_adapter=lambda *a, **k: None)
    with pytest.raises(NativeCompilationError) as caught:
        intake._source(args, binding, "synthetic")
    assert caught.value.status == "tool_unavailable"
