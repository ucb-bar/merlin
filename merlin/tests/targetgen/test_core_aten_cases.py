from __future__ import annotations

import json

import pytest

from merlin.targetgen._aten_opset_worker import core_opset
from merlin.targetgen.core_aten_capture import loader_source
from merlin.targetgen.core_aten_cases import (
    _aliases,
    _changed,
    _encode,
    _tensor_snapshot,
    case_document,
    core_aten_cases,
    corpus_document,
    decode_arguments,
    decode_value,
    resolve_overload,
)

torch = pytest.importorskip("torch")


def validate_result(case: dict, actual) -> None:
    """Eager oracle check against the comparison policy embedded in a case document."""
    comparison = case["comparison"]
    if comparison == "metadata":
        observed = _encode(actual, values=False)
        assert observed == case["expected"], f"metadata mismatch: expected {case['expected']!r}, got {observed!r}"
        return
    assert comparison == "torch_close", f"unknown Core ATen comparison policy: {comparison!r}"
    torch.testing.assert_close(actual, decode_value(case["expected"]), **(case.get("comparison_parameters") or {}))


@pytest.fixture(scope="module")
def opset() -> dict:
    return core_opset()


@pytest.fixture(scope="module")
def corpus(opset: dict) -> dict:
    return corpus_document(opset["ops"], pytorch_version=opset["torch"])


def test_registry_is_exact_bijection_with_core_tagged_overloads(opset: dict) -> None:
    assert list(core_aten_cases()) == opset["ops"]
    assert len(opset["ops"]) == 193
    assert opset["sha256"] == "2195c829a9f600e91247ee168374180c5b34367a10d5a6243723d0f75f82af65"


def test_every_case_resolves_and_executes_its_exact_overload(corpus: dict) -> None:
    assert corpus["complete"] is True
    assert corpus["case_count"] == corpus["overload_count"]
    for case in corpus["cases"]:
        overload = resolve_overload(case["overload"])
        assert str(overload) == case["overload"]
        assert str(overload._schema) == case["schema"]


def test_portable_arguments_round_trip_and_oracles_validate(corpus: dict) -> None:
    for case in corpus["cases"]:
        torch.manual_seed(case["rng_seed"])
        args, kwargs = decode_arguments(case["arguments"])
        actual = resolve_overload(case["overload"])(*args, **kwargs)
        validate_result(case, actual)


def test_corpus_is_byte_deterministic(opset: dict) -> None:
    first = corpus_document(opset["ops"], pytorch_version=opset["torch"])
    second = corpus_document(reversed(opset["ops"]), pytorch_version=opset["torch"])
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )


def test_factories_return_fresh_mutable_arguments() -> None:
    case = core_aten_cases()["aten._local_scalar_dense.default"]
    first_args, _ = case.make_arguments()
    second_args, _ = case.make_arguments()
    assert first_args[0] is not second_args[0]
    assert first_args[0].untyped_storage().data_ptr() != second_args[0].untyped_storage().data_ptr()


def test_portable_tensor_round_trip_preserves_nonzero_storage_offset() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)[1:, 1:]
    assert tensor.storage_offset() != 0
    rebuilt = decode_value(_encode(tensor))
    torch.testing.assert_close(rebuilt, tensor)
    assert rebuilt.shape == tensor.shape
    assert rebuilt.stride() == tensor.stride()
    assert rebuilt.storage_offset() == tensor.storage_offset()


def test_nonfinite_complex_and_zero_stride_nan_round_trip_as_strict_json() -> None:
    complex_document = _encode(complex(float("nan"), float("inf")))
    json.dumps(complex_document, allow_nan=False)
    rebuilt_complex = decode_value(complex_document)
    assert torch.isnan(torch.tensor(rebuilt_complex.real))
    assert rebuilt_complex.imag == float("inf")

    expanded = torch.tensor([float("nan")]).expand(3)
    expanded_document = _encode(expanded)
    json.dumps(expanded_document, allow_nan=False)
    rebuilt_expanded = decode_value(expanded_document)
    assert rebuilt_expanded.stride() == (0,)
    assert torch.isnan(rebuilt_expanded).all()


def test_mutation_and_alias_checks_handle_nan_signed_zero_and_empty_views() -> None:
    nan = torch.tensor(float("nan"))
    assert not _changed(_tensor_snapshot(nan), nan.clone())
    positive_zero = torch.tensor(0.0)
    negative_zero = torch.tensor(-0.0)
    assert _changed(_tensor_snapshot(positive_zero), negative_zero)

    base = torch.empty(1)
    empty_view = base[:0]
    unrelated = torch.empty(0)
    assert _aliases(base, empty_view)
    assert not _aliases(empty_view, unrelated)


def test_mutation_out_and_alias_contracts_are_recorded() -> None:
    resize = case_document(core_aten_cases()["aten.resize_.default"])
    out = case_document(core_aten_cases()["aten.atan2.out"])
    alias = case_document(core_aten_cases()["aten.alias.default"])
    assert resize["mutated_arguments"] == ["args[0]"]
    assert "kwargs.out" in out["mutated_arguments"]
    assert "result->kwargs.out" in out["output_input_aliases"]
    assert "result->args[0]" in alias["output_input_aliases"]


def test_backward_rng_symbolic_and_overload_families_are_not_collapsed(corpus: dict) -> None:
    names = {case["overload"] for case in corpus["cases"]}
    assert {
        "aten.convolution_backward.default",
        "aten.rand.default",
        "aten.sym_size.int",
        "aten.add.Tensor",
        "aten.add.Scalar",
    } <= names
    assert len(names) == corpus["case_count"]


@pytest.mark.parametrize(
    "name",
    [
        "aten.add.Tensor",
        "aten.arange.start_step",
        "aten.atan2.out",
        "aten.index.Tensor",
    ],
)
def test_compiler_adapter_loader_is_self_contained_and_preserves_oracle(name: str) -> None:
    case = case_document(core_aten_cases()[name])
    namespace: dict = {}
    exec(compile(loader_source(case), f"<{name}>", "exec"), namespace)  # noqa: S102
    model, inputs = namespace["get_model_and_inputs"]()
    torch.manual_seed(case["rng_seed"])
    validate_result(case, model(*inputs))


@pytest.mark.parametrize("layout", ["transposed", "padded", "sliced_offset", "zero_stride"])
def test_standalone_capture_loader_preserves_original_argument_storage(layout):
    torch = pytest.importorskip("torch")
    from merlin.targetgen.core_aten_cases import case_document_from_arguments

    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    tensor = {
        "transposed": values.T,
        "padded": torch.arange(24, dtype=torch.float32).reshape(3, 8)[:, ::2],
        "sliced_offset": torch.arange(28, dtype=torch.float32).reshape(7, 4)[1:4],
        "zero_stride": values[:1].expand(3, 4),
    }[layout]
    case = case_document_from_arguments("aten.clone.default", (tensor,), {})
    namespace = {}
    exec(compile(loader_source(case), "<standalone storage loader>", "exec"), namespace)  # noqa: S102
    model, (actual,) = namespace["get_model_and_inputs"]()
    assert actual.stride() == tensor.stride()
    assert actual.storage_offset() == tensor.storage_offset()
    assert torch.equal(actual, tensor)
    assert torch.equal(model(actual), tensor.clone())


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("value", [2.123456789 + 4.987654321j, complex(-0.0, -0.0), complex(float("nan"), 3.0)])
def test_standalone_scalar_loader_preserves_complex_components(dtype, value):
    from merlin.targetgen.core_aten_cases import case_document_from_arguments

    tensor = torch.tensor(value, dtype=dtype)
    case = case_document_from_arguments("aten._local_scalar_dense.default", (tensor,), {})
    namespace = {}
    exec(compile(loader_source(case), "<complex scalar loader>", "exec"), namespace)  # noqa: S102
    model, inputs = namespace["get_model_and_inputs"]()
    actual = model(*inputs)
    assert actual.dtype == dtype and actual.shape == torch.Size([])
    torch.testing.assert_close(actual, tensor, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(torch.view_as_real(actual)), torch.signbit(torch.view_as_real(tensor)))
