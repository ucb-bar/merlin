"""Ordinary source preparation binds current optional pointer ABI contracts."""

import json
from dataclasses import replace

import pytest
from test_dense_source import source
from test_selected_dense_contracts import inputs
from test_source_dense_lowering import run

from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.logical_borrowed_abi import LogicalBorrowedABI
from merlin.llvmlower.source_dense import SourceDenseLowering


def declaration():
    return dict(
        schema="merlin.logical_pointer_abi.v1",
        layout="dense_row_major",
        alignment_bytes=[4, 8, 16],
        access=["read_only", "read_only", "fully_written"],
        bounds="logical_elements",
        completion="synchronous",
    )


def selection(tmp_path, prepare=None, **controls):
    selected, _, storage, adapter = inputs(
        tmp_path,
        mutate=lambda f, s, d: d.update(logical_pointer_abi=declaration()),
    )
    effects = BorrowedBufferEffects("independently-reviewed-test-primitive-v1", True, True, True)
    cap = LogicalBorrowedABI(selected.dense.logical_pointer_abi, effects, "reviewed-test-logical-format-v1", 4096)
    seen = []

    def implementation(bindings, directory):
        seen.extend(bindings)
        if prepare is not None:
            prepare(bindings, directory)

    def build(bindings, context):
        pytest.fail("source-preparation test must never compile a provider")

    config = SourceDenseLowering(
        selected,
        storage,
        adapter,
        "host",
        64,
        80,
        implementation,
        build,
        private_ownership=True,
        external_buffers_disjoint=True,
        verification_timeout_ms=20_000,
        borrowed_effects=effects,
        logical_abi=cap,
    )
    return replace(config, **controls), seen


@pytest.mark.parametrize("borrow", [False, True])
def test_ordinary_preparation_owns_pinned_shared_bridge(tmp_path, borrow):
    config, seen = selection(tmp_path, borrow_contiguous=borrow)
    model, _ = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    result, _ = run(tmp_path, config, model)
    assert len(seen) == 1
    binding = seen[0]
    assert binding.logical_abi_source.is_file()
    assert binding.logical_kernel_symbol and binding.logical_observer_symbol is None
    bridge = binding.logical_abi_source.read_text()
    assert binding.logical_kernel_symbol in bridge
    record = json.loads((result.parent / "source_dense.json").read_text())
    assert record["bindings"][0]["logical_abi"]["borrow_contiguous"] is borrow
    binding.compilation.revalidate()


@pytest.mark.parametrize("mutation", ["bridge", "control", "effects", "format", "software"])
def test_provider_cannot_change_original_bridge_or_admission(tmp_path, mutation):
    def prepare(bindings, directory):
        if mutation == "bridge":
            path = bindings[0].logical_abi_source
            path.write_text(path.read_text() + "\nvoid undeclared_write(void) {}\n")
        elif mutation == "control":
            object.__setattr__(config, "borrow_contiguous", False)
        elif mutation == "effects":
            object.__setattr__(config.logical_abi.qualified_effects, "no_releases", False)
        elif mutation == "format":
            object.__setattr__(config.logical_abi, "format_qualification_identity", "self-endorsed")
        else:
            path = config.selected_contracts.software_path
            path.write_bytes(path.read_bytes() + b"\n")

    config, _ = selection(tmp_path, prepare)
    model, _ = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    with pytest.raises(ValueError, match="changed|replan"):
        run(tmp_path, config, model)
    assert not (tmp_path / "preparation" / "prepared_model_transform.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unknown"),
        ("alignment_bytes", [4, 3, 16]),
        ("alignment_bytes", [True, 8, 16]),
        ("alignment_bytes", [4, 8]),
        ("alignment_bytes", [4, 8, 1 << 63]),
        ("access", ["read_only"]),
        ("bounds", None),
        ("layout", ""),
        ("unknown", "field"),
    ],
)
def test_malformed_original_software_pointer_abi_refuses(tmp_path, field, value):
    row = declaration()
    row[field] = value
    with pytest.raises(ValueError, match="logical pointer ABI"):
        inputs(tmp_path, mutate=lambda f, s, d: d.update(logical_pointer_abi=row))


@pytest.mark.parametrize(
    "field,value",
    [
        ("layout", "opaque"),
        ("completion", "asynchronous"),
        ("bounds", "unknown"),
        ("access", ["read_write", "read_only", "fully_written"]),
    ],
)
def test_unknown_well_formed_software_stays_unqualified(tmp_path, field, value):
    row = declaration()
    row[field] = value
    selected, _, _, _ = inputs(tmp_path, mutate=lambda f, s, d: d.update(logical_pointer_abi=row))
    # Parsing a declared fact is distinct from granting executable support.
    assert selected.dense.logical_pointer_abi is not None
    with pytest.raises(ValueError, match="exact format/access/effects"):
        LogicalBorrowedABI(
            selected.dense.logical_pointer_abi,
            BorrowedBufferEffects("reviewed-effects-only", True, True, True),
            "format",
            4096,
        )


def test_missing_or_mismatched_original_contract_cannot_enable_borrowing(tmp_path):
    config, _ = selection(tmp_path)
    for changes in (
        {"selected_contracts": object()},
        {"borrowed_effects": None},
        {"borrow_contiguous": 1},
        {
            "logical_abi": replace(
                config.logical_abi, declaration=replace(config.logical_abi.declaration, alignment_bytes=(8, 8, 16))
            )
        },
    ):
        with pytest.raises(ValueError):
            replace(config, **changes)


def test_packing_budget_failure_retains_ordinary_unoptimized_source(tmp_path):
    config, seen = selection(tmp_path)
    config = replace(config, logical_abi=replace(config.logical_abi, max_stack_bytes=1))
    model, _ = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    result, _ = run(tmp_path, config, model)
    assert not seen
    assert result.read_bytes() == (tmp_path / "model.mlir").read_bytes()
