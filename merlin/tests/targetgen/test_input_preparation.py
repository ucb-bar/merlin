"""Input-copy audit context comes from admitted source and the ordinary logical ABI."""

import hashlib
from types import SimpleNamespace

import pytest

from merlin.targetgen.contract.interface_emit import emit_interface_mlir
from merlin.targetgen.input_preparation import source_convolution


@pytest.fixture
def capsule(tmp_path, monkeypatch):
    from merlin.llvmlower import device_shim
    from merlin.runtime.backends import base
    from merlin.targetgen.contract.build_recipe import KernelStackFramePolicy

    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _: SimpleNamespace(version=2, symbol="logical_entry"))
    monkeypatch.setattr(
        base,
        "harness_build_recipe",
        lambda target: SimpleNamespace(require_kernel_stack_frame=lambda: KernelStackFramePolicy("unit_entry", 4096)),
    )
    command = lambda opcode, operands, attributes={}: {  # noqa: E731
        "opcode": opcode,
        "operands": operands,
        "attributes": attributes,
    }
    buffer = {
        "abi_version": "0.1",
        "target": "synthetic",
        "tensors": {
            "image": {"role": "input", "shape": [1, 4, 4, 1], "dtype": "i8"},
            "weight": {"role": "weight", "shape": [9, 2], "dtype": "i8"},
            "output": {"role": "output", "shape": [16, 2], "dtype": "i32"},
        },
        "commands": [
            command("RES_PACK", {"src": "weight", "dst": "resident"}, {"layout": "packed_conv_rhs"}),
            command(
                "CONV2D",
                {"ifm": "image", "weight": "resident", "dst": "output"},
                {
                    "kernel": [3, 3, 1, 2],
                    "stride": [1, 1],
                    "padding": [1, 1, 1, 1],
                    "dilation": [1, 1],
                    "layout": "nhwc",
                    "epilogue": [],
                    "output_dtype": "i32",
                },
            ),
            command("EVICT", {"handle": "resident"}),
        ],
    }
    program = emit_interface_mlir(buffer)
    (tmp_path / "source.mlir").write_text(program)
    return {
        "operation": {"op": "conv2d"},
        "__dir__": str(tmp_path),
        "interface_mlir": "source.mlir",
        "software_screen": {"status": "admitted", "program_sha256": hashlib.sha256(program.encode()).hexdigest()},
    }


def test_original_logical_types_and_order_bind_the_copy_proof(capsule):
    context = source_convolution(capsule, target="synthetic")
    assert context["entry_symbol"] == "unit_entry"
    assert context["call_buffers"] == [
        {"role": "input_0", "shape": [1, 4, 4, 1], "dtype": "i8"},
        {"role": "weight_0", "shape": [9, 2], "dtype": "i8"},
        {"role": "out_0", "shape": [16, 2], "dtype": "i32"},
    ]


def test_selected_executable_entry_overrides_only_logical_symbol(capsule, monkeypatch):
    from merlin.runtime.backends import base
    from merlin.targetgen.contract.build_recipe import KernelStackFramePolicy

    # Neither a candidate spelling nor a target-derived logical name can select
    # the executable entry; changing the original recipe changes that selection.
    capsule["entry_symbol"] = "candidate_choice"
    first = source_convolution(capsule, target="synthetic")
    monkeypatch.setattr(
        base,
        "harness_build_recipe",
        lambda target: SimpleNamespace(
            require_kernel_stack_frame=lambda: KernelStackFramePolicy("selected_entry", 4096)
        ),
    )
    second = source_convolution(capsule, target="synthetic")
    assert first["entry_symbol"] == "unit_entry"
    assert second == {**first, "entry_symbol": "selected_entry"}


@pytest.mark.parametrize("failure", [KeyError("absent"), NotImplementedError("unavailable"), ValueError("unbound")])
def test_missing_selected_entry_cannot_fall_back_to_logical_name(capsule, monkeypatch, failure):
    from merlin.runtime.backends import base

    def absent(target):
        raise failure

    monkeypatch.setattr(base, "harness_build_recipe", absent)
    with pytest.raises(ValueError):
        source_convolution(capsule, target="synthetic")


@pytest.mark.parametrize("change", ["digest", "admission", "escape", "target"])
def test_unbound_sources_cannot_select_input_preparation(capsule, change):
    if change == "digest":
        capsule["software_screen"]["program_sha256"] = "0" * 64
    elif change == "admission":
        capsule["software_screen"]["status"] = "unknown"
    elif change == "escape":
        capsule["interface_mlir"] = "../source.mlir"
    if change == "target":
        assert source_convolution(capsule, target="other") is None
    else:
        with pytest.raises(ValueError):
            source_convolution(capsule, target="synthetic")


def test_other_pointer_conventions_do_not_inherit_logical_copy_permission(capsule, monkeypatch):
    from merlin.llvmlower import device_shim
    from merlin.targetgen.contract import harness_render

    logical = harness_render.logical_interface
    monkeypatch.setattr(harness_render, "logical_interface", lambda *args: list(reversed(logical(*args))))
    assert source_convolution(capsule, target="synthetic") is None
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _: SimpleNamespace(version=1))
    assert source_convolution(capsule, target="synthetic") is None
