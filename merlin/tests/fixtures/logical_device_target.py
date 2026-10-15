"""Explicit synthetic device capabilities for shared host routing and ABI tests.

These tests execute ordinary CPU reference kernels. Their chosen tile size and
readout are test data, not claims about an accelerator or a qualified deployment.
No simulator, external RTL checkout or support implementation is consulted.
"""

from fake_quant_layer import Oracle


def require_native_tools(*, device=False, host=False):
    """Admit CPU execution only with its actual tools; a broken selection fails."""
    import shutil

    import pytest

    from merlin.llvmlower import toolchain

    tools = {"clang": toolchain.clang()}
    selection_keys = ["MERLIN_CLANG", "MERLIN_MLIR_INSTALL", "MERLIN_IREE_BIN", "MERLIN_EXT_MERLIN_IREE"]
    if device or host:
        tools["mlir-translate"] = toolchain.mlir_translate()
        selection_keys.append("MERLIN_MLIR_TRANSLATE")
    if host:
        driver = toolchain._env("MERLIN_COMPILER_DRIVER", "python")
        selection_keys.append("MERLIN_COMPILER_DRIVER")
        if driver == "stock-cli":
            tools["mlir-opt"] = toolchain.llvm_install() / "bin" / "mlir-opt"
        else:
            tools["compiler Python"] = toolchain.compiler_python()
            selection_keys.extend(["MERLIN_COMPILER_PYTHON", "MERLIN_COMPILER_VENV", "MERLIN_M2M_VENV"])
    missing = [name for name, tool in tools.items() if shutil.which(str(tool)) is None]
    if missing:
        reason = "native CPU test requires " + ", ".join(missing)
        if any(toolchain._env(key) for key in selection_keys):
            pytest.fail("selected " + reason, pytrace=False)
        pytest.skip(reason)


def install(monkeypatch):
    from merlin.compile import mesh
    from merlin.system import offload
    from merlin.targetgen import readout_facet
    from merlin.targetgen.corpus_spec import CorpusBinding
    from merlin.xdsl_dialects.lowering import compute_groups

    readouts = [
        {"selector": "i8", "applies": ["bias_add", "acc_scale", "relu", "maxpool"]},
        {"selector": "i32", "applies": []},
    ]

    def facet(target):
        return readout_facet.ReadoutFacet(
            target=target,
            unit="unit0",
            element_dtype="i8",
            accumulator_dtype="i32",
            scale_granularities=("tensor",),
            readouts=tuple(readouts),
            operand_sum={"operands": 2, "operand_dtype": "i8", "operand_saturates": True},
        )

    def oracle(target, **_kwargs):
        result = Oracle(families=("contraction", "elementwise_map", "reduction"))
        result.target = target
        result.readout = readout_facet.TargetReadout((facet(target),))
        return result

    def binding(target, operand_dtype, accum_dtype, requant_output_dtype=None, **_kwargs):
        return CorpusBinding(
            target=target,
            tile_dim=16,
            operand_dtype=operand_dtype or "i8",
            accum_dtype=accum_dtype or "i32",
            integer=True,
            tiers=[],
            compare="exact_int",
            requant_output_dtype=requant_output_dtype,
            readout_inputs={"readouts": readouts},
        )

    monkeypatch.setattr(compute_groups, "TargetOracle", oracle)
    monkeypatch.setattr(mesh, "_mesh_tile_binding", binding)
    monkeypatch.setattr(offload, "device_dtype_triples", lambda _target: (("i8", "i8", "i32"),))
    monkeypatch.setattr(readout_facet, "for_target", lambda target: (facet(target),))
    monkeypatch.setattr(readout_facet, "epilogue_readouts", lambda _target: readout_facet.validated_readouts(readouts))
