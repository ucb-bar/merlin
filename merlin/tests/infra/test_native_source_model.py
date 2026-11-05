"""Actual generated host glue and checked source providers in one native image.

Neutral host-domain primitives establish no accelerator or timing qualification.
"""

import ctypes
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file
from test_dense_ordinary_model import capture
from test_dense_source_provider import DenseProvider
from test_dense_source_trace import check_trace
from test_selected_dense_contracts import inputs
from test_source_model_api import reviewed_selection

from merlin.common.digest import sha256_file
from merlin.llvmlower import codegen, toolchain
from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.llvmlower.lower import lower_model_file
from merlin.runtime.host_provider import revalidate_host_provider

CASES = [((3, 7, 5), (2, 3, 4), 1, False), ((4, 9, 7), (3, 2, 5), 2, True)]


class NativeDenseProvider(DenseProvider):
    @staticmethod
    def wrapper(index, symbol, bridge, problem, *, output_width=4):
        # Private scratch has the primitive's actual C element types. Byte
        # copies may alias these objects; typed primitive loads are then legal.
        return DenseProvider.wrapper(index, symbol, bridge, problem, output_width=output_width, typed_scratch=True)

    def build(self, bindings, context):
        # Match this fixture's original independent native qualifier: do not
        # synthesize undeclared libc helpers from private byte-copy loops.
        return super().build(bindings, replace(context, compiler_flags=(*context.compiler_flags, "-fno-builtin")))


def setup(tmp_path, monkeypatch, case=CASES[0]):
    if not toolchain.available() or not toolchain.llvm_nm().is_file():
        pytest.skip("native model compiler and symbol inspector required")
    monkeypatch.setenv("MERLIN_SINK_DEALLOCS", "1")
    monkeypatch.setenv("MERLIN_SPLIT_FRESH_DEALLOCS", "0")
    monkeypatch.setenv("MERLIN_SLICE_UNUSED_ELEMENTWISE_OUTPUT", "0")
    shape, tile, width, shared = case
    facts = tmp_path / "facts"
    facts.mkdir()
    model = inputs(facts, tile=tile, widths=(width, width, 4), shared=shared, shape=shape)
    bundle = tmp_path / "capture"
    expected, trace = capture(bundle, shape, width)
    return model, bundle, expected, trace


def execute(result, bundle, expected):
    """All host arithmetic/control executes in the generated native function."""
    record = result.stats["shared_source_lowering"]
    revalidate_host_provider(record["host_provider"])
    policy = PrivateHostImagePolicy(result.workdir.resolve())
    model = HostModel.load(str(result.host_so), image_policy=policy)
    assert model.image_sha256 == sha256_file(result.host_so)
    weights = load_file(bundle / "weights.safetensors")
    with np.load(bundle / "inputs.npz") as saved:
        arguments = [weights["w0"], weights["w1"], *(saved[f"in{i}"].copy() for i in range(3))]
    originals = [a.copy() for a in arguments]
    outputs = [np.full_like(a, 93) for a in expected]
    model([(a.ctypes.data, a.shape) for a in (*arguments, *outputs)])
    for got, want in zip(outputs, expected, strict=True):
        np.testing.assert_array_equal(got, want)
    # State and position have invocation-owned writable backing; upstream
    # bufferization may reuse it. Immutable weights and input remain unchanged.
    for got, want in zip(arguments[:3], originals[:3], strict=True):
        np.testing.assert_array_equal(got, want)
    revalidate_host_provider(record["host_provider"])
    return outputs, model


@pytest.mark.parametrize("case", CASES)
def test_native_model_runs_checked_dense_and_complete_recurrent_state(tmp_path, monkeypatch, case):
    model, bundle, expected, trace = setup(tmp_path, monkeypatch, case)
    observations = []
    for arm, streaming in (("baseline", True), ("retained", False)):
        provider = NativeDenseProvider(
            tmp_path / (arm + "_provider"), model, trace, streaming=streaming, target_console=False
        )
        result = lower_model_file(
            bundle / "model.mlir", tmp_path / arm, shared_source_lowerings=(reviewed_selection(provider),)
        )
        actual, native = execute(result, bundle, expected)
        contract = result.stats["shared_source_lowering"]
        assert contract["host_provider"]["status"] == "completed"
        assert "borrowed_buffer_lifetime" in result.stats
        assert contract["host_provider"]["native_dynamic_link"]["scope"].startswith("original host link")
        recipe = json.loads(Path(contract["compilation_recipe"]["path"]).read_text())
        assert "-Wl,-z,defs" in recipe["commands"][-1]["argv"]
        effects = []
        for item in provider.records:
            symbol = item["symbol"]
            count = getattr(native.lib, symbol + "_observe_count")
            count.argtypes = [ctypes.c_int]
            count.restype = ctypes.c_uint64
            # Native qualification uses another private image. These counters
            # come from the actual linked complete-model execution above.
            assert count(5) == 3
            size = getattr(native.lib, symbol + "_trace_size")
            size.restype = ctypes.c_uint64
            query = getattr(native.lib, symbol + "_trace")
            query.argtypes = [ctypes.c_int, ctypes.c_int]
            query.restype = ctypes.c_int64
            rows = [[query(i, j) for j in range(query(i, 1) + 2)] for i in range(size())]
            assert len(rows) % 3 == 0
            one = len(rows) // 3
            for call in range(3):
                effects.append(
                    check_trace(
                        rows[call * one : (call + 1) * one],
                        item["problem"],
                        provider.storage,
                        item["result"].compiled.kernel.args,
                    )
                )
        observations.append(
            dict(
                arm=arm,
                effects=effects,
                source=contract,
                elf_bytes=result.host_so.stat().st_size,
                outputs=[a.tolist() for a in actual],
            )
        )
    assert observations[0]["outputs"] == observations[1]["outputs"]
    assert sum(r["requested_read_bytes"] for r in observations[1]["effects"]) < sum(
        r["requested_read_bytes"] for r in observations[0]["effects"]
    )
    (tmp_path / "pair.json").write_text(json.dumps(observations, indent=2))


def test_device_domain_selection_refuses_native_before_preparation(tmp_path, monkeypatch):
    model, bundle, _, trace = setup(tmp_path, monkeypatch)
    provider = NativeDenseProvider(tmp_path / "provider", model, trace, target_console=False)
    selection = replace(reviewed_selection(provider), execution_domain="device")
    with pytest.raises(ValueError, match="only host-domain"):
        lower_model_file(bundle / "model.mlir", tmp_path / "build", shared_source_lowerings=(selection,))
    assert not (tmp_path / "build").exists()


def test_unavailable_native_source_schedule_keeps_complete_cpu_fallback(tmp_path, monkeypatch):
    model, bundle, expected, trace = setup(tmp_path, monkeypatch)
    provider = NativeDenseProvider(tmp_path / "provider", model, trace, target_console=False)
    selection = replace(reviewed_selection(provider), static_budget=0)
    result = lower_model_file(bundle / "model.mlir", tmp_path / "build", shared_source_lowerings=(selection,))
    assert result.stats["shared_source_lowering"]["host_provider"] is None
    assert not provider.records
    execute(result, bundle, expected)
    recipe = result.stats["shared_source_lowering"]["compilation_recipe"]
    assert json.loads(Path(recipe["path"]).read_text())["status"] == "completed"


@pytest.mark.parametrize("mutation", ("replace_provider", "link_object", "receipt", "new_import"))
def test_native_provider_custody_refuses_before_return(tmp_path, monkeypatch, mutation):
    from merlin.llvmlower import native_source_model

    model, bundle, _, trace = setup(tmp_path, monkeypatch)
    provider = NativeDenseProvider(tmp_path / "provider", model, trace, target_console=False)
    selection = reviewed_selection(provider)
    if mutation == "replace_provider":

        def build(bindings, context):
            original = provider.build(bindings, context)
            return replace(original, prepared_sha256="0" * 64)

        selection = replace(selection, build=build)
        diagnostic = "current model identity"
    elif mutation == "link_object":
        run = codegen._run

        def changed(command, **kwargs):
            answer = run(command, **kwargs)
            if "-shared" in command:
                obj = next(p for p in command if str(p).endswith("_primitive.o"))
                with Path(obj).open("ab") as stream:
                    stream.write(b"changed after actual native link")
            return answer

        monkeypatch.setattr(codegen, "_run", changed)
        diagnostic = "import changed"
    elif mutation == "receipt":
        close = native_source_model.close_host_provider

        def changed(record, *args, **kwargs):
            close(record, *args, **kwargs)
            Path(record["receipt_path"]).write_text("{}\n")

        monkeypatch.setattr(native_source_model, "close_host_provider", changed)
        diagnostic = "receipt changed"
    else:
        from merlin.runtime.host_provider import HostProviderObject, read_host_function_abi

        def build(bindings, context):
            original = provider.build(bindings, context)
            source = context.workdir / "new_import.c"
            source.write_text('extern int puts(const char*); void extra_provider(void){puts("unexpected");}\n')
            llvm, obj = source.with_suffix(".ll"), source.with_suffix(".o")
            commands = []
            for mode, output in ((("-S", "-emit-llvm"), llvm), (("-c",), obj)):
                argv = tuple(map(str, (context.compiler, *context.compiler_flags, *mode, source, "-o", output)))
                context.compile(argv, inputs=(source,), output=output)
                commands.append(argv)
            artifact = HostProviderObject(
                obj,
                sha256_file(obj),
                llvm,
                sha256_file(llvm),
                (read_host_function_abi(llvm, "extra_provider"),),
                (read_host_function_abi(llvm, "puts"),),
                tuple((p, sha256_file(p)) for p in (source, context.compiler)),
                tuple(commands),
                Path.cwd(),
            )
            return replace(original, objects=(*original.objects, artifact))

        selection = replace(selection, build=build)
        diagnostic = "unsupported dynamic import"
    with pytest.raises(ValueError, match=diagnostic):
        lower_model_file(bundle / "model.mlir", tmp_path / "build", shared_source_lowerings=(selection,))
    recipe = tmp_path / "build" / "compilation_recipe.json"
    if recipe.exists():
        assert json.loads(recipe.read_text())["status"] != "completed"
