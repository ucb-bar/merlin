"""Ordinary dense source routing uses checked current semantics and native glue.

The execution target is the existing neutral synchronous byte machine. These
checks supply no executable support, timing or numerical policy for hardware.
"""

import ctypes
import json
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_dense_source import source as dense_source
from test_dense_source_catalog import implementation
from test_pointwise_source_provider import Descriptor, Provider
from test_selected_dense_contracts import inputs

from merlin.common.digest import sha256_file
from merlin.compile.scheduling.block_schedule import Compute
from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract
from merlin.llvmlower.prepared_model_transform import apply_prepared_model_transform
from merlin.llvmlower.shared_source_routing import DenseSourceBackend, DenseSourceLimits, SharedSourceRouting
from merlin.runtime.host_provider import prepare_host_provider
from merlin.transforms.certificates import ir_text
from merlin.transforms.dense_planning import DenseResources


class DenseProvider:
    """Target test implementation only; production owns all source matching."""

    build = Provider.build

    def __init__(self, work, *, shape=(5, 14, 13), widths=(1, 1, 4), streaming=False):
        work.mkdir()
        selection = work / "selection"
        selection.mkdir()
        self.model = inputs(selection, shape=shape, widths=widths)
        self.contracts, _, self.storage, self.adapter = self.model
        self.work, self.streaming = work, streaming
        self.results, self.records, self.fragments = [], [], []
        self.proposals = 0
        self.disable = self.bad_output = False
        self.inject = None
        self.c_source = None

    def verify_selection(self):
        self.contracts.revalidate()

    def propose(self, problem, _operation, _directory):
        self.proposals += 1
        if self.disable:
            return None
        assert (problem.tile_m, problem.tile_k, problem.tile_n) == self.contracts.dense.primitive
        options = dict(storage=self.storage, adapter=self.adapter)
        if self.streaming:
            options["resources"] = DenseResources(
                1 << 20, 1 << 20, 1, 1, 2, partial_output_tiles=True, partial_output_blocks=True
            )
        if self.inject == "source":
            options["selected_contracts"] = self.contracts
        elif self.inject == "verification":
            options["verification_timeout_ms"] = 20_001
        elif self.inject == "runtime":
            options["external_buffers_disjoint"] = True
        elif self.inject == "instructions":
            encode = self.adapter.encode

            def incorrect(operation):
                call = encode(operation)
                return replace(call, instr="not_the_selected_instruction") if type(operation) is Compute else call

            options["adapter"] = replace(self.adapter, encode=incorrect)
        return options

    def emit(self, checked, symbol, problem, _directory):
        text = implementation(
            checked,
            problem,
            self.storage,
            self.contracts.dense.dtypes,
            symbol,
            lose_continuation=self.bad_output,
        )
        # Keep only the exact ABI bridge and declared counter entry points
        # public. The neutral inner loop is not a second imported ABI surface.
        for result, suffix in (
            ("uint64_t", "_count"),
            ("void", "_reset"),
            ("int", "_guards"),
            ("void", ""),
        ):
            text = text.replace(f"{result} {symbol}{suffix}(", f"static {result} {symbol}{suffix}(")
        marker = "typedef struct {void *allocated,*aligned;int64_t offset,sizes[2],strides[2];} neutral_descriptor;"
        text = text[: text.index(marker)]
        x, y, _ = self.contracts.dense.dtypes
        types = dict(i8="int8_t", i16="int16_t", i32="int32_t")
        m, k, n = problem.m, problem.k, problem.n
        bridge = "_mlir_ciface_borrowed_" + symbol
        text += f"""
void {bridge}(const neutral_descriptor *a,const neutral_descriptor *b,neutral_descriptor *c) {{
 {types[x]} lhs[{m * k}]; {types[y]} rhs[{k * n}]; int32_t out[{m * n}];
 if(a->sizes[0]!={m}||a->sizes[1]!={k}||b->sizes[0]!={k}||b->sizes[1]!={n}||
    c->sizes[0]!={m}||c->sizes[1]!={n})__builtin_trap();
 for(int i=0;i<{m};i++)for(int j=0;j<{k};j++)
  lhs[i*{k}+j]=(({types[x]}*)a->aligned)[a->offset+i*a->strides[0]+j*a->strides[1]];
 for(int i=0;i<{k};i++)for(int j=0;j<{n};j++)
  rhs[i*{n}+j]=(({types[y]}*)b->aligned)[b->offset+i*b->strides[0]+j*b->strides[1]];
 {symbol}(lhs,rhs,out);
 for(int i=0;i<{m};i++)for(int j=0;j<{n};j++)
  ((int32_t*)c->aligned)[c->offset+i*c->strides[0]+j*c->strides[1]]=out[i*{n}+j];
}}
"""
        private_names = (
            "bank_0",
            "bank_1",
            "bank_2",
            "command_count",
            "load_count",
            "store_count",
            "weight_row",
            "sum_row",
            "continuation",
            "columns",
            "put",
            "get",
            "do_lhs_load",
            "do_weight_load",
            "do_preload",
            "do_compute",
            "do_store",
        )
        text = re.sub(r"\b(" + "|".join(private_names) + r")\b", lambda match: symbol + "_" + match[0], text)
        self.fragments.append(text)
        self.results.append(checked)
        self.records.append(dict(symbol=bridge, shape=[m, k, n]))
        return FreshTensorWriterContract(symbol, 2, (2,), "borrowed_" + symbol, 32)

    def finalize(self, results, _output):
        assert tuple(self.results) == results and results
        self.c_source = self.work / "provider.c"
        symbols = [row["symbol"].removeprefix("_mlir_ciface_borrowed_") for row in self.records]
        counts = "+".join(f"{symbol}_count(index)" for symbol in symbols)
        guards = "&&".join(f"{symbol}_guards()" for symbol in symbols)
        resets = "".join(f"{symbol}_reset();" for symbol in symbols)
        self.c_source.write_text(
            "#include <stdint.h>\n"
            "typedef struct {void *allocated,*aligned;int64_t offset,sizes[2],strides[2];} neutral_descriptor;\n"
            + "\n".join(self.fragments)
            + f"\nuint64_t pointwise_count(int index) {{return index<3?({counts}):({guards});}}\n"
            + f"void pointwise_reset(void) {{{resets}}}\n"
        )


def qualify(provider, _results, _output):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native source/ABI qualifier unavailable")
    output = provider.work / "provider.so"
    subprocess.run(
        [compiler, "-shared", "-fPIC", "-O2", str(provider.c_source), "-o", str(output)],
        check=True,
        capture_output=True,
    )
    pinned = provider.work / f"provider-{sha256_file(output)}.so"
    output.replace(pinned)
    native = ctypes.CDLL(str(pinned))
    native.pointwise_count.restype = ctypes.c_uint64
    native.pointwise_reset()
    rng = np.random.default_rng(801)
    for record in provider.records:
        operation = getattr(native, record["symbol"])
        operation.argtypes = [ctypes.POINTER(Descriptor)] * 3
        m, k, n = record["shape"]
        dtypes = tuple(np.dtype({"i8": "i1", "i16": "<i2", "i32": "<i4"}[t]) for t in provider.contracts.dense.dtypes)
        for repeat in range(3):
            arrays = [
                rng.integers(-128, 128, shape).astype(dtype)
                for shape, dtype in zip(((m, k), (k, n)), dtypes[:2], strict=True)
            ]
            expected = (arrays[0].astype(np.int64) @ arrays[1].astype(np.int64)).astype(dtypes[-1])
            buffers, descriptors, originals = [], [], []
            for array in (*arrays, np.zeros((m, n), dtypes[-1])):
                rows, columns = array.shape
                memory = np.full((rows + 2, columns + 3), 71 + repeat, array.dtype)
                memory[1 : rows + 1, 1 : columns + 1] = array
                buffers.append(memory)
                originals.append(memory.copy())
                descriptors.append(
                    Descriptor(
                        memory.ctypes.data,
                        memory.ctypes.data,
                        columns + 4,
                        (ctypes.c_int64 * 2)(rows, columns),
                        (ctypes.c_int64 * 2)(columns + 3, 1),
                    )
                )
            operation(*(ctypes.byref(item) for item in descriptors))
            originals[-1][1 : m + 1, 1 : n + 1] = expected
            for observed, wanted in zip(buffers, originals, strict=True):
                np.testing.assert_array_equal(observed, wanted)
            assert native.pointwise_count(3) == 1
    receipt = dict(
        source_sha256=sha256_file(provider.c_source),
        library_sha256=sha256_file(pinned),
        complete_outputs=True,
        guards=True,
        descriptor_packing="strided inputs and output",
        scope="independent neutral native C; no target or timing qualification",
        counters=[native.pointwise_count(i) for i in range(3)],
    )
    (provider.work / "native_qualification.json").write_text(json.dumps(receipt, indent=2))


def routing(provider, *, ownership=True):
    def independent(results, output):
        qualify(provider, results, output)

    backend = DenseSourceBackend(
        provider.contracts,
        provider.propose,
        provider.emit,
        provider.finalize,
        provider.build,
        provider.verify_selection,
        limits=DenseSourceLimits(verification_timeout_ms=20_000),
        external_buffers_disjoint=ownership,
    )
    return SharedSourceRouting(backend, qualify=independent)


def prepare(tmp_path, *, shape=(5, 14, 13), widths=(1, 1, 4), streaming=False, defect=None):
    provider = DenseProvider(tmp_path / "provider", shape=shape, widths=widths, streaming=streaming)
    provider.inject = defect
    provider.bad_output = defect == "wrong_native"
    route = routing(provider, ownership=defect != "missing_runtime")
    source = tmp_path / "source.mlir"
    module, _ = dense_source(shape, provider.contracts.dense.dtypes)
    source.write_text(ir_text(module))
    selected = apply_prepared_model_transform(source, tmp_path / "preparation", route.prepare)
    return route, provider, source, selected


@pytest.mark.parametrize("shape,widths", [((5, 14, 13), (1, 1, 4)), ((1, 7, 9), (2, 1, 4)), ((3, 8, 5), (2, 2, 4))])
def test_current_source_dense_runs_independent_native_program_and_complete_inverse(tmp_path, shape, widths):
    route, provider, source, selected = prepare(tmp_path, shape=shape, widths=widths)
    report = json.loads((tmp_path / "preparation/shared_source_routing.json").read_text())
    assert report["routed_count"] == 1 and report["members"][0]["family"] == "dense"
    assert report["members"][0]["shape"] == list(shape) and "complete original" in report["inverse"]
    assert source.read_bytes() != selected.read_bytes()
    assert (
        provider.results[0].operation.inputs[0].type.element_type
        == provider.results[0].source.body.block.first_op.args[0].type.element_type
    )
    route._revalidate_prepared()


@pytest.mark.parametrize("defect", ["source", "verification", "runtime", "instructions", "wrong_native"])
def test_dense_policy_or_glue_cannot_replace_checks_or_selected_effects(tmp_path, defect):
    with pytest.raises((ValueError, AssertionError), match="(?i)source|verification|ownership|instruction|mismatch"):
        prepare(tmp_path, defect=defect)
    assert not (tmp_path / "preparation/selected.mlir").exists()


def test_missing_dense_runtime_stays_on_original_source_without_provider(tmp_path):
    route, provider, source, selected = prepare(tmp_path, defect="missing_runtime")
    assert selected.read_bytes() == source.read_bytes() and not provider.results
    assert prepare_host_provider(route.build, SimpleNamespace(prepared_path=selected), inspector=None) == ((), None)
    report = json.loads((tmp_path / "preparation/shared_source_routing.json").read_text())
    assert "ownership" in report["members"][0]["reason"]


@pytest.mark.parametrize("item", ["source", "selected", "software", "kernel"])
def test_dense_prepared_owner_rejects_source_contract_or_actual_kernel_drift(tmp_path, item):
    route, provider, source, selected = prepare(tmp_path)
    if item == "kernel":
        object.__setattr__(provider.results[0].compiled.kernel, "name", "changed")
    else:
        path = {
            "source": tmp_path / "preparation/input.mlir",
            "selected": selected,
            "software": provider.contracts.software_path,
        }[item]
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed|replan"):
        route.provider_for(SimpleNamespace(prepared_path=selected))


def test_dense_route_links_and_runs_complete_recurrent_model_with_original_readouts(tmp_path, record_property):
    from test_pointwise_source_whole_model import capture

    from merlin.llvmlower import toolchain
    from merlin.model_check import full_build as FB
    from merlin.runtime.backends import spike, spike_model

    if not toolchain.available() or not spike.available():
        pytest.skip("ordinary native lowering and functional simulator unavailable")
    bundle = tmp_path / "capture"
    expected, _trace = capture(bundle, (3, 7, 5))
    observations = []
    for name, streaming in (("baseline", True), ("optimized", False)):
        provider = DenseProvider(tmp_path / (name + "_provider"), streaming=streaming)
        route = routing(provider)
        work = tmp_path / name
        work.mkdir()
        built = spike_model.build(
            bundle,
            work,
            arena_mb=1,
            dram_bytes=64 << 20,
            code_reserve=16 << 20,
            backend="scalar",
            full_readback=True,
            **route.build_arguments(),
            host_llvm_bitcode=True,
            cflags_override=[
                "-march=rv64gc",
                "-mabi=lp64d",
                "-mcmodel=medany",
                "-O2",
                "-ffreestanding",
                "-fno-builtin",
            ],
        )
        console, _wall = FB.run_host(built["elf"], timeout_s=30)
        actual, _metadata = FB.read(console, bundle, built["build_hash"])
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(got, want)
        report = json.loads((work / "prepared_model_transform/shared_source_routing.json").read_text())
        assert report["routed_count"] == 2 and report["fallback_count"] >= 2
        assert "complete original" in report["inverse"]
        provider_receipt = json.loads((work / "host_provider/host_provider.json").read_text())
        assert provider_receipt["status"] == "completed"
        native = json.loads((provider.work / "native_qualification.json").read_text())
        observations.append(
            dict(
                arm=name,
                native=native,
                complete_outputs=True,
                linked_elf_sha256=sha256_file(Path(built["elf"])),
                kernel_sha256=[result.compiled.kernel.digest() for result in provider.results],
                selected_library=dict(provider.results[0].source_stamp.context.__dict__),
                scope="neutral byte machine in functional Spike; no accelerator cycles",
            )
        )
    assert observations[0]["native"]["counters"][0] > observations[1]["native"]["counters"][0]
    record_property("ordinary_dense_routing_pair", json.dumps(observations))


def test_wrong_recurrence_is_refused_before_target_policy_or_emission(tmp_path):
    provider = DenseProvider(tmp_path / "provider")
    route = routing(provider)
    module, operation = dense_source((5, 14, 13), provider.contracts.dense.dtypes)
    product = next(op for op in operation.walk() if op.name == "arith.muli")
    product.operands = (product.operands[0], product.operands[0])
    source = tmp_path / "source.mlir"
    source.write_text(ir_text(module))
    selected = apply_prepared_model_transform(source, tmp_path / "preparation", route.prepare)
    assert source.read_bytes() == selected.read_bytes() and provider.proposals == 0
    report = json.loads((tmp_path / "preparation/shared_source_routing.json").read_text())
    assert "both widened inputs" in report["members"][0]["reason"]
