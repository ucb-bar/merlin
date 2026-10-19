"""Float layout propagation across a recurrent model with opaque call boundaries."""

import importlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from safetensors.numpy import save_file
from test_dense_source_catalog import compilation_memory
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.compile.baremetal_model import compile_saved_model
from merlin.llvmlower import toolchain
from merlin.llvmlower.arena_bind import _call_target, _defines, _malloc_size
from merlin.llvmlower.graph_optimizations import GraphOptimizationContract
from merlin.model_check import full_build as FB
from merlin.runtime.backends import spike, spike_model
from merlin.transforms.certificates import ir_text


def model_source(shape):
    n, h, w, c = shape
    nhwc = "x".join(map(str, shape))
    nchw = f"{n}x{c}x{h}x{w}"
    identity = "affine_map<(n,h,w,c)->(n,h,w,c)>"
    return f"""module {{
 func.func private @opaque_producer(%x:tensor<{nhwc}xf32>,%s:tensor<{nhwc}xf32>)
 ->tensor<{nhwc}xf32> {{
  %e=tensor.empty():tensor<{nhwc}xf32>
  %r=linalg.generic {{indexing_maps=[{identity},{identity},{identity}],
   iterator_types=["parallel","parallel","parallel","parallel"]}}
   ins(%x,%s:tensor<{nhwc}xf32>,tensor<{nhwc}xf32>) outs(%e:tensor<{nhwc}xf32>) {{
   ^bb0(%a:f32,%b:f32,%unused:f32): %sum=arith.addf %a,%b:f32
   linalg.yield %sum:f32
  }}->tensor<{nhwc}xf32>
  func.return %r:tensor<{nhwc}xf32>
 }}
 func.func private @opaque_consumer(%x:tensor<{nhwc}xi8>,%position:tensor<i32>)
 ->tensor<{nhwc}xf32> {{
  %factor=arith.constant 0.125:f32
  %e=tensor.empty():tensor<{nhwc}xf32>
  %r=linalg.generic {{indexing_maps=[{identity},affine_map<(n,h,w,c)->()>,{identity}],
   iterator_types=["parallel","parallel","parallel","parallel"]}}
   ins(%x,%position:tensor<{nhwc}xi8>,tensor<i32>) outs(%e:tensor<{nhwc}xf32>) {{
   ^bb0(%a:i8,%p:i32,%unused:f32): %wide=arith.extsi %a:i8 to i32
   %offset=arith.addi %wide,%p:i32
   %value=arith.sitofp %offset:i32 to f32
   %scaled=arith.mulf %value,%factor:f32
   linalg.yield %scaled:f32
  }}->tensor<{nhwc}xf32>
  func.return %r:tensor<{nhwc}xf32>
 }}
 func.func @forward(%bias:tensor<1x{c}x1x1xf32>,%input:tensor<{nhwc}xf32>,
  %initial:tensor<{nhwc}xf32>,%position:tensor<i32>)
 ->(tensor<3x{nhwc}xi8>,tensor<{nhwc}xf32>,tensor<i32>,tensor<{nhwc}xf32>,tensor<1x{c}x1x1xf32>) {{
  %zero=arith.constant 0.0:f32
  %inverse_scale=arith.constant 2.0:f32
  %zero_point=arith.constant -7.0:f32
  %qmin=arith.constant -128.0:f32
  %qmax=arith.constant 127.0:f32
  %zero8=arith.constant 0:i8
  %one=arith.constant 1:i32
  %c0=arith.constant 0:index
  %c1=arith.constant 1:index
  %c3=arith.constant 3:index
  %he=tensor.empty():tensor<3x{nhwc}xi8>
  %hz=linalg.fill ins(%zero8:i8) outs(%he:tensor<3x{nhwc}xi8>)->tensor<3x{nhwc}xi8>
  %state,%history,%last=scf.for %step=%c0 to %c3 step %c1
    iter_args(%s=%initial,%hist=%hz,%p=%position)
    ->(tensor<{nhwc}xf32>,tensor<3x{nhwc}xi8>,tensor<i32>) {{
   %produced=func.call @opaque_producer(%input,%s)
    :(tensor<{nhwc}xf32>,tensor<{nhwc}xf32>)->tensor<{nhwc}xf32>
   %te=tensor.empty():tensor<{nchw}xf32>
   %transposed=linalg.transpose ins(%produced:tensor<{nhwc}xf32>)
    outs(%te:tensor<{nchw}xf32>) permutation=[0,3,1,2]
   %be=tensor.empty():tensor<{nchw}xf32>
   %biased=linalg.generic {{indexing_maps=[affine_map<(n,c,h,w)->(n,c,h,w)>,
    affine_map<(n,c,h,w)->(0,c,0,0)>,affine_map<(n,c,h,w)->(n,c,h,w)>],
    iterator_types=["parallel","parallel","parallel","parallel"]}}
    ins(%transposed,%bias:tensor<{nchw}xf32>,tensor<1x{c}x1x1xf32>)
    outs(%be:tensor<{nchw}xf32>) {{^bb0(%v:f32,%b:f32,%unused:f32):
     %sum=arith.addf %v,%b:f32
     linalg.yield %sum:f32
    }}->tensor<{nchw}xf32>
   %re=tensor.empty():tensor<{nchw}xf32>
   %relu=linalg.generic {{indexing_maps=[affine_map<(n,c,h,w)->(n,c,h,w)>,
    affine_map<(n,c,h,w)->(n,c,h,w)>],iterator_types=["parallel","parallel","parallel","parallel"]}}
    ins(%biased:tensor<{nchw}xf32>) outs(%re:tensor<{nchw}xf32>) {{
    ^bb0(%v:f32,%unused:f32): %positive=arith.cmpf ogt,%v,%zero:f32
     %clamped=arith.select %positive,%v,%zero:f32
     linalg.yield %clamped:f32
    }}->tensor<{nchw}xf32>
   %qe=tensor.empty():tensor<{nchw}xi8>
   %quantized=linalg.generic {{indexing_maps=[affine_map<(n,c,h,w)->(n,c,h,w)>,
    affine_map<(n,c,h,w)->(n,c,h,w)>],iterator_types=["parallel","parallel","parallel","parallel"]}}
    ins(%relu:tensor<{nchw}xf32>) outs(%qe:tensor<{nchw}xi8>) {{
    ^bb0(%v:f32,%unused:i8): %scaled=arith.mulf %v,%inverse_scale:f32
     %rounded=math.roundeven %scaled:f32
     %shifted=arith.addf %rounded,%zero_point:f32
     %low=arith.maximumf %shifted,%qmin:f32
     %high=arith.minimumf %low,%qmax:f32
     %wide=arith.fptosi %high:f32 to i32
     %byte=arith.trunci %wide:i32 to i8
     linalg.yield %byte:i8
    }}->tensor<{nchw}xi8>
   %backe=tensor.empty():tensor<{nhwc}xi8>
   %back=linalg.transpose ins(%quantized:tensor<{nchw}xi8>)
    outs(%backe:tensor<{nhwc}xi8>) permutation=[0,2,3,1]
   %next=func.call @opaque_consumer(%back,%p)
    :(tensor<{nhwc}xi8>,tensor<i32>)->tensor<{nhwc}xf32>
   %hist1="tensor.insert_slice"(%back,%hist,%step) <{{
    operandSegmentSizes=array<i32:1,1,1,0,0>,static_offsets=array<i64:-9223372036854775808,0,0,0,0>,
    static_sizes=array<i64:1,{n},{h},{w},{c}>,static_strides=array<i64:1,1,1,1,1>}}>
    :(tensor<{nhwc}xi8>,tensor<3x{nhwc}xi8>,index)->tensor<3x{nhwc}xi8>
   %pv=tensor.extract %p[]:tensor<i32>
   %pn=arith.addi %pv,%one:i32
   %p1=tensor.insert %pn into %p[]:tensor<i32>
   scf.yield %next,%hist1,%p1:tensor<{nhwc}xf32>,tensor<3x{nhwc}xi8>,tensor<i32>
  }}
  func.return %history,%state,%last,%input,%bias
   :tensor<3x{nhwc}xi8>,tensor<{nhwc}xf32>,tensor<i32>,tensor<{nhwc}xf32>,tensor<1x{c}x1x1xf32>
 }}
}}"""


def source_module(shape, *, annotated=False, unknown_yield_attr=False):
    module = mq.parse(model_source(shape))
    if annotated:
        scalar_yields = [op for op in module.walk()
                         if op.name == "linalg.yield" and op.parent_op().name == "linalg.generic"]
        assert len(scalar_yields) == 5
        for index, op in enumerate(scalar_yields):
            op.attributes.update({
                "prov.source_node_ids": ArrayAttr([StringAttr(f"neutral:prepared:{index}")]),
                "prov.origin_node_ids": ArrayAttr([
                    StringAttr(f"neutral:original:{index}"), StringAttr(f"neutral:original:{index + 1}"),
                    StringAttr(f"neutral:quantized:{index}"),
                ]),
                "prov.trace_role": StringAttr("lowering"),
            })
        if unknown_yield_attr:
            scalar_yields[2].attributes["execution.mode"] = StringAttr("reverse")
    module.verify()
    return module


def yield_metadata(source):
    metadata = {}
    for op in mq.parse(source).walk():
        if op.name != "linalg.yield" or "prov.source_node_ids" not in op.attributes:
            continue
        key = tuple(value.data for value in op.attributes["prov.source_node_ids"])
        assert key not in metadata
        metadata[key] = {name: str(value) for name, value in op.attributes.items()}
    return metadata


def capture(directory, shape, *, annotated=False, unknown_yield_attr=False):
    directory.mkdir()
    c = shape[-1]
    bias = np.resize(np.array([-5.25, -0.75, -0.25, 0.0, 0.25, 0.75, 5.25], "<f4"), c).reshape(1, c, 1, 1)
    data = np.resize(np.array([-300, -1.75, -0.25, -0.0, 0.0, 0.25, 0.75, 1.25, 2.25, 63.75, 65.25, 200], "<f4"), shape)
    initial = np.resize(np.array([-1.0, -0.25, 0, 0.25, 0.5], "<f4"), shape)
    state, position, history = initial.copy(), 7, []
    for _ in range(3):
        # Independent ordered f32 reference: retain every numerical boundary.
        produced = np.add(data, state, dtype=np.float32)
        biased = np.add(produced.transpose(0, 3, 1, 2), bias, dtype=np.float32)
        relu = np.where(biased > np.float32(0), biased, np.float32(0)).astype("<f4")
        scaled = np.multiply(relu, np.float32(2), dtype=np.float32)
        rounded = np.rint(scaled)
        shifted = np.add(rounded, np.float32(-7), dtype=np.float32)
        quantized = np.clip(shifted, np.float32(-128), np.float32(127)).astype("i1").transpose(0, 2, 3, 1)
        history.append(quantized.copy())
        wide = quantized.astype(np.int32) + np.int32(position)
        state = np.multiply(wide.astype("<f4"), np.float32(0.125), dtype=np.float32)
        position += 1
    expected = (np.stack(history), state, np.array(position, "<i4"), data.copy(), bias.copy())
    save_file(dict(bias=bias), directory / "weights.safetensors")
    (directory / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": dict(kind="param", weight="bias"), "1": dict(kind="input", name="input"),
        "2": dict(kind="input", name="state"), "3": dict(kind="input", name="position"),
    }))
    np.savez(directory / "inputs.npz", in0=data, in1=initial, in2=np.array(7, "<i4"))
    (directory / "input_order.json").write_text('{"input":0,"state":1,"position":2}')
    np.save(directory / "golden.npy", expected[0])
    names = ("history", "state", "position", "input", "bias")
    np.savez(directory / "goldens.npz", **dict(zip(names, expected, strict=True)))
    (directory / "output_order.json").write_text(json.dumps(names))
    module = source_module(shape, annotated=annotated, unknown_yield_attr=unknown_yield_attr)
    # Generic syntax preserves attributed yields as valid upstream MLIR. The
    # reviewed prepared representation is derived independently before builds.
    (directory / "model.mlir").write_text(ir_text(module) if annotated else str(module))
    artifacts = {p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p)) for p in directory.iterdir()}
    (directory / "capture_receipt.json").write_text(json.dumps(dict(
        schema="m2m.capture-receipt.v1", materialized_abi=dict(complete=True), artifacts=artifacts,
    )))
    return expected


def setup(tmp_path, monkeypatch, shape, *, annotated=False, unknown_yield_attr=False):
    if not toolchain.available() or not spike.available():
        pytest.skip("native lowering and functional simulator unavailable")
    bundle = tmp_path / "capture"
    expected = capture(bundle, shape, annotated=annotated, unknown_yield_attr=unknown_yield_attr)
    host = tmp_path / "host"
    host.mkdir()
    (host / "manifest.yaml").write_text(yaml.safe_dump(dict(
        target="host", run_id="layout-model", family="scalar_linalg",
        authoring=dict(mode="deterministic_generated_from_spec", generated_by_agent=False),
        outputs=dict(knobs="knobs.yaml"),
    )))
    (host / "knobs.yaml").write_text(yaml.safe_dump(dict(
        backend="scalar", dtype_strategy="fp32",
        cflags=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
    )))
    dts = tmp_path / "host.dts"
    dts.write_text('cpu@0 { device_type = "cpu"; riscv,isa = "rv64imafdc_zicsr_zifencei"; };\n')
    catalog = tmp_path / "boards.yaml"
    catalog.write_text(yaml.safe_dump(dict(schema_version=1, boards=dict(selected=dict(
        target="neutral", dram_bytes=64 << 20, dram_base=spike_model.DRAM_BASE,
        harts=1, console="htif", flow="baremetal", code_reserve=16 << 20, host_dts_sha256=sha256_file(dts),
    )))))
    root = tmp_path / "out"
    root.mkdir()
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    monkeypatch.setenv("MERLIN_STATIC_ARENA", "")
    return expected, root, dict(
        capture=bundle, package=host, board_catalog=catalog, board="selected", dts=dts,
        target="neutral", run="spike", arena_mb=1, timeout_s=30, readback="full",
        respect_captured_quantization_scope=True, host_llvm_bitcode=True,
    )


def actual_lowering(output, receipt):
    """Inspect the actual compiler inputs/outputs, without interpreting machine code."""
    source = output / "build/lower/model.mlir"
    module = mq.parse(source.read_text())
    permutations = [op for op in module.walk() if op.name == "linalg.transpose"]
    llvm = Path(receipt["output"]["host_artifacts"]["llvm_ir"]["path"])
    assert sha256_file(llvm) == receipt["output"]["host_artifacts"]["llvm_ir"]["sha256"]
    lines = llvm.read_text().splitlines()
    functions = {name: lines[start : end + 1] for name, start, end in _defines(lines)}
    allocations = [
        dict(function=name, bytes=_malloc_size(line.strip()), instruction=line.strip())
        for name, body in functions.items() for line in body if _call_target(line.strip()) == "malloc"
    ]
    calls = [
        _call_target(line.strip()) for line in functions["forward"]
        if _call_target(line.strip()) in ("opaque_producer", "opaque_consumer")
    ]
    assert sorted(calls) == ["opaque_consumer", "opaque_producer"]
    boundaries = {name: "\n".join(functions[name]) for name in calls}
    elf = Path(receipt["output"]["elf"])
    disassembly = subprocess.check_output(
        [str(toolchain.llvm_install() / "bin/llvm-objdump"), "-d", "--no-show-raw-insn", str(elf)], text=True
    )
    (output / "model.disassembly.txt").write_text(disassembly)
    relocations = subprocess.check_output(
        [str(toolchain.llvm_install() / "bin/llvm-objdump"), "-r", str(elf.parent / "model.o")], text=True
    )
    (output / "model.relocations.txt").write_text(relocations)
    opaque_relocations = [
        line for line in relocations.splitlines() if line.split() and line.split()[-1] in calls
    ]
    # Native optimization may inline a defined function after graph checking.
    # Observe this separately; logical boundary identity is checked above.
    linked_symbols = [name for name in calls if "<" + name + ">:" in disassembly]
    return dict(
        source=dict(path=str(source), sha256=sha256_file(source)),
        llvm=dict(path=str(llvm), sha256=sha256_file(llvm)),
        live_transpose_operations=len(permutations), allocations=allocations, opaque_functions=boundaries,
        opaque_calls=calls, linked_opaque_symbols=linked_symbols, object_opaque_relocations=opaque_relocations,
        machine_boundary_scope=(
            "native inlining permitted; relocations/symbols are observations, not execution counters"
        ),
        scope="actual compiler IR/allocation sites; not arbitrary machine-code equivalence",
    )


def implementation_pins():
    modules = {
        name: importlib.import_module("merlin.llvmlower." + name)
        for name in ("graph_optimizations", "layout_propagation", "layout_conformance", "quant_round_check")
    }
    modules["pipeline_state"] = importlib.import_module("merlin.transforms.pipeline_state")
    modules["dense_float_ir"] = importlib.import_module("merlin.common.dense_float_ir")
    return {name: sha256_file(Path(module.__file__)) for name, module in modules.items()}


def reviewed_preparation(options, root):
    from merlin.llvmlower.blob_weights import requested
    from merlin.llvmlower.quant_scope import FEATURE
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering

    directory = root / "reviewed_preparation"
    directory.mkdir()
    # Use the same public preparation stage and explicitly selected controls
    # as the ordinary builder, independently before either build. In particular,
    # its portable printer preserves attributed yields with valid generic syntax.
    with requested(True):
        source, features = prepare_for_lowering(
            options["capture"] / "model.mlir", directory,
            int8_compute=False, features=frozenset({FEATURE}), harts=1,
        )
    (directory / "review.json").write_text(json.dumps(dict(
        capture_source_sha256=sha256_file(options["capture"] / "model.mlir"),
        reviewed_source=dict(path=str(source), sha256=sha256_file(source)),
        input_features=[FEATURE], resulting_features=sorted(features),
        int8_compute=False, harts=1, blob_weights=True,
    ), indent=2))
    return source


@pytest.mark.parametrize("shape", [(1, 3, 5, 7), (2, 2, 7, 5)])
def test_public_model_preserves_float_quantization_and_eliminates_layout_materializations(
    tmp_path, monkeypatch, record_property, shape, annotated=False
):
    expected, root, options = setup(tmp_path, monkeypatch, shape, annotated=annotated)
    reviewed = reviewed_preparation(options, root)
    source_sha = sha256_file(reviewed)
    observations, outputs = [], []
    for name, enabled in (("original", False), ("layout", True)):
        # Select the same host pipeline and permissions in both arms. In
        # particular, graph selection disables automatic host destination rules.
        contract = GraphOptimizationContract(
            source_sha, "forward", propagate_layout=enabled,
            independent_element_order_unobserved=True, floating_environment_unobserved_nontrapping=True,
        )
        output = root / name
        pins = implementation_pins()
        started = time.perf_counter()
        with compilation_memory() as memory:
            receipt = compile_saved_model(**options, output=output, graph_optimizations=contract)
        elapsed = time.perf_counter() - started
        assert implementation_pins() == pins
        assert receipt["status"] == "verified_complete_output"
        result = receipt["output"]
        assert sha256_file(Path(result["elf"])) == result["elf_sha256"]
        for identity in result["host_artifacts"].values():
            assert sha256_file(Path(identity["path"])) == identity["sha256"]
        judged = result["per_output"]
        assert [r["name"] for r in judged] == ["history", "state", "position", "input", "bias"]
        assert all(r["passed"] and r["status"] == "compared" and r["mismatched_elements"] == 0 for r in judged)
        console = Path(result["console"])
        assert sha256_file(console) == result["console_sha256"]
        actual, metadata = FB.read(console.read_bytes(), options["capture"], result["build_hash"])
        assert len(actual) == len(expected) == 5
        for got, want in zip(actual, expected, strict=True):
            assert got.dtype == want.dtype and got.shape == want.shape
            assert got.tobytes() == want.tobytes()
        outputs.append(actual)
        lowered = actual_lowering(output, receipt)
        graph_files = {name: output / "build/lower" / filename for name, filename in (
            ("input", "host-graph-input.mlir"), ("optimized", "host-graph-optimized.mlir"),
            ("report", "host-graph-optimization.json"),
        )}
        graph = json.loads(graph_files["report"].read_text())
        for role, key in (("input", "input"), ("selected", "optimized"), ("report", "report")):
            identity = result["host_graph_optimization"][role]
            assert Path(identity["path"]) == graph_files[key].resolve()
            assert identity["sha256"] == sha256_file(graph_files[key])
        assert graph["source_sha256"] == source_sha == sha256_file(graph_files["input"])
        assert graph_files["input"].read_bytes() == reviewed.read_bytes()
        assert graph["transformed_sha256"] == sha256_file(graph_files["optimized"])
        assert graph["implementation_sources_sha256"] == pins
        if enabled:
            assert graph["layout"]["library_sources_sha256"] == pins
            assert graph["layout"]["after_transpose_bytes"] == 0
            work = graph["layout"]["pointwise_work"]
            assert work["after_scalar_evaluations"] == work["before_scalar_evaluations"]
        else:
            assert "layout" not in graph
        opaque_sources = [{
            fn.sym_name.data: str(fn) for fn in mq.parse(graph_files[role].read_text()).body.block.ops
            if fn.name == "func.func" and fn.sym_name.data in ("opaque_producer", "opaque_consumer")
        } for role in ("input", "optimized")]
        assert len(opaque_sources[0]) == 2 and opaque_sources[0] == opaque_sources[1]
        if annotated:
            original_metadata = yield_metadata(options["capture"] / "model.mlir")
            assert len(original_metadata) == 5
            assert yield_metadata(graph_files["input"]) == original_metadata
            assert yield_metadata(graph_files["optimized"]) == original_metadata
        observations.append(dict(
            arm=name, receipt=receipt, actual_lowering=lowered, compile_and_execute_seconds=elapsed,
            implementation_sources_sha256=pins, graph_report=graph,
            graph_artifacts={k: dict(path=str(p), sha256=sha256_file(p)) for k, p in graph_files.items()},
            compilation_and_execution_memory=memory, elf_bytes=Path(result["elf"]).stat().st_size,
            functional_forward_instructions=metadata["metrics"]["cycles"],
            numerical_policy="ordered-f32; RNE value then zero-point; signed clamp and exact integer conversion",
            scope="complete neutral host model; logical materializations establish no bus traffic or latency",
        ))
    for before, after in zip(*outputs, strict=True):
        assert before.tobytes() == after.tobytes()
    before, after = observations
    b, a = before["actual_lowering"], after["actual_lowering"]
    assert b["live_transpose_operations"] == 2 and a["live_transpose_operations"] == 0
    assert b["opaque_functions"] == a["opaque_functions"] and b["opaque_calls"] == a["opaque_calls"]
    assert len(a["allocations"]) < len(b["allocations"])
    assert before["receipt"]["inputs"]["capture_tree"] == after["receipt"]["inputs"]["capture_tree"]
    assert before["implementation_sources_sha256"] == after["implementation_sources_sha256"]
    # Recurrent materialization demand, only once absence in the emitted IR is
    # established. Each eliminated permutation reads and writes its payload.
    logical = 3 * int(np.prod(shape)) * (4 + 1) * 2
    report = dict(arms=observations, eliminated_logical_transpose_read_write_bytes=logical,
                  scope="typed logical traffic; not measured cache/DMA/bus traffic")
    (root / "pair.json").write_text(json.dumps(report, indent=2))
    record_property("host_layout_model_pair", json.dumps(report))


def test_annotated_scalar_yields_preserve_complete_float_model_and_provenance(
    tmp_path, monkeypatch, record_property
):
    test_public_model_preserves_float_quantization_and_eliminates_layout_materializations(
        tmp_path, monkeypatch, record_property, (1, 3, 5, 7), annotated=True
    )


def test_unknown_executable_yield_attribute_refuses_before_simulation(tmp_path, monkeypatch):
    _, root, options = setup(
        tmp_path, monkeypatch, (1, 3, 5, 7), annotated=True, unknown_yield_attr=True
    )
    reviewed = reviewed_preparation(options, root)
    contract = GraphOptimizationContract(
        sha256_file(reviewed), "forward", propagate_layout=True,
        independent_element_order_unobserved=True, floating_environment_unobserved_nontrapping=True,
    )
    simulator_calls = []

    def forbidden(*args, **kwargs):
        simulator_calls.append((args, kwargs))
        raise AssertionError("unknown executable yield attribute reached the simulator")

    monkeypatch.setattr(spike_model, "run_raw", forbidden)
    output = root / "refused"
    with pytest.raises(ValueError, match="scalar|cost contract"):
        compile_saved_model(**options, output=output, graph_optimizations=contract)
    assert not simulator_calls
    failed = json.loads((output / "baremetal_model.json").read_text())
    assert failed["status"] == "failed"
    assert not (output / "console.bin").exists()
    actual = output / "build/lower/host-graph-input.mlir"
    assert actual.read_bytes() == reviewed.read_bytes()
    assert "execution.mode" in actual.read_text()
