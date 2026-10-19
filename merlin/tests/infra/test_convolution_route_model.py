"""Bounded ordinary convolution preparation, complete recurrence and real requests.

The source retains explicit padding. Host preparation is not a device gather,
cache/bus measurement or latency prediction.
"""

import json

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_dense_source import contraction

from merlin.common.digest import sha256_file
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.certificates import ir_text

CASES = (
    dict(
        shape=(1, 2, 6, 8),
        outputs=3,
        kernel=(2, 3),
        stride=(2, 1),
        dilation=(1, 2),
        padding=(1, 2, 0, 1),
        output=(3, 7),
        width=1,
        budget=60,
        tile=(2, 3, 4),
        shared=False,
    ),
    dict(
        shape=(2, 3, 3, 22),
        outputs=5,
        kernel=(3, 2),
        stride=(1, 2),
        dilation=(2, 1),
        padding=(2, 0, 1, 1),
        output=(2, 11),
        width=2,
        budget=180,
        tile=(3, 4, 5),
        shared=True,
    ),
)


def wrap(value, bits):
    return (int(value) + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def reference(image, weights, geometry, initial=0, position=7):
    """Independent original ci,kh,kw MAC order, including actual padding zeros."""
    batch, channels, height, width = image.shape
    features, _, kh, kw = weights.shape
    oh, ow = geometry["output"]
    sh, sw = geometry["stride"]
    dh, dw = geometry["dilation"]
    top, left, _, _ = geometry["padding"]
    state, history, epochs, overflow = initial, [], [], False
    for _ in range(3):
        mixed = np.empty_like(image)
        for index in np.ndindex(image.shape):
            mixed[index] = wrap(int(image[index]) + state, 8 * image.dtype.itemsize)
        output = np.empty((features, batch * oh * ow), dtype="<i4")
        for n in range(batch):
            for f in range(features):
                for h in range(oh):
                    for w in range(ow):
                        value, unwrapped = 0, 0
                        for ci in range(channels):
                            for r in range(kh):
                                for c in range(kw):
                                    y, x = h * sh + r * dh - top, w * sw + c * dw - left
                                    a = int(mixed[n, ci, y, x]) if 0 <= y < height and 0 <= x < width else 0
                                    product = a * int(weights[f, ci, r, c])
                                    unwrapped += product
                                    value = wrap(value + product, 32)
                        overflow |= not -(1 << 31) <= unwrapped < (1 << 31)
                        output[f, (n * oh + h) * ow + w] = value
        epochs.append((mixed.copy(), weights.copy(), output.copy()))
        history.append(output)
        state = wrap(state + int(output[0, 0]) + position, 32)
        position = wrap(position + 1, 32)
    return (
        (np.stack(history), np.array(state, "<i4"), np.array(position, "<i4"), image.copy(), weights.copy()),
        epochs,
        overflow,
    )


def model_source(g):
    n, ci, h, w = g["shape"]
    f, (kh, kw), (oh, ow) = g["outputs"], g["kernel"], g["output"]
    sh, sw = g["stride"]
    dh, dw = g["dilation"]
    top, left, bottom, right = g["padding"]
    ph, pw, k, positions = h + top + bottom, w + left + right, ci * kh * kw, n * oh * ow
    dt = "i8" if g["width"] == 1 else "i16"
    image = f"tensor<{n}x{ci}x{h}x{w}x{dt}>"
    padded = f"tensor<{n}x{ci}x{ph}x{pw}x{dt}>"
    weight = f"tensor<{f}x{ci}x{kh}x{kw}x{dt}>"
    gather = f"tensor<{ci}x{kh}x{kw}x{n}x{oh}x{ow}x{dt}>"
    result = f"tensor<{f}x{positions}xi32>"
    history = f"tensor<3x{f}x{positions}xi32>"
    dense = contraction((f, k, positions), (dt, dt, "i32"), lhs="weights", rhs="columns", output="conv")
    return f"""module {{
func.func @forward(%weight:{weight},%input:{image},%initial:tensor<i32>,%position:tensor<i32>)
 ->({history},tensor<i32>,tensor<i32>,{image},{weight}) {{
%zero=arith.constant 0:{dt}
%zero32=arith.constant 0:i32
%one=arith.constant 1:i32
%c0=arith.constant 0:index
%c1=arith.constant 1:index
%c3=arith.constant 3:index
%weights=tensor.collapse_shape %weight [[0],[1,2,3]]:{weight} into tensor<{f}x{k}x{dt}>
%he=tensor.empty():{history}
%hz=linalg.fill ins(%zero32:i32) outs(%he:{history})->{history}
%state,%history,%last=scf.for %step=%c0 to %c3 step %c1
 iter_args(%s=%initial,%h=%hz,%p=%position)->(tensor<i32>,{history},tensor<i32>) {{
%me=tensor.empty():{image}
%mixed=linalg.generic {{indexing_maps=[affine_map<(a,b,c,d)->(a,b,c,d)>,affine_map<(a,b,c,d)->()>,
 affine_map<(a,b,c,d)->(a,b,c,d)>],iterator_types=["parallel","parallel","parallel","parallel"]}}
 ins(%input,%s:{image},tensor<i32>) outs(%me:{image}) {{^bb0(%x:{dt},%sv:i32,%old:{dt}):
 %small=arith.trunci %sv:i32 to {dt}
 %added=arith.addi %x,%small:{dt}
 linalg.yield %added:{dt}
}}->{image}
%pe=tensor.empty():{padded}
%pz=linalg.fill ins(%zero:{dt}) outs(%pe:{padded})->{padded}
%padded="tensor.insert_slice"(%mixed,%pz) <{{operandSegmentSizes=array<i32:1,1,0,0,0>,
 static_offsets=array<i64:0,0,{top},{left}>,static_sizes=array<i64:{n},{ci},{h},{w}>,
 static_strides=array<i64:1,1,1,1>}}> :({image},{padded})->{padded}
%ge=tensor.empty():{gather}
%g=linalg.generic {{indexing_maps=[affine_map<(a,b,c,d,e,f)->(d,a,e*{sh}+b*{dh},f*{sw}+c*{dw})>,
 affine_map<(a,b,c,d,e,f)->(a,b,c,d,e,f)>],
 iterator_types=["parallel","parallel","parallel","parallel","parallel","parallel"]}}
 ins(%padded:{padded}) outs(%ge:{gather}) {{^bb0(%v:{dt},%unused:{dt}):
 linalg.yield %v:{dt}
}}->{gather}
%flat=tensor.collapse_shape %g [[0,1,2,3,4,5]]:{gather} into tensor<{k * positions}x{dt}>
%columns=tensor.expand_shape %flat [[0,1]] output_shape [{k},{positions}]
 :tensor<{k * positions}x{dt}> into tensor<{k}x{positions}x{dt}>
{dense}
%h1="tensor.insert_slice"(%conv,%h,%step) <{{operandSegmentSizes=array<i32:1,1,1,0,0>,
 static_offsets=array<i64:-9223372036854775808,0,0>,static_sizes=array<i64:1,{f},{positions}>,
 static_strides=array<i64:1,1,1>}}> :({result},{history},index)->{history}
%chosen=tensor.extract %conv[%c0,%c0]:{result}
%oldstate=tensor.extract %s[]:tensor<i32>
%pv=tensor.extract %p[]:tensor<i32>
%sumstate=arith.addi %oldstate,%chosen:i32
%sv1=arith.addi %sumstate,%pv:i32
%s1=tensor.insert %sv1 into %s[]:tensor<i32>
%pn=arith.addi %pv,%one:i32
%p1=tensor.insert %pn into %p[]:tensor<i32>
scf.yield %s1,%h1,%p1:tensor<i32>,{history},tensor<i32>
}}
func.return %history,%state,%last,%input,%weight:{history},tensor<i32>,tensor<i32>,{image},{weight}
}}
}}"""


def capture(directory, geometry):
    directory.mkdir()
    dtype = "i1" if geometry["width"] == 1 else "<i2"
    image = np.resize(np.array([-128, 127, 91, -1, 0, 1] if dtype == "i1" else [32767], dtype=dtype), geometry["shape"])
    weight_shape = (geometry["outputs"], geometry["shape"][1], *geometry["kernel"])
    weights = np.resize(np.array([-3, 2, 1, -1, 4] if dtype == "i1" else [32767, -32768], dtype=dtype), weight_shape)
    if geometry["width"] == 2:
        weights[0, ...] = 32767
    expected, epochs, overflow = reference(image, weights, geometry)
    assert overflow == (geometry["width"] == 2)
    save_file({"weight": weights}, directory / "weights.safetensors")
    (directory / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": dict(kind="param", weight="weight"),
                "1": dict(kind="input", name="input"),
                "2": dict(kind="input", name="state"),
                "3": dict(kind="input", name="position"),
            }
        )
    )
    np.savez(directory / "inputs.npz", in0=image, in1=np.array(0, "<i4"), in2=np.array(7, "<i4"))
    (directory / "input_order.json").write_text(json.dumps(dict(input=0, state=1, position=2)))
    np.save(directory / "golden.npy", expected[0])
    names = ("history", "state", "position", "input", "weight")
    np.savez(directory / "goldens.npz", **dict(zip(names, expected, strict=True)))
    (directory / "output_order.json").write_text(json.dumps(names))
    module = parse_mlir_text(model_source(geometry))
    for op in module.walk():
        op.attributes.pop("prov.region_id", None)
    module.verify()
    (directory / "model.mlir").write_text(ir_text(module))
    (directory / "capture_receipt.json").write_text(
        json.dumps(
            dict(
                schema="m2m.capture-receipt.v1",
                materialized_abi=dict(complete=True),
                artifacts={p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p)) for p in directory.iterdir()},
            )
        )
    )
    return expected, epochs


@pytest.mark.parametrize("geometry", CASES)
def test_convolution_capture_has_defined_padding_and_ordered_overflow_reference(tmp_path, geometry):
    from types import SimpleNamespace

    from test_convolution_route_provider import ConvolutionProvider
    from test_dense_ordinary_model import product

    expected, epochs = capture(tmp_path / "capture", geometry)
    assert len(expected) == 5 and len(epochs) == 3
    assert not np.array_equal(epochs[0][0], epochs[1][0])
    module = parse_mlir_text((tmp_path / "capture/model.mlir").read_text())
    module.verify()
    assert sum(op.name == "scf.for" for op in module.walk()) == 1
    assert sum(op.name == "linalg.fill" for op in module.walk()) == 3
    for image, weights, wanted in epochs:
        columns = ConvolutionProvider.columns(SimpleNamespace(geometry=geometry), image)
        actual = product(weights.reshape(weights.shape[0], -1), columns)
        np.testing.assert_array_equal(actual, wanted)


def observe_calls(console, provider, native):
    from test_dense_source_trace import check_trace

    pending, calls, descriptors = {}, [], []
    for line in console.splitlines():
        if line.startswith(b"MERLIN_CONV_PANEL "):
            index, pointer, rows, cols, stride0, stride1 = map(int, line.split()[1:])
            assert pointer > 0 and stride0 == cols and stride1 == 1
            descriptors.append(dict(index=index, pointer=pointer, shape=[rows, cols]))
        elif line.startswith(b"MERLIN_DENSE_TRACE "):
            index, *row = map(int, line.split()[1:])
            pending.setdefault(index, []).append(row)
        elif line.startswith(b"MERLIN_DENSE_COUNTS "):
            index, *counts = map(int, line.split()[1:])
            record = provider.records[index]
            trace = pending.pop(index)
            effects = check_trace(trace, record["problem"], provider.storage, record["result"].compiled.kernel.args)
            calls.append(dict(index=index, trace=trace, effects=effects, counters=counts))
    assert not pending and calls == native["calls"]
    assert len(calls) == len(descriptors) == len(provider.trace)
    assert [c["index"] for c in calls] == [t[0] for t in provider.trace] == [d["index"] for d in descriptors]
    return calls, descriptors


def actual_panel_buffers(trace_root, work, provider, geometry, *, panel):
    """Follow actual bufferized call inputs; a reused malloc address is insufficient."""
    from test_owned_buffer_host_model import generic_snapshot
    from xdsl.ir import BlockArgument, Operation

    from merlin.common import mlir_query as mq

    trace = json.loads((trace_root / "trace.json").read_text())
    rows = []
    for stage in trace["stages"]:
        if stage["kind"] != "native" or stage["stage"] != "mlir:one-shot-bufferize":
            continue
        for file in stage["files"]:
            source = trace_root / file
            generic = work / ("bufferized-" + str(len(rows)) + ".mlir")
            module = mq.parse(generic_snapshot(source, generic))
            # Native upstream verified the whole input. The older xDSL arith
            # verifier rejects harmless native trunci overflowFlags defaults;
            # independently verify every allocation inspected here instead.
            allocations = [op for op in module.walk() if op.name in ("memref.alloc", "memref.alloca")]
            for op in allocations:
                op.verify()
            symbols = {record["symbol"].removesuffix("_primitive"): record for record in provider.records}

            def root(value, seen=()):
                assert value not in seen, "cyclic alias outside supported structured carry"
                seen = (*seen, value)
                owner = value.owner
                if isinstance(value, BlockArgument):
                    parent = owner.parent_op()
                    assert parent.name == "scf.for" and value.index > 0, "unknown panel buffer owner"
                    return root(parent.iter_args[value.index - 1], seen)
                assert isinstance(owner, Operation)
                if owner.name in (
                    "memref.cast",
                    "memref.collapse_shape",
                    "memref.expand_shape",
                    "memref.subview",
                    "memref.reinterpret_cast",
                ):
                    return root(owner.operands[0], seen)
                if owner.name == "scf.for":
                    return root(owner.iter_args[value.index], seen)
                assert owner.name in ("memref.alloc", "memref.alloca"), owner.name
                return owner

            discovered, checkpoint_violations = {}, []
            for call in module.walk():
                if call.name != "func.call" or call.properties["callee"].string_value() not in symbols:
                    continue
                name = call.properties["callee"].string_value()
                allocation = root(call.operands[1])
                ancestors, current = [], call.parent_op()
                while current is not None:
                    ancestors.append(current)
                    current = current.parent_op()
                loop = next(op for op in ancestors if op.name == "scf.for")
                if panel:
                    # Preserve the observed upstream counterexample. This
                    # checkpoint cannot qualify final reuse: later stock
                    # lowering may hoist the actual malloc out of every loop.
                    owner = allocation.parent_op()
                    while owner is not None:
                        if owner is loop:
                            checkpoint_violations.append(
                                dict(symbol=name, reason="activation panel allocated inside the window loop")
                            )
                        owner = owner.parent_op()
                discovered[name] = allocation
            assert set(discovered) == set(symbols)
            unique = set(discovered.values())
            panel_bytes = 0
            for allocation in unique:
                typ = allocation.results[0].type
                shape = tuple(typ.get_shape())
                assert all(d > 0 for d in shape)
                panel_bytes += int(np.prod(shape)) * typ.element_type.width.data // 8
            n, ci, h, w = geometry["shape"]
            top, left, bottom, right = geometry["padding"]
            pad_shape = (n, ci, h + top + bottom, w + left + right)
            pad = [a for a in allocations if tuple(a.results[0].type.get_shape()) == pad_shape]
            assert pad, "original padded image must still be accounted"
            output_sites = []
            for allocation in allocations:
                typ = allocation.results[0].type
                shape = tuple(typ.get_shape())
                if len(shape) != 2 or shape[0] != geometry["outputs"] or str(typ.element_type) != "i32":
                    continue
                output_sites.append(dict(shape=list(shape), payload_bytes=int(np.prod(shape)) * 4))
            rows.append(
                dict(
                    path=str(source),
                    sha256=sha256_file(source),
                    generic_sha256=sha256_file(generic),
                    activation_buffers=len(unique),
                    activation_payload_bytes=panel_bytes,
                    full_padded_image_payload_bytes=int(np.prod(pad_shape)) * geometry["width"],
                    static_padded_allocation_sites=len(pad),
                    static_i32_output_allocation_sites=output_sites,
                    output_allocation_scope="actual pre-DCE sites, not executed malloc count or peak RAM",
                    panel_allocations_outside_window_loops=panel and not checkpoint_violations,
                    checkpoint_violations=checkpoint_violations,
                    qualification_scope="intermediate observation only; independent final-native proof required",
                    budget_scope=(
                        "activation packing roots only; excludes padded image, complete output/history, "
                        "output/writer tiles, C ABI scratch, native-engine state and code"
                    ),
                )
            )
    assert len(rows) == 1
    return rows[0]


def actual_final_packing(work, built, provider, geometry, selection, *, panel):
    """Consume the production proof; baseline observes its original full extent."""
    from dataclasses import asdict
    from pathlib import Path

    from merlin.llvmlower.borrowed_buffer_calls import read_borrowed_calls
    from merlin.verify.host_packing_storage import PackingStorageUse, inspect_packing_storage

    qualification = json.loads((provider.work / "original_allocation_qualification.json").read_text())
    allocator = qualification["runtime_source"]
    assert sha256_file(Path(allocator["path"])) == allocator["sha256"]
    recipe = json.loads((work / "compilation_recipe.json").read_text())
    assert recipe["status"] == "completed"
    allocator_builds = [row for row in recipe["commands"] if any(p["path"] == allocator["path"] for p in row["inputs"])]
    assert len(allocator_builds) == 1
    allocation_object = allocator_builds[0]["output"]
    assert any(
        p["path"] == allocator["path"] and p["sha256"] == allocator["sha256"] for p in allocator_builds[0]["inputs"]
    )
    assert sha256_file(Path(allocation_object["path"])) == allocation_object["sha256"]
    assert any(
        p["path"] == allocation_object["path"] and p["sha256"] == allocation_object["sha256"]
        for p in recipe["commands"][-1]["inputs"]
    )
    roster_path = work / "lower/borrowed-buffer-calls.json"
    original_calls = read_borrowed_calls(json.loads(roster_path.read_text()))
    uses = tuple(
        PackingStorageUse(binding.writer.wrapper_symbol or binding.writer.symbol, 1) for binding in provider.bindings
    )
    if panel:
        links = list(work.glob("host_provider/packing_storage_*/source_packing.json"))
        assert len(links) == 1, "bounded route requires the original production native qualification"
        link = json.loads(links[0].read_text())
        assert link["schema"] == "merlin.source_packing_storage.v1"
        report = link["native"]
        assert link["source_payload_bytes"] == report["logical_payload_bytes"]
        assert report["max_payload_bytes"] == geometry["budget"]
        # The completed host transaction must retain the original report pins.
        pins = built["host_provider"]["compilation_pins"]
        assert pins[str(links[0].resolve())] == sha256_file(links[0])
        assert all(pins[path] == digest for path, digest in report["pins"].items())
    else:
        n, ci, _, _ = geometry["shape"]
        kh, kw = geometry["kernel"]
        oh, ow = geometry["output"]
        full = n * ci * kh * kw * oh * ow * geometry["width"]
        # This is an observer bound from the actual original full source extent,
        # not a change to the fixed target facts or selected optimization budget.
        report = inspect_packing_storage(
            built["host_artifacts"],
            borrowed_calls=original_calls,
            uses=uses,
            allocation_effects=selection.packing_allocation_effects,
            max_payload_bytes=full,
            entry="forward",
            diagnostics=work / "full_packing_observation",
        )
        assert report["logical_payload_bytes"] == full
    assert report["schema"] == "merlin.final_packing_storage.v1"
    assert report["allocation_effects"] == qualification["effects"]
    assert json.loads(json.dumps(report["original_borrowed_calls"])) == json.loads(
        json.dumps([asdict(c) for c in original_calls])
    )
    assert {(row["symbol"], row["argument"]) for row in report["uses"]} == {
        (use.wrapper_symbol, use.argument) for use in uses
    }
    assert all(row["allocation_outside_all_cfg_cycles"] for row in report["buffers"])
    assert all(len(row["release_operations"]) == 1 for row in report["buffers"])
    assert report["allocator_requested_bytes"] == (report["logical_payload_bytes"] + report["alignment_overhead_bytes"])
    assert all(sha256_file(Path(path)) == digest for path, digest in report["pins"].items())
    record = report["report"]
    assert sha256_file(Path(record["path"])) == record["sha256"]
    return report


def realized_gather_work(selected_path):
    """Count actual selected scalar-copy domains, not physical cache traffic."""
    module = parse_mlir_text(selected_path.read_text())
    rows = []
    for op in module.walk():
        if op.name != "linalg.generic" or len(op.inputs) != 1 or len(op.outputs) != 1:
            continue
        typ = op.outputs[0].type
        if len(typ.get_shape()) != 6 or len(op.inputs[0].type.get_shape()) != 4:
            continue
        body = list(op.body.block.ops)
        if len(body) != 1 or body[0].name != "linalg.yield" or tuple(body[0].operands) != (op.body.block.args[0],):
            continue
        invocations, owner = 1, op.parent_op()
        while owner is not None:
            if owner.name == "scf.for":
                values = []
                for value in (owner.lb, owner.ub, owner.step):
                    assert value.owner.name == "arith.constant", "unknown gather invocation bound"
                    values.append(value.owner.value.value.data)
                low, high, step = values
                assert step > 0
                invocations *= max(0, (high - low + step - 1) // step)
            owner = owner.parent_op()
        elements = int(np.prod(typ.get_shape()))
        width = typ.element_type.width.data // 8
        rows.append(
            dict(
                shape=list(typ.get_shape()),
                invocations=invocations,
                logical_read_bytes=elements * width * invocations,
                logical_write_bytes=elements * width * invocations,
            )
        )
    assert rows
    return dict(
        regions=rows,
        logical_read_bytes=sum(r["logical_read_bytes"] for r in rows),
        logical_write_bytes=sum(r["logical_write_bytes"] for r in rows),
        scope="actual selected source copy domains, not native instruction/cache/DRAM bytes",
    )


@pytest.mark.parametrize("geometry", CASES)
def test_ordinary_convolution_uses_bounded_live_panels_and_checks_complete_state(tmp_path, record_property, geometry):
    import hashlib
    import time
    from pathlib import Path

    from test_convolution_route_provider import ConvolutionProvider
    from test_dense_ordinary_model import build_model
    from test_dense_source_catalog import compilation_memory
    from test_selected_dense_contracts import inputs as dense_inputs

    from merlin.common import compile_trace
    from merlin.llvmlower import toolchain
    from merlin.model_check import full_build as FB
    from merlin.runtime.backends import spike
    from merlin.targetgen.elf_lanes import read_program_section

    if not toolchain.available() or not spike.available():
        pytest.skip("ordinary native lowering and functional simulator unavailable")
    n, ci, _, _ = geometry["shape"]
    kh, kw = geometry["kernel"]
    oh, ow = geometry["output"]
    k, features, positions = ci * kh * kw, geometry["outputs"], n * oh * ow
    facts = tmp_path / "capabilities"
    facts.mkdir()
    model = dense_inputs(
        facts,
        tile=geometry["tile"],
        widths=(geometry["width"], geometry["width"], 4),
        shared=geometry["shared"],
        shape=(features, k, positions),
    )
    bundle = tmp_path / "capture"
    expected, epochs = capture(bundle, geometry)
    observations, outputs = [], []
    for arm, route in (("baseline", "full"), ("bounded", "auto")):
        provider = ConvolutionProvider(
            tmp_path / (arm + "_provider"), model, geometry, epochs, budget=geometry["budget"], route=route
        )
        selection = provider.selection()
        work = tmp_path / arm
        work.mkdir()
        trace_root = work / "lowering_trace"
        started = time.perf_counter()
        memory = {}
        try:
            with (
                compilation_memory() as memory,
                compile_trace.session(compile_trace.Request(str(trace_root), dump_after=("mlir:one-shot-bufferize",))),
            ):
                built = build_model(bundle, work, selection)
        finally:
            seconds = time.perf_counter() - started
            # Retain costs even if production qualification or a later observer
            # fails. Do not lose real measurements behind the success verdict.
            (work / "build_observations.json").write_text(
                json.dumps(
                    dict(
                        compile_seconds=seconds,
                        compilation_memory=memory,
                        scope="sampled build process costs; no accelerator latency claim",
                    ),
                    indent=2,
                )
            )
        console, wall = FB.run_host(built["elf"], timeout_s=30)
        (work / "console.bin").write_bytes(console)
        (work / "execution_observations.json").write_text(
            json.dumps(
                dict(
                    simulator_process_wall_seconds=wall,
                    console_sha256=sha256_file(work / "console.bin"),
                    scope="functional simulator process wall time; not accelerator latency",
                ),
                indent=2,
            )
        )
        actual, metadata = FB.read(console, bundle, built["build_hash"])
        assert len(actual) == len(expected) == 5
        for got, want in zip(actual, expected, strict=True):
            assert got.dtype == want.dtype and got.shape == want.shape and got.tobytes() == want.tobytes()
        outputs.append(actual)
        native = json.loads((provider.work / "native_qualification.json").read_text())
        calls, descriptors = observe_calls(console, provider, native)
        panel = route == "auto"
        buffers = actual_panel_buffers(trace_root, work, provider, geometry, panel=panel)
        final_storage = actual_final_packing(work, built, provider, geometry, selection, panel=panel)
        if panel:
            # Regression/rationale: high-level carried buffers do not prove the
            # intermediate actually reused them. Final native proof above is
            # mandatory before accepting the narrower final workspace claim.
            assert buffers["checkpoint_violations"]
            assert not buffers["panel_allocations_outside_window_loops"]
            assert final_storage["logical_payload_bytes"] <= geometry["budget"]
            for epoch in range(3):
                per_epoch = len(descriptors) // 3
                current = descriptors[epoch * per_epoch : (epoch + 1) * per_epoch]
                for index in range(len(provider.records)):
                    selected = [d for d in current if d["index"] == index]
                    assert len(selected) > 1 and len({d["pointer"] for d in selected}) == 1
        else:
            assert len(provider.records) == 1 and len(calls) == 3
            assert buffers["activation_payload_bytes"] == k * positions * geometry["width"]
        sections = []
        for record in provider.records:
            code, section = read_program_section(built["elf"], record["section"], max_bytes=64 << 10)
            assert code and not section["writable"]
            sections.append(dict(**section, sha256=hashlib.sha256(code).hexdigest()))
            record["result"].revalidate()
        (source_path,) = work.rglob("source_dense.json")
        source = json.loads(source_path.read_text())
        assert source["selected_sha256"] == sha256_file(source_path.with_suffix(".mlir"))
        gather_work = realized_gather_work(source_path.with_suffix(".mlir"))
        assert (
            gather_work["logical_read_bytes"]
            == gather_work["logical_write_bytes"]
            == 3 * k * positions * geometry["width"]
        )
        host = built["host_provider"]
        assert host["status"] == "completed" and len(host["objects"]) == len(provider.records)
        for pin in (*host["source_model"], *(r[key] for r in host["objects"] for key in ("object", "llvm"))):
            assert pin["sha256"] == sha256_file(Path(pin["path"]))
        recipe = json.loads((work / "compilation_recipe.json").read_text())
        # Total payload sums over actual invoked kernels, not over one selected
        # static group. Repeated weight packing/dispatch is intentionally visible.
        terminal = [
            next(c["counters"] for c in reversed(calls) if c["index"] == i) for i in range(len(provider.records))
        ]
        totals = [sum(row[i] for row in terminal) for i in range(6)]
        observations.append(
            dict(
                arm=arm,
                route=route,
                budget_bytes=geometry["budget"],
                actual_calls=calls,
                actual_input_descriptors=descriptors,
                total_counters=totals,
                native_qualification=native,
                abi_scratch_by_static_kernel=[
                    dict(
                        index=i,
                        logical_payload_bytes=sum(arg.nbytes for arg in r["result"].compiled.kernel.args),
                        declared_bytes_with_guards=sum(arg.nbytes for arg in r["result"].compiled.kernel.args)
                        + 3 * 128,
                        scope="reviewed C scratch arrays; excludes compiler frame padding and local-engine arrays",
                    )
                    for i, r in enumerate(provider.records)
                ],
                actual_buffers=buffers,
                final_native_storage=final_storage,
                actual_source_gather_work=gather_work,
                source_receipt=source,
                source_receipt_sha256=sha256_file(source_path),
                facts_sha256=sha256_file(facts / "facts.json"),
                software_sha256=sha256_file(facts / "software.yaml"),
                library_sources=source["bindings"][0]["schedule"]["pipeline"]["library_sources_sha256"],
                compile_seconds=seconds,
                compilation_memory=memory,
                elf_sha256=sha256_file(built["elf"]),
                elf_bytes=built["elf"].stat().st_size,
                elf_sections=FB.elf_sections(built["elf"]),
                actual_kernel_sections=sections,
                parent_compiler_commands=len(recipe["commands"]),
                provider_compiler_commands=sum(len(r["compile_commands"]) for r in host["objects"]),
                functional_forward_instructions=metadata["metrics"]["cycles"],
                simulator_process_wall_seconds=wall,
                physical_traffic_bytes=None,
                host_accelerator_component_seconds=None,
                scope=(
                    "neutral host preparation and qualified dense primitives; "
                    "logical payload/capacity results, no latency claim"
                ),
            )
        )
    for before, after in zip(*outputs, strict=True):
        np.testing.assert_array_equal(before, after)
    before, after = observations
    for field in ("facts_sha256", "software_sha256", "library_sources"):
        assert before[field] == after[field]
    assert before["source_receipt"]["source_sha256"] == after["source_receipt"]["source_sha256"]
    assert (
        before["actual_buffers"]["full_padded_image_payload_bytes"]
        == after["actual_buffers"]["full_padded_image_payload_bytes"]
    )
    assert after["actual_buffers"]["activation_payload_bytes"] < before["actual_buffers"]["activation_payload_bytes"]
    assert (
        after["final_native_storage"]["logical_payload_bytes"] < before["final_native_storage"]["logical_payload_bytes"]
    )
    assert (
        after["final_native_storage"]["allocator_requested_bytes"]
        < before["final_native_storage"]["allocator_requested_bytes"]
    )
    assert before["total_counters"][2] == after["total_counters"][2] == 3 * features * positions * 4
    assert after["total_counters"][5] > before["total_counters"][5] == 3
    # With timing unknown, automatic selection enforces the explicitly chosen
    # activation working-set budget; it makes no speed claim.
    (tmp_path / "pair.json").write_text(json.dumps(observations, indent=2))
    record_property("bounded_convolution_complete_model_pair", json.dumps(observations))


def test_wrong_actual_panel_reduction_refuses_before_provider_link(tmp_path):
    from test_convolution_route_provider import ConvolutionProvider
    from test_dense_ordinary_model import build_model
    from test_selected_dense_contracts import inputs as dense_inputs

    from merlin.llvmlower import toolchain

    if not toolchain.available():
        pytest.skip("native ABI qualification unavailable")
    geometry = CASES[0]
    facts = tmp_path / "capabilities"
    facts.mkdir()
    k = geometry["shape"][1] * int(np.prod(geometry["kernel"]))
    positions = geometry["shape"][0] * int(np.prod(geometry["output"]))
    model = dense_inputs(facts, tile=geometry["tile"], shape=(geometry["outputs"], k, positions))
    bundle = tmp_path / "capture"
    _, epochs = capture(bundle, geometry)
    provider = ConvolutionProvider(
        tmp_path / "provider", model, geometry, epochs, budget=geometry["budget"], mutation="continuation"
    )
    work = tmp_path / "model"
    work.mkdir()
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        build_model(bundle, work, provider.selection())
    assert not (provider.work / "native_qualification.json").exists()
    assert not list(work.rglob("*.elf"))


def test_changed_production_packing_report_refuses_before_model_link(tmp_path):
    from test_convolution_route_provider import ConvolutionProvider
    from test_dense_ordinary_model import build_model
    from test_selected_dense_contracts import inputs as dense_inputs

    from merlin.llvmlower import toolchain

    if not toolchain.available():
        pytest.skip("native ABI qualification unavailable")
    geometry = CASES[0]
    facts = tmp_path / "capabilities"
    facts.mkdir()
    k = geometry["shape"][1] * int(np.prod(geometry["kernel"]))
    positions = geometry["shape"][0] * int(np.prod(geometry["output"]))
    model = dense_inputs(facts, tile=geometry["tile"], shape=(geometry["outputs"], k, positions))
    bundle = tmp_path / "capture"
    _, epochs = capture(bundle, geometry)
    provider = ConvolutionProvider(
        tmp_path / "provider", model, geometry, epochs, budget=geometry["budget"], mutation="packing_report"
    )
    work = tmp_path / "model"
    work.mkdir()
    with pytest.raises(ValueError, match="changed|packing"):
        build_model(bundle, work, provider.selection())
    assert provider.report_tampered
    assert list(work.glob("host_provider/packing_storage_*/source_packing.json"))
    assert not list(work.rglob("*.elf"))
