"""Mixed native loops/images execute through source-owned persistent buffers."""

from dataclasses import replace

import numpy as np
import pytest
from test_compact_dense import project
from test_dense_source import contraction
from test_multioutput_fusion import SPACES, WORD, model, scalar_bodies
from test_selected_dense_contracts import inputs

from merlin.compile.connected_regions import compile_connected_region, plan_connected_region
from merlin.compile.scheduling.program_limits import AddressWindow, Continuation
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.sched.ir import Const, Kernel, evaluate, instances


def fixture(tmp_path, widths=(2, 2, 2)):
    left_type, right_type, acc_type = (f"i{w * 8}" for w in widths)
    dense_dir, point_dir = tmp_path / "dense", tmp_path / "point"
    dense_dir.mkdir()
    point_dir.mkdir()
    dense = inputs(dense_dir, widths=widths, shape=(3, 7, 5))
    _, _, selected = model(point_dir, width=widths[2] * 8)
    clip, rectify = scalar_bodies(widths[2] * 8)[1:]
    text = f"""module {{func.func @mixed(%a:tensor<3x7x{left_type}>,%b:tensor<7x5x{right_type}>)
       ->(tensor<3x5x{acc_type}>,tensor<3x5xi8>,tensor<3x5x{acc_type}>) {{
      {contraction((3, 7, 5), (left_type, right_type, acc_type))}
      %ce=tensor.empty():tensor<3x5xi8>
      %re=tensor.empty():tensor<3x5x{acc_type}>
    """
    import re

    for name, init, dtype, body in [("c", "ce", "i8", clip), ("d", "re", acc_type, rectify)]:
        body = re.sub(r"%([a-z]+)", lambda m: "%" + name + "_" + m[1], body)
        text += f"""%{name}=linalg.generic {{indexing_maps=[affine_map<(m,n)->(m,n)>,
          affine_map<(m,n)->(m,n)>],iterator_types=["parallel","parallel"]}}
          ins(%r:tensor<3x5x{acc_type}>) outs(%{init}:tensor<3x5x{dtype}>) {{
          ^bb0(%{name}_a:{acc_type},%{name}_init:{dtype}): {body}
          linalg.yield %{name}_v:{dtype} }} ->tensor<3x5x{dtype}>\n"""
    text += f"return %r,%c,%d:tensor<3x5x{acc_type}>,tensor<3x5xi8>,tensor<3x5x{acc_type}> }} }}"
    module = parse_mlir_text(text)
    operations = tuple(op for op in module.walk() if op.name == "linalg.generic")
    region = plan_connected_region(module, operations)
    windows = (
        AddressWindow("input0", 0, 21 * widths[0]),
        AddressWindow("input1", 0, 35 * widths[1]),
        AddressWindow("output1", 0, 15 * widths[2]),
        AddressWindow("output0", 0, 15),
        AddressWindow("output1", 64, 15 * widths[2]),
    )

    def point_options(region, stage, bindings):
        role = region.stages.index(stage)
        call = selected["kernel"].body[role]
        changes = {"src0": 5, "r0": bindings.inputs[0].start, "dst": 4 if role == 1 else 5, "w": bindings.output.start}
        call = replace(call, args=tuple((k, Const(changes.get(k, v.value))) for k, v in call.args))
        return {
            k: v
            for k, v in selected.items()
            if k not in {"bindings", "selected_contracts", "composition_law", "kernel", "objects"}
        } | {
            "kernel": Kernel("point", (), (call,)),
            "objects": (),
            "fusion_qualification_identity": "neutral-byte-machine-scalar-v1",
        }

    def dense_options(region, stage, bindings):
        return dict(
            storage=dense[2], adapter=dense[3], name="mixed_dense", static_budget=80, verification_timeout_ms=20000
        )

    settings = dict(
        windows=windows,
        byte_orders=("little",) * 5,
        selected_contracts=selected["selected_contracts"],
        dense_contracts=dense[0],
        continuation=Continuation("neutral-synchronous-buffer-runtime", SPACES[2:], True, True, 8),
        stage_options=point_options,
        dense_options=dense_options,
    )
    return region, settings, dense


def launch_native(invocation, inputs, output):
    assert all(view.readonly for view in inputs) and not output.readonly
    if invocation.images is None:
        from merlin.compile.scheduling.block_schedule import Compute, Load, Preload

        compiled = invocation.product.compiled
        left = np.frombuffer(inputs[0], dtype="<i2").reshape(compiled.kernel.args[0].shape)
        right = np.frombuffer(inputs[1], dtype="<i2").reshape(compiled.kernel.args[1].shape)
        result = np.frombuffer(output, dtype="<i2").reshape(compiled.kernel.args[2].shape)
        memories = {"lhs": {}, "weight": {}, "acc": {}}
        preload = None
        for call, env in instances(compiled.kernel):
            op = project(replace(call, args=tuple((k, evaluate(v, env)) for k, v in call.args)))
            if isinstance(op, Load):
                source = left if op.role == "lhs" else right
                for i in range(op.rows):
                    for j in range(op.cols):
                        memories[op.role][op.row + i, j] = int(source[op.dram_row + i, op.dram_col + j])
            elif isinstance(op, Preload):
                preload = op
            elif isinstance(op, Compute):
                for i in range(op.rows):
                    for j in range(preload.cols):
                        slot = preload.accumulator_row + i, j
                        value = memories["acc"][slot] if preload.accumulate else 0
                        for k in range(op.cols):
                            value = (
                                value
                                + memories["lhs"][op.input_row + i, k] * memories["weight"][preload.weight_row + k, j]
                                + 32768
                            ) % 65536 - 32768
                        memories["acc"][slot] = value
            else:
                for i in range(op.rows):
                    for j in range(op.cols):
                        result[op.dram_row + i, op.dram_col + j] = memories["acc"][op.accumulator_row + i, j]
    else:
        binding = invocation.bindings
        for image in invocation.images:
            for opcode, src0, src1, dst, r0, r1, out, elements in WORD.iter_unpack(image):
                assert SPACES[src0] == binding.inputs[0].space and SPACES[dst] == binding.output.space
                for i in range(elements):
                    offset = r0 - binding.inputs[0].start + 2 * i
                    value = int.from_bytes(inputs[0][offset : offset + 2], "little", signed=True)
                    value = min(127, max(-128, value)) if opcode == 1 else max(0, value)
                    size = 1 if opcode == 1 else 2
                    position = out - binding.output.start + size * i
                    output[position : position + size] = value.to_bytes(size, "little", signed=True)


def runtime_buffers():
    a = ((np.arange(21) * 7919) % 65536 - 32768).astype("<i2").reshape(3, 7)
    b = ((np.arange(35) * 3571) % 65536 - 32768).astype("<i2").reshape(7, 5)
    buffers = {name: bytearray([193] * 512) for name in SPACES[2:]}
    buffers["input0"][:42] = a.tobytes()
    buffers["input1"][:70] = b.tobytes()
    return buffers, a, b


def test_mixed_dense_pointwise_outputs_fanout_and_exact_modular_semantics(tmp_path):
    region, settings, _ = fixture(tmp_path)
    assert region.outputs == (2, 3, 4) and region.lifetimes[2][1] == 3
    product = compile_connected_region(region, **settings)
    assert product.report["stage_forms"] == [
        "checked_compact_dense",
        "checked_pointwise_images",
        "checked_pointwise_images",
    ]
    buffers, a, b = runtime_buffers()
    visible = product.execute(buffers, launch_native)
    expected = (a.astype(np.int64) @ b.astype(np.int64)).astype("<i2")
    assert bytes(visible[2]) == expected.tobytes()
    assert bytes(visible[3]) == expected.clip(-128, 127).astype("i1").tobytes()
    assert bytes(visible[4]) == expected.clip(0).astype("<i2").tobytes()
    assert buffers["input0"][:42] == a.tobytes() and buffers["input1"][:70] == b.tobytes()
    assert buffers["output1"][30:64] == bytes([193] * 34)
    assert buffers["output1"][94:] == bytes([193] * (512 - 94))
    assert all(v.readonly for v in visible.values())
    assert product.report["execution_qualification"] == "not_established"


@pytest.mark.parametrize(
    "failure", ["live_alias", "runtime_alias", "short", "late_mutation", "source_override", "swallowed_epilogue"]
)
def test_mixed_boundary_refusals(tmp_path, failure):
    region, settings, _ = fixture(tmp_path)
    original = settings["dense_options"]
    if failure == "live_alias":
        settings["windows"] = (*settings["windows"][:4], settings["windows"][2])
    if failure in ("source_override", "swallowed_epilogue"):
        key = "operation" if failure == "source_override" else "output_operation"
        settings["dense_options"] = lambda *args: dict(original(*args), **{key: region.stages[1].operation})
    if failure in ("live_alias", "source_override", "swallowed_epilogue"):
        with pytest.raises(ValueError):
            compile_connected_region(region, **settings)
        return
    product = compile_connected_region(region, **settings)
    buffers, _, _ = runtime_buffers()
    if failure == "runtime_alias":
        buffers["input1"] = buffers["input0"]
    if failure == "short":
        buffers["input0"] = bytearray(2)
    if failure == "late_mutation":
        object.__setattr__(product.invocations[0].bindings, "inputs", ())
    called = []
    with pytest.raises(ValueError):
        product.execute(buffers, lambda *args: called.append(args))
    assert not called


def test_workflow_consumes_mixed_products_and_keeps_unfused_fallback(tmp_path):
    from merlin.compile import fusion_workflow as F

    region, settings, _ = fixture(tmp_path)
    contract = dict(capability_contract={}, numerical_policy_sha256="a" * 64)
    rows = F.discover(region.source, **contract)["candidates"]
    row = next(r for r in rows if r["operation_ordinals"] == [s.ordinal for s in region.stages])

    def realize(private):
        return F.realize_candidate(private, row, connected=settings, **contract)

    def emit(realization, session):
        if realization.receipt["status"] == "unfused":
            return str(realization.unfused)
        program = realization.artifact
        assert len(program.ordered_invocations()) == 3
        buffers, _, _ = runtime_buffers()
        return {k: bytes(v) for k, v in program.execute(buffers, launch_native).items()}

    results = list(F.evaluate_fusions(region.source, (F.FusionAlternative("mixed", realize),), emit))
    assert [r.status for r in results] == ["compiled", "compiled"], [r.reason for r in results]
    assert results[1].realization.receipt["route"] == "connected_mixed"
    assert set(results[1].evaluation.result.artifact) == {2, 3, 4}
    bad = dict(settings, dense_options=lambda *args: dict(settings["dense_options"](*args), operation=None))
    declined = F.realize_candidate(region.source, row, connected=bad, **contract)
    assert declined.receipt["status"] == "declined" and declined.unfused is region.source


def test_ordered_products_refuse_mutation_and_unfinished_launcher(tmp_path):
    region, settings, _ = fixture(tmp_path)
    product = compile_connected_region(region, **settings)
    buffers, _, _ = runtime_buffers()
    with pytest.raises(ValueError, match="complete synchronously"):
        product.execute(buffers, lambda *args: object())
    object.__setattr__(product.invocations[0].product.compiled, "kernel", Kernel("wrong", (), ()))
    with pytest.raises(ValueError, match="changed"):
        product.ordered_invocations()


def test_dense_pointwise_dense_plan_retains_source_proven_zero_initializers(tmp_path):
    import importlib.util

    from merlin.common.paths import repo_root

    spec = importlib.util.spec_from_file_location(
        "mixed_fusion_example", repo_root() / "examples/compiler/fusion_exploration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = parse_mlir_text(dict(module.patterns())["simple_integer_mlp"])
    operations = tuple(op for op in source.walk() if op.name == "linalg.generic")
    region = plan_connected_region(source, operations)
    assert [s.kind for s in region.stages] == ["contraction", "elementwise_map", "contraction"]


def test_compile_time_example_consumes_native_loop_emission_and_image_bindings(tmp_path):
    import importlib.util

    from merlin.common.paths import repo_root
    from merlin.sched.isa import InstrDef, InstructionSet, Operand

    region, settings, _ = fixture(tmp_path)
    product = compile_connected_region(region, **settings)
    spec = importlib.util.spec_from_file_location(
        "connected_c_example", repo_root() / "examples/compiler/connected_composition.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    observed = []

    def instruction_set(invocation):
        definitions = {}
        for call, _ in instances(invocation.product.compiled.kernel):
            definitions[call.instr] = InstrDef(
                call.instr,
                tuple(Operand(k, "int") for k, _ in call.args),
                render_c=lambda args, n=call.instr: f"target_{n}({', '.join(args)});",
            )
        return InstructionSet("explicit-caller-ISA", definitions, {"i16": "int16_t"}, "/* caller completion */")

    def pointwise(invocation, symbol):
        observed.append((symbol, invocation.images, invocation.bindings))
        return f"void {symbol}(void) {{ /* target image launch is caller-owned */ }}"

    def sequence(program, entries):
        assert [i.stage.output for _, i in entries] == [2, 3, 4]
        assert entries[1][1].bindings.inputs[0] == entries[0][1].bindings.output
        return "/* caller emits pointer bindings and complete-before-next launch sequence here */"

    source = example.emit_connected_c(
        product, instruction_set=instruction_set, emit_pointwise=pointwise, emit_sequence=sequence
    )
    assert "connected_stage_0" in source and "for (" in source and "target_compute(" in source
    assert [entry[0] for entry in observed] == ["connected_stage_1", "connected_stage_2"]
    assert all(entry[1] for entry in observed)


def test_runtime_uses_explicit_nonzero_backing_origins(tmp_path):
    region, settings, _ = fixture(tmp_path)
    settings["windows"] = (
        AddressWindow("input0", 16, 42),
        AddressWindow("input1", 32, 70),
        AddressWindow("output1", 64, 30),
        AddressWindow("output0", 8, 15),
        AddressWindow("output1", 128, 30),
    )
    product = compile_connected_region(region, **settings)
    buffers, a, b = runtime_buffers()
    origins = {"input0": 16, "input1": 32, "output0": 8, "output1": 64}
    visible = product.execute(buffers, launch_native, backing_origins=origins)
    expected = (a.astype(np.int64) @ b.astype(np.int64)).astype("<i2")
    assert bytes(visible[2]) == expected.tobytes()
    assert bytes(visible[3]) == expected.clip(-128, 127).astype("i1").tobytes()
    assert bytes(visible[4]) == expected.clip(0).astype("<i2").tobytes()
    for invalid in ({}, dict(origins, input0=-1), dict(origins, input0=17)):
        with pytest.raises(ValueError):
            product.execute(buffers, launch_native, backing_origins=invalid)


def test_dense_same_ssa_operands_cannot_claim_disjoint_pointer_arguments():
    text = f"""module {{func.func @alias(%a:tensor<3x3xi16>)->tensor<3x3xi16> {{
      {contraction((3, 3, 3), ("i16", "i16", "i16"), rhs="a")}
      %init=tensor.empty():tensor<3x3xi16>
      %out=linalg.generic {{indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>],
          iterator_types=["parallel","parallel"]}}
        ins(%r:tensor<3x3xi16>) outs(%init:tensor<3x3xi16>) {{
        ^bb0(%x:i16,%unused:i16):
        %zero=arith.constant 0:i16
        %value=arith.maxsi %x,%zero:i16
        linalg.yield %value:i16 }} ->tensor<3x3xi16>
      return %out:tensor<3x3xi16> }} }}"""
    module = parse_mlir_text(text)
    region = plan_connected_region(module, tuple(op for op in module.walk() if op.name == "linalg.generic"))
    with pytest.raises(ValueError, match="pointer arguments alias"):
        compile_connected_region(
            region,
            windows=(),
            byte_orders=(),
            selected_contracts=None,
            continuation=None,
            stage_options=lambda *args: None,
            dense_options=lambda *args: None,
        )
