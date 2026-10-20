"""Neutral selected-IR consumption, independently of package transport claims.

This is a small actual emitter/LLVM/native acceptance probe. It establishes no
Gemmini effect law, hardware support, timing result or packing/fusion support.
The host owns the typed fixture, reference, emitted-call checker and runtime.
"""

import ctypes
import inspect
import json
import subprocess
from pathlib import Path

import pytest
import test_target_configuration_native as native
from test_selected_schedule_contracts import inputs

from merlin.common.digest import sha256_file
from merlin.common.invocation_record import verify as verify_invocation
from merlin.llvmlower.compilation_recipe import CompilationRecipe, verify_completed_recipe
from merlin.targetgen import capsule_common as CC
from merlin.targetgen import package_runtime as P


def declared_contract(requirement):
    """Synthetic interface declarations; only configuration is exercised here."""
    names = ("computation", "device", "emission")
    commands = ("lower_interface_to_target", "emit_target_bundle", "emit_target_bundle")
    irs = ("typed_computation", "selected_target_mlir", "native_artifact")
    return {
        "schema": "merlin.shared_pass_contract.v1",
        "boundaries": {
            name: {
                "ir": ir,
                "command": command,
                "contracts": dict(semantics="original-neutral-v1", effects="neutral-C-v1", capabilities="fixture-v1"),
                "features": list(requirement["foundations"]),
            }
            for name, command, ir in zip(names, commands, irs, strict=True)
        },
        "pipeline": requirement["pipeline"],
        "controls": {
            "default_policy": [],
            "choices": {name: dict(enabled=[False], max_growth_ops=0) for name in requirement["families"]},
            "max_repairs": 0,
            "retain_original": True,
        },
        "families": [
            dict(
                name=name,
                status="unknown",
                reason="not exercised by this neutral probe",
                boundary="computation",
                requirements=needs,
            )
            for name, needs in requirement["families"].items()
        ],
    }


def check_emitted_calls(selected_module, llvm_module):
    """Reconstruct actual selected instructions and actual LLVM operand DAGs.

    No optimizer report, package consumption.json, or requested plan is read.
    The qualified fixture implements these four operations as external C calls.
    Unknown operations/callees and nonconstant scalar arguments refuse.
    """
    from xdsl.dialects import llvm

    selected_module.verify()
    llvm_module.verify()
    expected = []
    for op in selected_module.body.block.ops:
        if op.name == "arith.constant":
            continue
        values = tuple(v.owner.value.value.data for v in op.operands)
        slot, a, b = values
        name = op.name.split(".")[-1]
        arguments = {"set": (slot, a), "reset": (slot,), "opaque": (), "work": ("input", "output", a, b)}
        assert name in arguments, "unqualified selected operation"
        expected.append(("neutral_" + name, arguments[name]))
    definition = [op for op in llvm_module.body.block.ops if isinstance(op, llvm.FuncOp) and op.body.blocks]
    assert len(definition) == 1 and definition[0].sym_name.data == "neutral_kernel"
    block = definition[0].body.block
    values = dict(zip(block.args, ("input", "output"), strict=True))
    actual = []
    for op in block.ops:
        if isinstance(op, llvm.ConstantOp):
            values[op.results[0]] = op.value.value.data
        elif isinstance(op, llvm.CallOp):
            assert not op.results and op.callee is not None
            actual.append((op.callee.root_reference.data, tuple(values[v] for v in op.operands)))
        else:
            assert isinstance(op, llvm.ReturnOp) and not op.operands, "unknown emitted work"
    assert actual == expected, "emitted effects differ from actual selected IR"
    return sum(name == "neutral_set" for name, _ in actual)


def execute_guarded(library):
    """Three real state epochs with complete independent modular reference."""
    guards = []

    def array(ctype, initial):
        size = ctypes.sizeof(ctype) * len(initial)
        raw = (ctypes.c_uint8 * (size + 32))(*([193] * (size + 32)))
        value = (ctype * len(initial)).from_buffer(raw, 16)
        value[:] = initial
        guards.append((raw, size))
        return value

    image = [(31 * i + 253) % 256 for i in range(native.COUNT)]
    initial = [(3 * i + 1) % 256 for i in range(native.COUNT)]
    data = array(ctypes.c_uint8, image)
    state = array(ctypes.c_uint8, initial)
    history = array(ctypes.c_uint8, [0] * (3 * native.COUNT))
    position = array(ctypes.c_uint32, [7])
    counts = array(ctypes.c_uint64, [0] * 5)
    settings = array(ctypes.c_uint64, [0] * 2)
    library.model.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int]
    library.model.restype = None
    library.model(data, history, state, position, counts, settings, 0)
    expected, carry = [], initial
    for epoch in range(7, 10):
        carry = [(((x + s + epoch) % 256) * 2 + 3) % 256 for x, s in zip(image, carry, strict=True)]
        expected.extend(carry)
    assert all(list(raw[:16]) == list(raw[size + 16 :]) == [193] * 16 for raw, size in guards)
    assert list(data) == image, "actual readonly input changed"
    return (
        [list(history), list(state), list(position), list(data), list(settings)],
        [expected, carry, [10], image, [3, 2]],
        list(counts),
    )


def close_records(records):
    """Close both arms after every emitter/parser/native callback has returned."""
    for record in records:
        for invocation in record["invocations"]:
            verify_invocation(Path(invocation))
    for record in records:
        for role in ("artifacts", "original_sources"):
            assert record[role] == {path: sha256_file(Path(path)) for path in record[role]}, (
                "collective custody changed"
            )


def build_arm(tmp_path, *, enabled, fault=None):
    from xdsl.parser import Parser

    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
    from merlin.llvmlower import toolchain
    from merlin.targetgen import shared_pass_context, shared_pass_contract
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from merlin.targetgen.shared_pass_contract import shared_pass_requirement

    translator, compiler = Path(toolchain.mlir_translate()), Path(toolchain.clang())
    if not translator.is_file() or not compiler.is_file():
        pytest.skip("native translator/compiler required; no consumption qualification")
    facts = tmp_path / "facts"
    facts.mkdir()
    selected, *_ = inputs(facts, width=4, endian="<")
    pkg = native.package(tmp_path, selected, 32)
    requirement = shared_pass_requirement()
    pkg.manifest["shared_pass_contract"] = declared_contract(requirement)
    manifest = pkg.directory / "manifest.yaml"
    manifest.write_text(json.dumps(pkg.manifest))
    driver = pkg.directory / "driver.py"
    text = driver.read_text()
    if fault == "ignore_selected":
        # Its consumption receipt STILL names/hashes selected.target.mlir.
        text = text.replace(
            "Parser(context, source.read_text())", "Parser(context, Path('original.target.mlir').read_text())"
        )
    elif fault == "used_constant":
        text = text.replace(
            "actual.append((op.name.split('.')[1], *values))",
            "if op.name.endswith('.set') and values[0] == 0: values[1] += 1\n"
            "        actual.append((op.name.split('.')[1], *values))",
        )
    driver.write_text(text)
    pkg = P.load_package(pkg.directory, shared_pass_requirement=requirement)
    source = tmp_path / "input.mlir"
    source.write_text("module {}\n")
    runtime = tmp_path / "runtime.c"
    runtime.write_text(native.RUNTIME)
    owners = [
        Path(inspect.getsourcefile(f))
        for f in (build_arm, check_emitted_calls, execute_guarded, native.package, CC.lower_interface)
    ]
    files = [source, runtime, *pkg.directory.iterdir(), selected.facts_path, selected.software_path, *owners]
    files += [Path(module.__file__) for module in (P, shared_pass_context, shared_pass_contract)]
    files += [Path(inspect.getsourcefile(verify_invocation))]
    original = {p: sha256_file(p) for p in files if p.is_file()}
    selection = ConfigurationIRSelection(selected, native.QUALIFICATION, enabled=enabled)
    work = tmp_path / "build"
    _, artifact = CC.lower_interface(
        pkg,
        source,
        work,
        contract=None,
        timeout=30,
        target_configuration=selection,
        shared_pass_requirement=requirement,
    )
    receipt = work / "target_configuration.json"
    receipt_sha = sha256_file(receipt)
    mlir, ll, obj, image = [work / name for name in ("lowered.llvm.mlir", "kernel.ll", "kernel.o", "model.so")]
    assert mlir.read_text() == artifact
    recipe = CompilationRecipe(work, producer=Path(inspect.getsourcefile(build_arm)))
    for index, path in enumerate((*original, receipt, work / "selected.target.mlir", mlir)):
        recipe.bind_preparation(str(index), path)

    def run(argv):
        return subprocess.run(argv, capture_output=True, text=True)

    recipe.run([translator, "--mlir-to-llvmir", mlir, "-o", ll], runner=run, inputs=(mlir,), output=ll)
    recipe.run([compiler, "-O2", "-fPIC", "-c", ll, "-o", obj], runner=run, inputs=(ll,), output=obj)
    recipe.run(
        [compiler, "-O2", "-shared", "-fPIC", obj, runtime, "-o", image],
        runner=run,
        inputs=(obj, runtime),
        output=image,
    )
    recipe.completed(image)
    recipe_sha = sha256_file(recipe.path)
    image_sha = sha256_file(image)
    products = {str(path): sha256_file(path) for path in work.rglob("*") if path.is_file()}
    actual, expected, counts = execute_guarded(ctypes.CDLL(str(image)))
    actual_selected = selection.parse((work / "selected.target.mlir").read_text())
    actual_llvm = Parser(make_llvm_context(), artifact).parse_module()
    CC.revalidate_target_configuration(pkg, work, selection, expected_receipt_sha256=receipt_sha)
    verify_completed_recipe(recipe.path, executable=image, expected_recipe_sha256=recipe_sha)
    assert image_sha == sha256_file(image) and original == {p: sha256_file(p) for p in original}
    assert products == {path: sha256_file(Path(path)) for path in products}
    return dict(
        actual=actual,
        expected=expected,
        counts=counts,
        selected=actual_selected,
        llvm=actual_llvm,
        image_sha256=image_sha,
        recipe_sha256=recipe_sha,
        receipt_sha256=receipt_sha,
        artifacts=products,
        original_sources={str(path): digest for path, digest in original.items()},
        invocations=[str(path) for path in work.rglob("invocation.json")],
    )


def test_selected_effects_reach_actual_emitter_object_and_recurrent_model(tmp_path):
    records = []
    for enabled in (False, True):
        arm = tmp_path / str(enabled)
        arm.mkdir()
        record = build_arm(arm, enabled=enabled)
        assert record["actual"] == record["expected"]
        sites = check_emitted_calls(record["selected"], record["llvm"])
        assert sites == (5 if enabled else 10)
        assert record["counts"] == [3 * sites, 9, 3, 3, 3 * native.COUNT]
        if enabled:
            dropped = record["llvm"].clone()
            call = next(op for op in dropped.walk() if op.name == "llvm.call")
            call.parent_block().erase_op(call)
            with pytest.raises(AssertionError, match="emitted effects differ"):
                check_emitted_calls(record["selected"], dropped)
        records.append({k: v for k, v in record.items() if k not in ("selected", "llvm")})
    assert records[0]["actual"] == records[1]["actual"]
    close_records(records)
    (tmp_path / "pair.json").write_text(json.dumps(records, indent=2))


def test_ignoring_selected_ir_fails_effect_check_even_when_all_values_agree(tmp_path):
    record = build_arm(tmp_path, enabled=True, fault="ignore_selected")
    assert record["actual"] == record["expected"]
    assert record["counts"][0] == 30  # Actual loaded code retained all ten setter sites.
    with pytest.raises(AssertionError, match="emitted effects differ"):
        check_emitted_calls(record["selected"], record["llvm"])


def test_used_constant_replacement_fails_independent_complete_reference(tmp_path):
    record = build_arm(tmp_path, enabled=True, fault="used_constant")
    assert record["counts"][0] == 15  # A command-count-only check would incorrectly pass.
    assert record["actual"] != record["expected"]
    with pytest.raises(AssertionError, match="emitted effects differ"):
        check_emitted_calls(record["selected"], record["llvm"])


@pytest.mark.parametrize("role", ["artifacts", "original_sources"])
def test_later_arm_cannot_replace_an_earlier_arms_retained_files(tmp_path, role):
    records = []
    for name in ("baseline", "candidate"):
        path = tmp_path / name
        path.write_text(name)
        records.append(dict(artifacts={}, original_sources={}, invocations=[]))
        records[-1][role][str(path)] = sha256_file(path)
    (tmp_path / "baseline").write_text("changed during later candidate")
    with pytest.raises(AssertionError, match="collective custody changed"):
        close_records(records)
