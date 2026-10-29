"""Bounded ownership protocol controls with a FAKE PassManager, no SDK/native observation.

CheckerTools.__post_init__ is intentionally not called: this is not toolchain
qualification and no worker job may be submitted by these tests.
"""

import ast
import builtins
import copy
import hashlib
import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from pathlib import Path

import pytest
from buffer_rank_one_inputs import source

from merlin.common.paths import repo_root
from merlin.llvmlower import leading_cse as cse
from merlin.transforms.certificates import parse_ir
from merlin.verify import compiler_memory_origin as origin
from merlin.verify import pass_validation as pv
from merlin.verify.compiler_transforms import memory_lowering_pipeline
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate
from merlin.xdsl_dialects._common import text


@contextmanager
def active(work, *, passes=frozenset({"cse"})):
    tools = object.__new__(pv.CheckerTools)  # Protocol-only fixture, never checked tools.
    object.__setattr__(tools, "sdk_bin", work / "never-SDK")
    object.__setattr__(tools, "compiler_python", Path(sys.executable))
    selected = pv.PassValidation(SpeculativeGate(work / "gate", interrupt=False), tools, passes)
    with pv.validating(selected):
        yield selected
    assert selected.gate.verdicts == []


@contextmanager
def scope(work, *, passes=frozenset({"prune_dead_pure_tensor_ops"}), upstream=None):
    with active(work, passes=passes) as validation:
        original = source()
        with origin.compiler_scope(
            work, original_source=original, memory_pipeline=memory_lowering_pipeline(), validation=validation
        ) as owner:
            selected_upstream = original if upstream is None else upstream
            owner.bind_upstream(selected_upstream)
            (work / "model.mlir").write_text(selected_upstream)
            yield owner, validation


def fake_native_capture(
    work,
    owner,
    monkeypatch,
    *,
    changed=False,
    pipeline=None,
    canonicalized=None,
    actual_source=None,
    input_argv=None,
    printer=None,
):
    """Exercise the wrapper protocol, deliberately no torch-mlir/native execution."""
    log = []
    printer = (lambda value: value) if printer is None else printer
    precanonical = printer(text(parse_ir(source() if actual_source is None else actual_source), generic=True))
    before = canonicalized if canonicalized is not None else precanonical
    after = before + ("\n// fake changed CSE output" if changed else "")
    operation = types.SimpleNamespace(asm=precanonical, context=object())
    operation.get_asm = lambda **options: operation.asm

    class FakeNative:
        @classmethod
        def parse(cls, pipeline, context=None):
            manager = cls()
            manager.pipeline = pipeline
            return manager

        def run(self, op):
            log.append(self.pipeline)
            if self.pipeline == "builtin.module(canonicalize)":
                op.asm = before
            if self.pipeline == "builtin.module(cse)":
                op.asm = after

        def enable_ir_printing(self, **options):
            pass

    torch = types.ModuleType("torch_mlir")
    manager_module = types.ModuleType("torch_mlir.passmanager")
    manager_module.PassManager = FakeNative
    torch.passmanager = manager_module
    ir_module = types.ModuleType("torch_mlir.ir")

    class FakeModule:
        @classmethod
        def parse(cls, raw, context=None):
            assert context is operation.context
            generic = printer(text(parse_ir(raw), generic=True))
            return types.SimpleNamespace(operation=types.SimpleNamespace(get_asm=lambda **options: generic))

    ir_module.Module = FakeModule
    torch.ir = ir_module
    pipeline = pipeline or memory_lowering_pipeline()
    input_path = work / "model.mlir"
    issued = cse.bind_runner("pass\n", work, pipeline=pipeline, upstream=(input_path, input_path.read_text()))
    runner = work / "run_lowering.py"
    runner.write_text(issued)
    child_sys = types.ModuleType("sys")
    child_sys.argv = [
        str(runner),
        str(input_path if input_argv is None else input_argv),
        str(work / "model.ll"),
        pipeline,
    ]
    original_import = builtins.__import__

    def child_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Only this exec namespace sees fake child imports; the interpreter's
        # module registry, builtins, canonical Merlin modules and argv stay intact.
        if level == 0:
            if name == "sys":
                return child_sys
            if name == "torch_mlir":
                return torch
            if name == "torch_mlir.passmanager":
                return manager_module if fromlist else torch
            if name == "torch_mlir.ir":
                return ir_module if fromlist else torch
        return original_import(name, globals, locals, fromlist, level)

    namespace = {"__file__": str(runner), "__builtins__": {**vars(builtins), "__import__": child_import}}
    original_argv = sys.argv
    exec(compile(issued, str(runner), "exec"), namespace)
    namespace["PassManager"].parse("builtin.module(" + pipeline + ")").run(operation)
    assert sys.argv is original_argv
    assert log == [
        "builtin.module(canonicalize)",
        "builtin.module(cse)",
        "builtin.module(" + ",".join(pipeline.split(",")[2:]) + ")",
    ]
    return issued, before, after


def test_original_scope_identity_and_canonical_generic_DCE_epoch(tmp_path):
    with scope(tmp_path) as (owner, validation):
        assert origin.current() is owner and owner.validation is validation
        assert owner.validate() is owner
        assert owner.validate_dce_before(text(parse_ir(source()), generic=True)) is owner
        with pytest.raises(ValueError, match="source epoch"):
            owner.validate_dce_before(source().replace("arith.addi", "arith.subi"))
    assert origin.current() is None
    with pytest.raises(ValueError, match="original active"):
        owner.validate()


@pytest.mark.parametrize("copying", [copy.copy, copy.deepcopy])
def test_copied_or_constructed_scope_never_restores_ownership(tmp_path, copying):
    with scope(tmp_path) as (owner, _):
        foreign = copying(owner)
        with pytest.raises(ValueError, match="original active"):
            foreign.validate()
        constructed = origin.MemoryScope(owner.work, owner.validation, owner.source_sha256, owner.pipeline, owner.nonce)
        with pytest.raises(ValueError, match="original active"):
            constructed.validate()


def test_foreign_thread_even_with_copied_context_refuses(tmp_path):
    with scope(tmp_path) as (owner, _):
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(context.run, owner.validate)
            with pytest.raises(ValueError, match="thread"):
                future.result()


def test_pipeline_mismatch_and_missing_original_validation_refuse(tmp_path):
    with active(tmp_path) as selected:
        with pytest.raises(ValueError, match="canonical memory pipeline"):
            with origin.compiler_scope(
                tmp_path, original_source=source(), memory_pipeline="canonicalize,cse", validation=selected
            ):
                pytest.fail("wrong pipeline entered")
    with pytest.raises(ValueError, match="active typed"):
        with origin.compiler_scope(
            tmp_path, original_source=source(), memory_pipeline=memory_lowering_pipeline(), validation=None
        ):
            pytest.fail("no original validation entered")


def test_fake_protocol_identity_observation_submits_zero_jobs_zero_credit(tmp_path, monkeypatch):
    with scope(tmp_path, passes=frozenset({"cse"})) as (owner, selected):
        _, before, after = fake_native_capture(tmp_path, owner, monkeypatch)
        assert before == after
        assert cse.submit(tmp_path) == 0 and selected.gate.verdicts == []
        record = json.loads((tmp_path / "leading-cse/index.json").read_text())[0]
        assert "checked" not in record and record["nonce"] == owner.nonce


def test_actual_foreign_precannon_operation_refuses_under_original_A_owner(tmp_path, monkeypatch):
    original_source = source
    with pytest.raises(RuntimeError, match="pre-canonicalize whole module"):
        with scope(tmp_path, passes=frozenset({"cse"})) as (owner, selected):
            # Reproduce validator's counterprobe: only the fake operation source
            # changes after original A/U selection. Guarded compiler functions stay original.
            monkeypatch.setitem(globals(), "source", lambda: original_source().replace("arith.addi", "arith.subi"))
            fake_native_capture(tmp_path, owner, monkeypatch)
            pytest.fail("foreign whole module reached identity submission")
    assert selected.gate.verdicts == []
    assert not (tmp_path / "leading-cse/index.json").exists()


def test_original_A_upstream_U_and_external_canonicalize_B_remain_distinct(tmp_path, monkeypatch):
    import hashlib

    from merlin.llvmlower.passes_xdsl import preprocess_text

    original = source()
    upstream, _ = preprocess_text(original)
    assert original != upstream
    canonicalized = text(parse_ir(upstream), generic=True) + "\n"
    with scope(tmp_path, passes=frozenset({"cse"}), upstream=upstream) as (owner, selected):
        fake_native_capture(tmp_path, owner, monkeypatch, actual_source=upstream, canonicalized=canonicalized)
        assert cse.submit(tmp_path) == 0 and selected.gate.verdicts == []
        record = json.loads((tmp_path / "leading-cse/index.json").read_text())[0]
        assert record["source_sha256"] == hashlib.sha256(original.encode()).hexdigest()
        assert record["upstream_sha256"] == hashlib.sha256(upstream.encode()).hexdigest()
        assert record["source_sha256"] != record["upstream_sha256"]
        assert record["precanonical_sha256"] != record["before_sha256"]


@pytest.mark.parametrize("mutation", ["IR", "line_endings"])
def test_original_upstream_file_changed_after_capture_refuses(tmp_path, monkeypatch, mutation):
    with pytest.raises(GateFailure, match="input file changed"):
        with scope(tmp_path, passes=frozenset({"cse"})) as (owner, _):
            fake_native_capture(tmp_path, owner, monkeypatch)
            changed = (
                source().replace("arith.addi", "arith.subi") if mutation == "IR" else source().replace("\n", "\r\n")
            )
            (tmp_path / "model.mlir").write_bytes(changed.encode("utf-8"))
            cse.submit(tmp_path)


@pytest.mark.parametrize("mutation", ["input", "runner"])
def test_files_mutated_after_consumption_refuse_at_successful_scope_exit(tmp_path, monkeypatch, mutation):
    with pytest.raises(ValueError, match="input file changed|frozen memory runner"):
        with scope(tmp_path, passes=frozenset({"cse"})) as (owner, selected):
            fake_native_capture(tmp_path, owner, monkeypatch)
            assert cse.submit(tmp_path) == 0 and selected.gate.verdicts == []
            path = tmp_path / ("model.mlir" if mutation == "input" else "run_lowering.py")
            path.write_bytes(path.read_bytes() + b"\n// changed after consumption\n")
    assert selected.gate.verdicts == []


def test_original_post_consumption_error_is_not_masked_by_exit_file_guard(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="original failure after consumption"):
        with scope(tmp_path, passes=frozenset({"cse"})) as (owner, selected):
            fake_native_capture(tmp_path, owner, monkeypatch)
            assert cse.submit(tmp_path) == 0 and selected.gate.verdicts == []
            (tmp_path / "model.mlir").write_bytes(b"changed after consumption")
            raise RuntimeError("original failure after consumption")


def test_copied_upstream_file_cannot_replace_original_argv_path(tmp_path, monkeypatch):
    copied = tmp_path / "copied.mlir"
    copied.write_text(source())
    with pytest.raises(RuntimeError, match="input/argv"):
        with scope(tmp_path, passes=frozenset({"cse"})) as (owner, _):
            fake_native_capture(tmp_path, owner, monkeypatch, input_argv=copied)


def test_native_printer_SSA_names_do_not_change_upstream_custody(tmp_path, monkeypatch):
    def printer(value):
        return value.replace("%seed0", "%arg0").replace("%buffer0", "%allocation0")

    original = source()
    assert printer(original) != original
    assert origin._canonical_source(original) == origin._canonical_source(printer(original))
    with scope(tmp_path, passes=frozenset({"cse"})) as (owner, selected):
        fake_native_capture(tmp_path, owner, monkeypatch, printer=printer)
        actual = (tmp_path / "leading-cse/0001.precanonical.mlir").read_text()
        assert "%arg0" in actual and "%allocation0" in actual and "%seed0" not in actual
        assert cse.submit(tmp_path) == 0 and selected.gate.verdicts == []


def test_native_metadata_dictionary_order_preserves_all_entries_and_custody(tmp_path, monkeypatch):
    from xdsl.dialects.builtin import StringAttr

    module = parse_ir(source())
    module.attributes.update({"scope.zeta": StringAttr("last"), "scope.alpha": StringAttr("first")})
    upstream = text(module, generic=True)

    def printer(value):
        parsed = parse_ir(value)
        for operation in parsed.walk():
            operation.properties = dict(sorted(operation.properties.items()))
            operation.attributes = dict(sorted(operation.attributes.items()))
        return text(parsed, generic=True)

    native_order = printer(upstream)
    assert native_order != upstream
    before, after = parse_ir(upstream), parse_ir(native_order)
    assert [(op.name, dict(op.properties), dict(op.attributes)) for op in before.walk()] == [
        (op.name, dict(op.properties), dict(op.attributes)) for op in after.walk()
    ]
    assert origin._canonical_source(upstream) == origin._canonical_source(native_order)
    with scope(tmp_path, passes=frozenset({"cse"}), upstream=upstream) as (owner, selected):
        fake_native_capture(tmp_path, owner, monkeypatch, actual_source=upstream, printer=printer)
        assert cse.submit(tmp_path) == 0 and selected.gate.verdicts == []


@pytest.mark.parametrize("mutation", ["value", "function_type", "symbol"])
def test_changed_property_values_remain_refused_after_metadata_sort(tmp_path, monkeypatch, mutation):
    original = source()
    position = original.rfind("}")
    upstream = original[:position] + "\nfunc.func private @contract_only(i32) -> i32\n" + original[position:]
    foreign_raw = {
        "value": upstream.replace("arith.constant 1 : index", "arith.constant 2 : index"),
        "function_type": upstream.replace("@contract_only(i32) -> i32", "@contract_only(i64) -> i32"),
        "symbol": upstream.replace("@contract_only", "@changed_contract"),
    }[mutation]
    foreign = text(parse_ir(foreign_raw), generic=True)
    assert origin._canonical_source(upstream) != origin._canonical_source(foreign)
    with pytest.raises(GateFailure, match="pre-canonicalize whole module"):
        with scope(tmp_path, passes=frozenset({"cse"}), upstream=upstream) as (owner, _):
            fake_native_capture(tmp_path, owner, monkeypatch, actual_source=upstream)
            directory = tmp_path / "leading-cse"
            for kind in ("precanonical", "upstream"):
                (directory / f"0001.{kind}.mlir").write_text(foreign)
            entries = json.loads((directory / "index.json").read_text())
            for key in ("precanonical_sha256", "upstream_generic_sha256"):
                entries[0][key] = hashlib.sha256(foreign.encode()).hexdigest()
            (directory / "index.json").write_text(json.dumps(entries))
            cse.submit(tmp_path)


@pytest.mark.parametrize("mutation", ["operation", "operand", "callee", "attribute", "type"])
def test_forged_precannon_evidence_cannot_relabel_foreign_whole_module(tmp_path, monkeypatch, mutation):
    import hashlib

    with pytest.raises(GateFailure, match="pre-canonicalize whole module"):
        with scope(tmp_path, passes=frozenset({"cse"})) as (owner, _):
            fake_native_capture(tmp_path, owner, monkeypatch)
            directory = tmp_path / "leading-cse"
            original = source()
            foreign_text = {
                "operation": original.replace("arith.addi", "arith.subi"),
                "operand": original.replace("arith.addi %observed0, %observed1", "arith.addi %observed0, %observed0"),
                "callee": original.replace("@touch", "@other_touch"),
                "attribute": original.replace("alignment = 16", "alignment = 32"),
                "type": original.replace("i32", "i64"),
            }[mutation]
            foreign = text(parse_ir(foreign_text), generic=True)
            assert origin._canonical_source(original) != origin._canonical_source(foreign)
            for kind in ("precanonical", "upstream"):
                (directory / f"0001.{kind}.mlir").write_text(foreign)
            entries = json.loads((directory / "index.json").read_text())
            for key in ("precanonical_sha256", "upstream_generic_sha256"):
                entries[0][key] = hashlib.sha256(foreign.encode()).hexdigest()
            (directory / "index.json").write_text(json.dumps(entries))
            cse.submit(tmp_path)


def test_malformed_precannon_evidence_is_safe_refusal(tmp_path, monkeypatch):
    import hashlib

    with pytest.raises(GateFailure, match="whole-module parse/render refused"):
        with scope(tmp_path, passes=frozenset({"cse"})) as (owner, _):
            fake_native_capture(tmp_path, owner, monkeypatch)
            directory = tmp_path / "leading-cse"
            malformed = "builtin.module { invalid MLIR syntax"
            for kind in ("precanonical", "upstream"):
                (directory / f"0001.{kind}.mlir").write_text(malformed)
            entries = json.loads((directory / "index.json").read_text())
            for key in ("precanonical_sha256", "upstream_generic_sha256"):
                entries[0][key] = hashlib.sha256(malformed.encode()).hexdigest()
            (directory / "index.json").write_text(json.dumps(entries))
            cse.submit(tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "nonce",
        "pipeline",
        "runner_sha256",
        "before_sha256",
        "schema",
        "upstream_sha256",
        "input_path",
        "precanonical_sha256",
        "upstream_generic_sha256",
    ],
)
def test_changed_capture_metadata_refuses(tmp_path, monkeypatch, field):
    with pytest.raises(GateFailure):
        with scope(tmp_path, passes=frozenset({"cse"})) as (owner, _):
            fake_native_capture(tmp_path, owner, monkeypatch)
            path = tmp_path / "leading-cse/index.json"
            entries = json.loads(path.read_text())
            entries[0][field] = "forged"
            path.write_text(json.dumps(entries))
            cse.submit(tmp_path)


def test_changed_actual_memory_CSE_is_unsupported_not_tensor_fallback(tmp_path, monkeypatch):
    with pytest.raises(GateFailure, match="changed original-memory CSE"):
        with scope(tmp_path, passes=frozenset({"cse"})) as (owner, _):
            fake_native_capture(tmp_path, owner, monkeypatch, changed=True)
            cse.submit(tmp_path)


def test_actual_runner_text_changes_refuse(tmp_path, monkeypatch):
    with pytest.raises(GateFailure, match="frozen memory runner"):
        with scope(tmp_path, passes=frozenset({"cse"})) as (owner, _):
            fake_native_capture(tmp_path, owner, monkeypatch)
            with (tmp_path / "run_lowering.py").open("a") as output:
                output.write("# mutation\n")
            cse.submit(tmp_path)


def test_missing_capture_refuses_on_successful_scope_exit(tmp_path):
    with pytest.raises(ValueError, match="consume its authentic CSE"):
        with scope(tmp_path, passes=frozenset({"cse"})):
            pass


def test_original_failure_is_not_masked_by_missing_capture_exit(tmp_path):
    with pytest.raises(RuntimeError, match="original failure"):
        with scope(tmp_path, passes=frozenset({"cse"})):
            raise RuntimeError("original failure")


def test_finish_without_validated_records_is_not_skip_authority(tmp_path):
    with pytest.raises((ValueError, OSError)):
        with scope(tmp_path, passes=frozenset({"cse"})) as (owner, _):
            owner.finish_cse(tmp_path)


def test_standalone_skip_only_legacy_behavior_is_unchanged(tmp_path):
    directory = tmp_path / cse.DIRECTORY
    directory.mkdir()
    (directory / "index.json").write_text(json.dumps([{"skipped": "bufferized module"}]))
    with active(tmp_path):
        with pytest.raises(GateFailure, match="no tensor-level canonicalize,cse"):
            cse.submit(tmp_path)


def test_saved_memory_records_outside_original_scope_refuse(tmp_path, monkeypatch):
    with active(tmp_path) as selected:
        with origin.compiler_scope(
            tmp_path, original_source=source(), memory_pipeline=memory_lowering_pipeline(), validation=selected
        ) as owner:
            owner.bind_upstream(source())
            (tmp_path / "model.mlir").write_text(source())
            fake_native_capture(tmp_path, owner, monkeypatch)
            assert cse.submit(tmp_path) == 0
        with pytest.raises(GateFailure, match="no original live memory scope"):
            cse.submit(tmp_path)


def test_memory_binding_requires_full_actual_pipeline_argument(tmp_path):
    with pytest.raises(ValueError, match="exact selected canonical pipeline"):
        with scope(tmp_path, passes=frozenset({"cse"})):
            cse.bind_runner("pass\n", tmp_path, upstream=(tmp_path / "model.mlir", source()))


def test_real_xdsl_preprocessing_DCE_before_joins_actual_formal_memory_epoch(tmp_path, monkeypatch):
    from buffer_rank_one_inputs import chain

    from merlin.llvmlower.passes_xdsl import preprocess_text
    from merlin.verify.compiler_memory_dce import CHECK

    actual = chain(tiled=False)[-1]
    captured = []
    original_observe, original_submit = pv.observe, pv.submit
    with active(tmp_path, passes=frozenset({"prune_dead_pure_tensor_ops"})) as selected:
        with origin.compiler_scope(
            tmp_path, original_source=actual, memory_pipeline=memory_lowering_pipeline(), validation=selected
        ) as owner:

            def observe_only(check, *, before, after, **options):
                assert check == CHECK
                assert pv.observe is original_observe and pv.submit is original_submit
                owner.validate_dce_before(before)
                captured.append((before, after))
                return None  # This host-only sink launches no proof job.

            # Preserve the now-source-bound real submit/observe dispatch functions.
            monkeypatch.setattr(selected.gate, "submit", observe_only)
            preprocess_text(actual)
    assert captured and any(before != after for before, after in captured)
    assert pv.observe is original_observe and pv.submit is original_submit


def test_derived_canonical64_pipeline_is_the_recorded_actual_string(tmp_path, monkeypatch):
    from merlin.llvmlower.pipeline import _bind_index_width

    pipeline = _bind_index_width(memory_lowering_pipeline(), 64)
    with scope(tmp_path, passes=frozenset({"cse"})) as (owner, _):
        fake_native_capture(tmp_path, owner, monkeypatch, pipeline=pipeline)
        assert cse.submit(tmp_path) == 0
        entry = json.loads((tmp_path / "leading-cse/index.json").read_text())[0]
        assert entry["effective_pipeline"] == pipeline
        assert entry["pipeline"] == "builtin.module(" + pipeline + ")"


def test_diagnostic_apply_body_and_unselected_runner_stay_legacy(tmp_path):

    actual = ast.parse((repo_root() / "src/merlin/llvmlower/leading_cse.py").read_text())
    apply_body = next(n for n in actual.body if isinstance(n, ast.FunctionDef) and n.name == "apply")
    # Fixed pre-integration diagnostic compatibility guard, not an execution proof.
    assert hashlib.sha256(ast.dump(apply_body, include_attributes=False).encode()).hexdigest() == (
        "a3c28874d2a2d50563279c1c9ebee9b7e0d0ec44a01c1487148e19f65098414c"
    )
    assert cse._PRELUDE.count("        after = operation.get_asm(print_generic_op_form=True)") == 1
    assert cse.bind_runner("unaltered", tmp_path) == "unaltered"


def test_lowering_scope_lexically_covers_preprocess_runner_and_submit():

    tree = ast.parse((repo_root() / "src/merlin/llvmlower/lower.py").read_text())
    scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "ExitStack"
            for item in node.items
        )
    ]
    assert len(scopes) == 1
    calls = [node for node in ast.walk(scopes[0]) if isinstance(node, ast.Call)]
    enter = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "compiler_scope")
    preprocess = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "preprocess")
    native = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "lower_to_llvm_ir")
    submit = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "submit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "leading_cse"
    )
    bind_upstream = next(
        node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "bind_upstream"
    )
    assert enter.lineno < preprocess.lineno < bind_upstream.lineno < native.lineno < submit.lineno
