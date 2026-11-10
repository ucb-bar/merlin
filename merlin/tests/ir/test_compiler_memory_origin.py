"""Bounded memory-DCE ownership and native input/runner custody controls.

CheckerTools.__post_init__ is intentionally not called: this is not toolchain
qualification and no worker job may be submitted by these tests.
"""

import copy
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from pathlib import Path

import pytest
from buffer_rank_one_inputs import source

from merlin.transforms.certificates import parse_ir
from merlin.verify import compiler_memory_origin as origin
from merlin.verify import pass_validation as pv
from merlin.verify.compiler_transforms import memory_lowering_pipeline
from merlin.verify.speculative_gate import SpeculativeGate
from merlin.xdsl_dialects._common import text


@contextmanager
def active(work, *, passes=frozenset({"prune_dead_pure_tensor_ops"})):
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


@pytest.mark.parametrize("copying", ([copy.copy, copy.deepcopy])[:1])
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


def bind_native_inputs(work, owner, *, pipeline=None):
    source_path = work / "model.mlir"
    owner.bind_input((source_path, source_path.read_text()), work)
    issued = owner.bind_runner("# ordinary upstream runner\n", work, pipeline=pipeline or memory_lowering_pipeline())
    (work / "run_lowering.py").write_text(issued)
    return issued


@pytest.mark.parametrize("changed", (["model.mlir", "run_lowering.py"])[:1])
def test_native_inputs_cannot_change_before_delivery(tmp_path, changed):
    with pytest.raises(ValueError, match="input file changed|frozen memory runner"):
        with scope(tmp_path) as (owner, _):
            bind_native_inputs(tmp_path, owner)
            (tmp_path / changed).write_text("changed bytes")


def test_original_error_is_not_masked_by_delivery_guard(tmp_path):
    with pytest.raises(RuntimeError, match="original compile failure"):
        with scope(tmp_path) as (owner, _):
            bind_native_inputs(tmp_path, owner)
            (tmp_path / "model.mlir").write_text("changed bytes")
            raise RuntimeError("original compile failure")


def test_foreign_input_path_and_changed_pipeline_refuse(tmp_path):
    with scope(tmp_path) as (owner, _):
        other = tmp_path / "copy.mlir"
        other.write_text((tmp_path / "model.mlir").read_text())
        with pytest.raises(ValueError, match="frozen upstream epoch"):
            owner.bind_input((other, other.read_text()), tmp_path)
        with pytest.raises(ValueError, match="canonical pipeline"):
            bind_native_inputs(tmp_path, owner, pipeline="canonicalize,cse")


def test_explicit_index64_pipeline_retains_native_input_custody(tmp_path):
    from merlin.llvmlower.pipeline import _bind_index_width

    with scope(tmp_path) as (owner, _):
        issued = bind_native_inputs(tmp_path, owner, pipeline=_bind_index_width(memory_lowering_pipeline(), 64))
        assert owner.validate_runner(tmp_path) == hashlib.sha256(issued.encode()).hexdigest()
        assert not (tmp_path / "leading-cse").exists()


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
            # The actual epoch join is exercised above; a capture-only sink
            # cannot authorize a changed boundary under the coverage controller.
            with pytest.raises(ValueError, match="actual proof job"):
                preprocess_text(actual)
    assert captured and any(before != after for before, after in captured)
    assert pv.observe is original_observe and pv.submit is original_submit
