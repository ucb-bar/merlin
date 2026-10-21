"""Original permission/callback freshness for compact retained weights."""

from dataclasses import replace

import pytest
from test_compact_dense import fixture
from test_compact_pe_weights import ADAPTER_PE, encode, project, schedule

from merlin.compile.scheduling.compact_dense import emit_compact_dense
from merlin.compile.scheduling.compact_verification import verify_compact_dense
from merlin.transforms.dense_planning import dense_blocks


@pytest.mark.parametrize("boundary", ["encode", "project"])
def test_permission_mutation_is_rejected_at_the_first_callback(boundary):
    p, r, s = fixture()
    good = schedule(p, r, s)
    calls = []
    adapter = None

    def changed(op):
        calls.append(1)
        object.__setattr__(adapter, "retained_weights", False)
        return encode(op) if boundary == "encode" else project(op)

    adapter = replace(ADAPTER_PE, **{boundary: changed})
    with pytest.raises(ValueError, match="permissions or callbacks changed"):
        if boundary == "encode":
            emit_compact_dense(
                p, dense_blocks(p, r)[-1], s, adapter, name="dense", dtypes=("i8", "i8", "i32"), reuse_pe_weights=True
            )
        else:
            verify_compact_dense(good.kernel, p, r, s, adapter, static_budget=100, timeout_ms=20_000)
    assert len(calls) == 1


def test_default_metadata_mutation_of_the_same_callback_is_rejected():
    p, r, s = fixture()
    calls = []

    def changed(operation, permission=False):
        calls.append(1)
        changed.__defaults__ = (True,)
        return encode(operation)

    adapter = replace(ADAPTER_PE, encode=changed)
    with pytest.raises(ValueError, match="permissions or callbacks changed"):
        emit_compact_dense(
            p, dense_blocks(p, r)[-1], s, adapter, name="dense", dtypes=("i8", "i8", "i32"), reuse_pe_weights=True
        )
    assert len(calls) == 1


def test_replacing_bound_callback_owner_refuses_before_replacement_executes():
    p, r, s = fixture()
    calls = []

    class Owner:
        def project(self, call):
            calls.append(self)
            return project(call)

    original, replacement = Owner(), Owner()
    adapter = None

    def changed(operation):
        object.__setattr__(adapter, "project", replacement.project)
        return encode(operation)

    adapter = replace(ADAPTER_PE, encode=changed, project=original.project)
    with pytest.raises(ValueError, match="permissions or callbacks changed"):
        emit_compact_dense(
            p, dense_blocks(p, r)[-1], s, adapter, name="dense", dtypes=("i8", "i8", "i32"), reuse_pe_weights=True
        )
    assert calls == []


def test_current_qualification_changes_invalidate_the_issued_result():
    p, r, s = fixture()
    adapter = replace(ADAPTER_PE)
    result = schedule(p, r, s, adapter=adapter)
    object.__setattr__(adapter, "retained_weights", False)
    with pytest.raises(ValueError, match="permissions or callbacks changed"):
        result.revalidate()


@pytest.mark.parametrize("callback", ["revalidate", "record"])
def test_selected_owner_cannot_change_original_adapter_at_result_revalidation(tmp_path, monkeypatch, callback):
    from test_selected_dense_contracts import inputs
    from test_selected_dense_contracts import schedule as selected_schedule

    model = inputs(tmp_path)
    selected, _, _, adapter = model
    result = selected_schedule(model)
    original = getattr(type(selected), callback)
    calls = []

    def changed(self):
        value = original(self)
        calls.append(1)
        object.__setattr__(adapter, "retained_weights", True)
        return value

    monkeypatch.setattr(type(selected), callback, changed)
    with pytest.raises(ValueError, match="permissions or callbacks changed"):
        result.revalidate()
    assert calls
