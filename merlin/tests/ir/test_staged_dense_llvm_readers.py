"""The original context factory's code remains frozen across tool callbacks."""

import pytest
from test_staged_dense_llvm_build import setup

from merlin.common import invocation_record as I
from merlin.llvmlower import staged_dense_llvm as W


def test_first_tool_callback_cannot_change_same_context_factory_code(tmp_path, monkeypatch):
    original_run, calls = I.run, []

    def observed(argv, **kwargs):
        if not calls:
            calls.append("original tool callback")
            factory = W.make_llvm_context
            monkeypatch.setattr(factory, "__code__", factory.__code__.replace(co_name="changed"))
        return original_run(argv, **kwargs)

    monkeypatch.setattr(I, "run", observed)
    selection, _, _, _ = setup(tmp_path / "owner")
    assert not calls
    work = tmp_path / "transform"
    work.mkdir()
    with pytest.raises(ValueError, match="staged LLVM.*changed|replan"):
        selection.transform(tmp_path / "owner/original.ll", work)
    assert calls == ["original tool callback"]
