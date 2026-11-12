"""Feature edit order must agree with native source runner order."""

import pytest

from merlin.llvmlower import fp8_narrowing as narrow, fp8_widening as widen
from merlin.llvmlower.loop_tensor_destinations import PRELUDE as LOOP_PRELUDE


def test_fp8_source_edits_commute_and_keep_narrowing_before_widening():
    passes = ["canonicalize", "one-shot-bufferize{bufferize-function-boundaries}"]
    expected = [narrow.MARKER, widen.MARKER, *passes]
    assert widen.edit_pipeline(narrow.edit_pipeline(passes)) == expected
    assert narrow.edit_pipeline(widen.edit_pipeline(passes)) == expected
    assert passes[0] == "canonicalize"
    for wrong in (["canonicalize", narrow.MARKER], [narrow.MARKER, narrow.MARKER]):
        with pytest.raises(ValueError): widen.edit_pipeline(wrong)


@pytest.mark.parametrize("loop_destinations", (False, True))
def test_composed_source_runners_keep_callbacks_and_consume_markers_once(loop_destinations):
    events, calls = [], []
    scope = {"_run_stages": lambda *args: calls.append(args)}
    exec(widen.PRELUDE, scope)
    exec(narrow.PRELUDE, scope)
    scope["_fp8_widen"] = lambda module: events.append("widen")
    scope["_fp8_narrow"] = lambda module: events.append("narrow")
    tail = "one-shot-bufferize{bufferize-function-boundaries}"
    if loop_destinations:
        exec(LOOP_PRELUDE, scope)
        scope["_ltd_reuse"] = lambda ctx, module: events.append("destinations")
        tail = "__merlin_loop_tensor_destinations__," + tail
    pipeline = narrow.MARKER + "," + widen.MARKER + ",canonicalize," + tail
    ctx, module = object(), object()
    callbacks = (7, ("mid",), ("late",), ("post",), ("pre",))
    scope["_run_stages"](ctx, module, pipeline, *callbacks)
    assert events == ["narrow", "widen", *(["destinations"] if loop_destinations else [])]
    assert scope["_F8N_CONSUMED"] and scope["_F8W_CONSUMED"]
    if loop_destinations:
        assert calls == [(ctx, module, "canonicalize", 0, (), (), (), ("pre",)),
            (ctx, module, "one-shot-bufferize{bufferize-function-boundaries}", 7, ("mid",), ("late",), ("post",), ())]
    else:
        assert calls == [(ctx, module, "canonicalize," + tail, *callbacks)]
