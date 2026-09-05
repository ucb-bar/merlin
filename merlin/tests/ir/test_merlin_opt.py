"""``merlin-opt`` is the tool every ``// RUN:`` line in this repo needs, so its contract is tested.

Three properties, each locking a real failure mode:

1. Every catalog pass is either registered or reported unregistrable WITH A REASON. A driver that
   silently exposes 10 of 12 passes is indistinguishable from one that has 10 — the same
   "a check that could not run reported success" shape this repo keeps hitting.
2. A registered pass actually transforms IR (the clone-and-return staged passes must have their
   result transplanted back into the module the pass API mutates).
3. A broken catalog entry fails startup rather than vanishing from the list.
"""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def _catalog():
    from merlin.xdsl_dialects.lowering import passes as P
    return P.all_catalogs()


def test_every_catalog_pass_is_registered_or_explained():
    from merlin.xdsl_dialects import opt

    ok, skipped = opt.merlin_passes()
    reasons = dict(skipped)
    for info in _catalog():
        assert info.name in ok or info.name in reasons, (
            f"{info.name} is neither registered nor reported: a pass cannot silently disappear")
    for name, reason in reasons.items():
        assert reason and reason.strip(), f"{name} skipped with an empty reason"


def test_unregistrable_passes_are_the_non_ir_ones():
    """The two dispatch-program stages consume an OutlineResult/DispatchProgram, not a ModuleOp.

    Locked because the temptation is to force them into ModulePass shape and fake a full catalog.
    Their declared dialects already say they are not IR-to-IR.
    """
    from merlin.xdsl_dialects import opt

    _, skipped = opt.merlin_passes()
    by_name = {p.name: p for p in _catalog()}
    for name, _reason in skipped:
        info = by_name[name]
        assert "dispatch-program" in (info.input_dialect + info.output_dialect), (
            f"{name} was skipped but its declared dialects claim it is IR-to-IR")


def test_registered_pass_transforms_the_module():
    """merlin-apply-schedule must leave schedule ops behind, in place."""
    from merlin.xdsl_dialects import contract, opt

    ok, _ = opt.merlin_passes()
    module = contract.build_example()
    before = _common.text(module)
    assert "schedule." not in before

    pass_cls = ok["merlin-apply-schedule"]()
    from xdsl.context import Context
    pass_cls().apply(Context(), module)

    after = _common.text(module)
    assert "schedule." in after, "the clone-and-return result was not transplanted back into op"


def test_broken_catalog_entry_fails_loudly():
    from merlin.xdsl_dialects import opt

    with pytest.raises(Exception):
        opt._resolve("merlin.xdsl_dialects.no_such_module.no_such_fn")
