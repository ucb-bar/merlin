"""The per-op MR caps a SEARCH may name must be registered EAGERLY, and MR=1 must be a rung.

Two separate failure modes are pinned here, both of which this repo has already paid for once on the
neighbouring register-block family:

1. **Lazily-registered is silently unproposable.** ``perop_register_block_mr<N>`` was minted only
   inside ``wholemodel_proposer.refinement_forks``. Anywhere else -- a ``--features`` string, a
   package's ``compiler_features``, a fresh process asking ``_composes`` whether the cap composes --
   the name resolved to a ``KeyError``, which ``_composes`` swallows and reports as "does not
   compose". A cap that reads as DECLINED rather than as ABSENT is exactly the shape of bug
   ``test_every_lever_is_reachable`` exists to catch, one layer down (that test enumerates lever
   MODULES; this one is about a name minted on demand inside a module that IS imported).

2. **The expert's own MR was not in the ladder.** XNNPACK's int8 GEMM is
   ``xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4v__rvv`` -- MR=1 -- and the lifted expert CCA reads
   ``compute.register_block (1, ('vsetvlmax', 4.0))``. ``MRPAD_INT8_TILES`` says in its own comment
   that its ladder "started at MR=2" and that "the expert's MR was not in the search space at all";
   the per-op ladder had the same hole, which left the expert's M reachable only through the
   fixed-tile family -- and that family REPLACES the derived per-op N as well as the M, so the two
   axes could never be varied independently.

Registration is default-off, so none of this moves the frozen baseline: the sentinel carries no
schedule or cflags hook and raises if it ever reaches lowering unresolved.
"""
from __future__ import annotations

import json
import subprocess
import sys

from merlin.common.paths import merlin_dir, repo_root
from merlin.llvmlower import impr_features as F


def test_the_ladder_is_registered_at_import_not_on_demand():
    """A FRESH interpreter that imports only ``impr_features`` must already know every rung.

    Importing the proposer first would mask the bug, because the proposer mints the names itself.
    """
    code = (
        "import json\n"
        "from merlin.llvmlower import impr_features as F\n"
        "known = F.known()\n"
        "print(json.dumps([n for n in F.PEROP_MR_LADDER if n not in known]))\n"
    )
    env = {"PYTHONPATH": str(merlin_dir() / "python"), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=repo_root(), env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    missing = json.loads(r.stdout.strip().splitlines()[-1])
    assert missing == [], f"rungs registered only on demand, so unreachable by name: {missing}"


def test_every_rung_round_trips_to_its_cap():
    """The name is the join key between the feature set and ``block_table``'s MR cap."""
    caps = [F.parse_perop_mr_sentinel(n) for n in F.PEROP_MR_LADDER]
    assert None not in caps, f"a rung does not parse as a cap sentinel: {F.PEROP_MR_LADDER}"
    assert caps == sorted(caps) and len(set(caps)) == len(caps)
    for name, cap in zip(F.PEROP_MR_LADDER, caps):
        assert F.perop_mr_sentinel(cap) == name


def test_the_experts_mr_is_a_rung_on_both_ladders():
    """MR=1 is the expert's own register block; a search that cannot name it cannot re-test it."""
    from merlin.mining.wholemodel_proposer import _MR_CAP_LADDER

    assert 1 in _MR_CAP_LADDER, "the proposer cannot propose the expert's MR=1 per-op block"
    assert F.parse_perop_mr_sentinel(F.PEROP_MR_LADDER[0]) == 1


def test_the_default_cap_is_not_given_a_second_spelling():
    """``perop_register_block`` already derives under ``zephyr_model.perop_mr_cap()``.

    Registering that same cap again as a named rung would make two identical builds look like two
    arms of an A/B -- the search would spend a whole whole-model lowering to rediscover its parent.
    """
    from merlin.runtime.backends.zephyr_model import perop_mr_cap

    assert perop_mr_cap() not in [F.parse_perop_mr_sentinel(n) for n in F.PEROP_MR_LADDER]


def test_a_rung_composes_with_the_champion_stack_minus_its_own_block():
    """The cap REPLACES the plain sentinel (both emit a full schedule), so the caller must drop it.

    Pinned because the composition rule is the thing that turns a mis-built candidate into a
    ``CompositionError`` instead of into a silently mis-attributed measurement.
    """
    from merlin.mining.wholemodel_proposer import _composes

    champion = ["prepack_weight_layout", "perop_register_block", "promote_buffers_to_stack",
                "expand_memref_copy", "cse_through_provenance"]
    base = [f for f in champion if f != "perop_register_block"]
    for name in F.PEROP_MR_LADDER:
        assert _composes(base + [name]), f"{name} does not compose onto the champion base"
        assert not _composes(champion + [name]), (
            f"{name} stacked on the plain sentinel must be refused -- two full schedule "
            "replacements cannot both apply")


def test_the_sentinel_still_refuses_to_reach_lowering():
    """Default-off and inert by construction: it is a REQUEST, consumed at preparation time."""
    import pytest

    for name in F.PEROP_MR_LADDER:
        feat = F.get(name)
        assert feat.edit_schedule is None
        with pytest.raises(RuntimeError, match="reached the lowering pipeline unresolved"):
            feat.edit_pipeline([])
