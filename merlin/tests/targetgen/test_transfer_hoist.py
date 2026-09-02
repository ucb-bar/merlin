"""The `PC` lever: issue stage two's transfer before stage one's wait.

The claim is a DIFFERENTIAL over a paired run -- the same dependence chain with and without the hoist --
so the pair is only worth measuring if the two programs differ in ORDER and in nothing else. Both halves
are checked here, because each fails in its own way: if the order does not change the delta is zero by
construction, and if the instruction multiset changes the delta is attributable to the extra or missing
instruction rather than to the ordering.

⚠️ WHY THIS IS SOUND ONLY ON A TARGET WHOSE COMPLETION IS EXPLICIT. The hoisted body reuses the DMA
argument registers for the second issue before the first transfer has been awaited, which is correct only
if the controller latched its descriptor when it accepted the command. That is a DERIVED fact, not an
assumption about DMA engines: gemmini's `explicit_completion` trait is satisfied from its own elaborated
FIRRTL (LoadController and StoreController each expose a decoupled completion channel), and on a target
where the trait is UNKNOWN the family gates off rather than running a program nobody established the
correctness of.
"""
from __future__ import annotations

import sys
from collections import Counter

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "tests" / "targetgen"))

import test_layer_workload_gen as LW  # noqa: E402
from merlin.perf import workload_gen as WG  # noqa: E402

_KW = dict(control_flow=LW.CF, settle=LW.SETTLE, subnormal_operand_flush=False)


def _plan(**extra):
    return WG.plan_matmul(LW.synthetic_facts(), LW.OPS, m=4, k=8, n=4, **_KW, **extra)


def _instructions(plan) -> list[str]:
    """The emitted instructions in order, read from the assembly's own per-word annotation."""
    out = []
    for line in plan.kernel_s.splitlines():
        if "#" not in line or "]" not in line:
            continue
        out.append(line.split("#", 1)[1].split("]", 1)[1].strip())
    return out


def test_the_hoist_preserves_the_instruction_multiset():
    """Same instructions, same counts. This is what the comparand cancels.

    The second wait is deliberately KEPT in the hoisted body even though the channel is already idle by
    then and it retires immediately: dropping it would make the pair differ by an instruction, and the
    cycle delta would stop isolating the ordering.
    """
    base, hoist = _plan(), _plan(hoist_transfers=True)
    a = Counter(i.split()[0] for i in _instructions(base))
    b = Counter(i.split()[0] for i in _instructions(hoist))
    assert a == b, f"multiset differs: only-in-base={a - b}, only-in-hoisted={b - a}"
    assert len(base.words) == len(hoist.words), "the two programs must be the same length"


def test_the_hoist_actually_changes_the_order():
    """The over-correction mirror: a knob that preserves the multiset by doing nothing at all.

    Checked structurally on the issue/wait positions rather than by comparing text: the unhoisted body
    is strictly serialized (issue, wait, issue, wait) and the hoisted one has both issues outstanding
    before the first wait.
    """
    base, hoist = _plan(), _plan(hoist_transfers=True)
    assert base.kernel_s != hoist.kernel_s, "the knob emitted an identical program"

    def positions(plan, mnemonic):
        return [i for i, ins in enumerate(_instructions(plan)) if ins.split()[0] == mnemonic]

    b_load, b_wait = positions(base, LW.OPS.dma_load), positions(base, LW.OPS.dma_wait)
    h_load, h_wait = positions(hoist, LW.OPS.dma_load), positions(hoist, LW.OPS.dma_wait)
    assert len(b_load) >= 2 and len(h_load) >= 2, "need two transfers to hoist one past the other"

    # unhoisted: the first wait precedes the second issue -- strictly serialized.
    assert b_wait[0] < b_load[1], (
        f"the unhoisted body is not serialized (load {b_load[:2]}, wait {b_wait[:2]}); there is then "
        f"nothing for the hoist to be a hoist against")
    # hoisted: both issues precede the first wait.
    assert h_load[1] < h_wait[0], (
        f"stage two's transfer was NOT issued before stage one's wait (load {h_load[:2]}, "
        f"wait {h_wait[:2]})")


def test_the_unhoisted_program_is_unchanged_by_the_knob_existing():
    """Default off, byte-identical. The lever must not quietly re-shape every existing workload."""
    assert _plan().kernel_s == _plan(hoist_transfers=False).kernel_s


def test_the_hoist_is_confined_to_the_transfer_pair():
    """Every tensor op keeps its place relative to the others, so only the transfers moved."""
    base, hoist = _plan(), _plan(hoist_transfers=True)
    tensor = {LW.OPS.tile_load, LW.OPS.transpose, LW.OPS.weight_push,
              LW.OPS.contract, LW.OPS.contract_accumulate, LW.OPS.acc_read}
    seq_a = [i.split()[0] for i in _instructions(base) if i.split()[0] in tensor]
    seq_b = [i.split()[0] for i in _instructions(hoist) if i.split()[0] in tensor]
    assert seq_a == seq_b, "the tensor-op order changed; the hoist must move transfers only"
