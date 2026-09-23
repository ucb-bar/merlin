"""Whether a kernel's configuration is live where it is used -- and where a configuration is dead.

CONFIGURATION IS STATE, NOT STATEMENT ORDER. These accelerators are configured by instructions that set
a register and return: a stride, a dataflow, a transpose, an activation. Nothing in a later instruction
names the configuration it depends on, so the dependence exists only in the order the two were written.
A reordering, a hoist, or a tile whose stride differs from the last one configured produces a device
that is configured -- just not for this call. It returns wrong data, promptly, with nothing reported.

WHY THIS IS GENERIC AND NOT ONE TARGET'S PROBLEM. One target's binding carries this by hand, field by
field: "config_ld id 0 stride != the loop's", "config_st stride does not match", "config_ex transposes
differ from the loop's (or no config_ex)". Three spellings of one question -- is the value this call
assumes the value that is live? -- written out per field because the IR had no way to say it. Declared
as ``sets``/``assumes`` on the call, the question is asked once here for every target, and the per-target
work shrinks to naming the keys.

THE SECOND FINDING IS A SAVING, NOT A DEFECT. A configuration that is overwritten before any call
assumes it is DEAD: the instruction issued, cost its dispatch slot, and changed nothing that was read.
That is the elidable-config case, and it is reported separately from the misconfigurations because it
does not make the kernel wrong -- it makes it slower. Reporting the two in one list would force a caller
to choose between ignoring wrong answers and ignoring free speedups.

WHAT THIS DOES NOT DO. It does not know what any key MEANS. ``sets``/``assumes`` are opaque key/value
pairs compared for equality, so nothing here has an opinion about strides or dataflows -- which is what
keeps it target-free. A target that declares no configuration gets no findings rather than a default.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.sched.ir import Kernel, Loop
from merlin.sched.ir.expr import Expr, free_vars, render

__all__ = ["ConfigReport", "check_config"]


class _Varies:
    """A configuration whose value depended on a loop variable, seen from outside that loop.

    Not a value and not an absence. `sets stride=i` inside `for i in 0..5` leaves a different stride
    live on every iteration, and which one survives the loop is arithmetic this check does not do. The
    rendered text `"i"` is NOT that value: comparing it to a later `needs stride=i` in a different loop
    reports agreement between two unrelated numbers, and comparing it to `needs stride=4` reports a
    mismatch on a kernel that is correct. Both were measured. So the value is replaced by this at the
    loop's edge, and a read of it is reported as UNDECIDABLE rather than guessed in either direction.
    """

    __slots__ = ("var",)

    def __init__(self, var: str) -> None:
        self.var = var


def _text(value) -> str:
    return render(value) if isinstance(value, Expr) else repr(value)


@dataclass(frozen=True)
class ConfigReport:
    """What is wrong, what is wasted, and how much was looked at.

    ``problems`` are calls reading a configuration that is not live. ``dead`` are configurations no call
    read before they were replaced -- a cost, not an error.
    """

    problems: tuple[str, ...]
    dead: tuple[str, ...]
    #: Configuration keys the kernel uses at all. Zero means this kernel declares no configuration, in
    #: which case a clean report says nothing about it -- a caller printing only `ok` cannot tell an
    #: unconfigured kernel from a correctly configured one.
    keys: tuple[str, ...]
    #: Reads this check could not decide -- the live value varied with a loop that has since ended.
    #: Kept out of `problems` because it is not a defect, and out of silence because it is not clean.
    undecidable: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems


def _walk(body, live: dict, seen: dict, problems: list, dead: list, undecidable: list, keys: set, depth: int) -> None:
    """Interpret ``body`` in order, carrying the live configuration forward.

    ``live`` maps key -> (value, what set it). ``seen`` marks whether the live value has been read since
    it was set; an unread value that is replaced is the dead-configuration finding.
    """
    for stmt in body:
        if isinstance(stmt, Loop):
            # A loop body is walked TWICE when it can run twice, and the second pass is not redundant.
            # Iteration 0 sees the configuration left by the statements before the loop; every later
            # iteration sees what the loop's own TAIL left. Those are different states, so a body whose
            # head assumes a value its tail overwrites is correct on the first iteration and wrong on
            # all the rest:
            #
            #     config_a() sets stride=64
            #     for i in 0..4:
            #       mm()        needs stride=64      <-- right once, wrong three times
            #       config_b()  sets  stride=32
            #
            # An earlier version of this check walked the body once and justified it by claiming a
            # second pass would report nothing new. That was false, and this is the case it missed.
            # Two passes reach the fixpoint for this shape because the second pass starts from the
            # state the first one ended in, and a third would start from the same place again.
            _walk(stmt.body, live, seen, problems, dead, undecidable, keys, depth + 1)
            if stmt.extent > 1:
                _walk(stmt.body, live, seen, problems, dead, undecidable, keys, depth + 1)
            # Leaving the loop, any live value that mentioned its variable is no longer a value here.
            for key, (value, source) in list(live.items()):
                if isinstance(value, Expr) and stmt.var in free_vars(value):
                    live[key] = (_Varies(stmt.var), source)
            continue

        for key, want in stmt.assumes:
            keys.add(key)
            if key not in live:
                problems.append(
                    f"{stmt.instr}: needs {key}={_text(want)}, but no earlier call set {key}. The device "
                    "is configured for whatever ran before this kernel, which is not a property of this "
                    "kernel."
                )
                continue
            have, source = live[key]
            seen[key] = True
            if isinstance(have, _Varies):
                undecidable.append(
                    f"{stmt.instr}: needs {key}={_text(want)}, and the live value was set by {source} "
                    f"from an expression in {have.var!r}, so it differs per iteration of a loop that has "
                    "ended. Which value survives is arithmetic this check does not do -- reported as "
                    "undecidable rather than guessed."
                )
                continue
            if _text(have) != _text(want):
                problems.append(
                    f"{stmt.instr}: needs {key}={_text(want)}, but {source} left {key}={_text(have)} "
                    "live. The device is configured, just not for this call -- which returns wrong data "
                    "rather than an error."
                )

        for key, value in stmt.sets:
            keys.add(key)
            if key in live and not seen.get(key):
                prior, source = live[key]
                if not isinstance(prior, _Varies) and _text(prior) != _text(value):
                    dead.append(
                        f"{source}: set {key}={_text(prior)}, which {stmt.instr} replaced with "
                        f"{_text(value)} before any call read it. That configuration cost a dispatch "
                        "slot and changed nothing."
                    )
            live[key] = (value, stmt.instr)
            seen[key] = False


def check_config(kernel: Kernel) -> ConfigReport:
    """Configuration findings for ``kernel``, target-free.

    Takes no machine: a configuration key is whatever the target's binding chose to call it, and this
    compares values for equality without interpreting them. A machine parameter would suggest it
    validates keys against the device, which it does not and must not -- that belongs to the binding
    that knows what the keys mean.
    """
    problems: list[str] = []
    dead: list[str] = []
    keys: set[str] = set()
    undecidable: list[str] = []
    _walk(kernel.body, {}, {}, problems, dead, undecidable, keys, 0)
    # A loop body is walked twice, so a finding inside one is raised twice. Deduplicated by message
    # rather than by suppressing the second pass: which pass found it is an artefact of how this check
    # is written, and the reader should not have to know that a repeated line means a loop.
    return ConfigReport(
        problems=tuple(dict.fromkeys(problems)),
        dead=tuple(dict.fromkeys(dead)),
        undecidable=tuple(dict.fromkeys(undecidable)),
        keys=tuple(sorted(keys)),
    )
