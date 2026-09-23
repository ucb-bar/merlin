"""What a command buffer ASKED the store path for, against what the target could DEMONSTRABLY absorb.

A whole-model emission decides, per contraction, how much of the layer travels with it. The
contraction itself is the easy part: every compiler finds it. What decides a model's wall clock is
the bias, the requantization, the activation and the pool that follow it -- per-element work that the
unit which just computed the accumulator can apply on its way out, and that nothing asks it to.
Measured on a deployable ResNet-50 for one target: 109 commands, 53 convolutions, and not one stage
of readout requested on any of them; the accelerator idle 96.84% of the window.

This module supplies the ORACLE for that, and deliberately not the answer. It compares two
statements about the same program:

``asked``
    read from the buffer's own commands -- the ``epilogue`` attribute on each store-path site, in
    the ABI's own vocabulary (``merlin/contract/command_buffer_abi.yaml``).
``admissible``
    derived, per site, by the capability-driven grouper that already exists in target-agnostic core
    (:mod:`~merlin.xdsl_dialects.lowering.compute_groups` closes a group only while the target's own
    contract admits the next stage, and :mod:`~merlin.xdsl_dialects.lowering.group_command` restates
    the closed group as the device program it asks for). The target's own readout facet licenses the
    stage templates (:func:`merlin.targetgen.readout_facet.epilogue_capability`).

What comes out is the DIFFERENCE, and :func:`agent_view` is the redaction that may be handed to a
compiler-writing agent: which store-path sites carry admissible readout the buffer left on the host,
and how much of it -- never the grouping that found them, never the geometry, never the multipliers.
Handing over the grouping would answer the question the experiment exists to ask.

THREE OUTCOMES, AND THE MIDDLE ONE IS THE POINT. ``gap`` and ``clean`` both require that a comparison
was actually possible. When the capability cannot be derived, when the alignment cannot be verified,
or when the capture licenses no stage at all, the verdict is ``incomplete`` -- which is never a pass.
A buffer that asks for nothing and a capture that licenses nothing produce identical numbers, and a
gate that cannot tell them apart scores a compiler on its input.

Nothing here names a target. The target is a parameter; the licensed stages, the granularities and
the clamp all come from its own derived readout, and the ABI vocabulary comes from the ABI.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from merlin.perf.gate_phase import STATUS_INCOMPLETE

SCHEMA = "epilogue_oracle_v1"

#: The comparison ran and found admissible readout the buffer did not ask for.
VERDICT_GAP = "gap"
#: The comparison ran and every admissible stage was asked for.
VERDICT_CLEAN = "clean"
#: The comparison could not be made. Carries ``why``. Never a pass, at any phase.
VERDICT_INCOMPLETE = STATUS_INCOMPLETE

#: The command attribute that names the readout a site asks for. One spelling, from the ABI.
EPILOGUE_ATTR = "epilogue"
#: A convolution command's window, ``[kh, kw, ci, co]`` (``command_buffer_abi.yaml``, CONV2D).
KERNEL_ATTR = "kernel"


class OracleError(ValueError):
    """The comparison was asked for in a form it cannot be made in."""


@dataclass(frozen=True)
class Ask:
    """One store-path site of a command buffer, as the buffer states it."""

    index: int
    opcode: str
    #: The stages the site asks the store path for, in the ABI's vocabulary.
    stages: tuple[str, ...]
    #: The site declared an ``epilogue`` and it was empty -- distinct from declaring none at all.
    declared_empty: bool
    #: ``(reduction extent, feature extent)`` where the command states its window, else ``None``.
    extents: tuple[int, int] | None


@dataclass(frozen=True)
class Admission:
    """One store-path site of the derived grouping, as the target's own contract admits it."""

    index: int
    op: str
    stages: tuple[str, ...]
    extents: tuple[int, int] | None


def _int_list(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        out.append(int(item))
    return out


def _command_extents(attributes: Mapping[str, Any]) -> tuple[int, int] | None:
    """``(reduction extent, feature extent)`` of a windowed command, from its declared kernel.

    The reduction a convolution performs is ``ci * kh * kw`` whatever form the target runs it in, and
    the features are its output channels. Both sides of this comparison can state those two numbers
    without agreeing on anything else -- which is exactly what makes them usable to VERIFY an
    alignment rather than to create one. A command that declares no kernel yields ``None`` and is
    aligned without a checksum, and the report says how many such sites there were.
    """
    kernel = _int_list(attributes.get(KERNEL_ATTR))
    if kernel is None or len(kernel) != 4:
        return None
    kh, kw, ci, co = kernel
    return (ci * kh * kw, co)


def asks(commands: Iterable[Mapping[str, Any]]) -> list[Ask]:
    """The store-path sites of ``commands``, in buffer order.

    A store-path site is a command whose readout NARROWS the accumulator into a declared container
    -- :data:`merlin.runtime.commandbuffer.NARROWING_OPCODES`, which is the ABI's own statement of
    which commands have a readout to configure. A command that computes into an accumulator and
    leaves it there (a resident matmul) is not a site; the commit that reads it out is.
    """
    from merlin.runtime.commandbuffer import EPILOGUE_STAGE_SET, NARROWING_OPCODES

    out: list[Ask] = []
    for index, command in enumerate(commands):
        opcode = command.get("opcode")
        attributes = command.get("attributes") or {}
        if not isinstance(attributes, Mapping):
            raise OracleError(f"command {index} declares attributes that are not a mapping")
        # A recorded SLICE of a buffer carries a site's attributes without its opcode. Such a site
        # is still a site: it was recorded because it is one. Dropping it would silently shrink the
        # population being judged, which is the failure this whole module is about.
        if opcode is not None and opcode not in NARROWING_OPCODES:
            continue
        raw = attributes.get(EPILOGUE_ATTR)
        if raw is not None and not isinstance(raw, (list, tuple)):
            raise OracleError(f"command {index} declares an {EPILOGUE_ATTR!r} that is not a list")
        stages = tuple(str(stage) for stage in (raw or ()))
        unknown = [stage for stage in stages if stage not in EPILOGUE_STAGE_SET]
        if unknown:
            raise OracleError(
                f"command {index} asks for {unknown}, which the ABI's epilogue vocabulary has no name for"
            )
        out.append(
            Ask(
                index=index,
                opcode=str(opcode) if opcode is not None else "",
                stages=stages,
                declared_empty=raw is not None and not stages,
                extents=_command_extents(attributes),
            )
        )
    return out


def _entry_extents(entry: Mapping[str, Any]) -> tuple[int, int] | None:
    op = entry.get("op")
    if op == "conv2d":
        try:
            return (int(entry["ci"]) * int(entry["kh"]) * int(entry["kw"]), int(entry["N"]))
        except (KeyError, TypeError, ValueError):
            return None
    try:
        return (int(entry["K"]), int(entry["N"]))
    except (KeyError, TypeError, ValueError):
        return None


def admissions(module: Any, target: str, *, weight_args: Any = None, oracle: Any = None) -> dict[str, Any]:
    """What ``target`` admits at each store-path site of ``module``, in the ABI's stage vocabulary.

    The grouping is the one already in core: growth stops at the first stage the target's own
    contract refuses, so a group's stages ARE the admitted ones -- this function only restates them.
    A group the device form cannot state is counted with its reason under ``unstatable`` and is never
    dropped, because a site nobody could state is not a site nobody needed.
    """
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command as GC

    groups = [g for g in CG.form_groups(module, target, oracle=oracle) if g.placement != CG.HOST]
    sites: list[Admission] = []
    unstatable: dict[str, int] = {}
    for position, group in enumerate(groups):
        try:
            entry = GC.program(group, weight_args=weight_args).entry
        except (CG.NoCapsuleForm, ValueError) as error:
            unstatable[f"{type(error).__name__}: {error}"] = unstatable.get(f"{type(error).__name__}: {error}", 0) + 1
            continue
        sites.append(
            Admission(
                index=position,
                op=str(entry.get("op") or ""),
                stages=tuple(str(stage) for stage in entry.get("epilogue") or ()),
                extents=_entry_extents(entry),
            )
        )
    return {"sites": sites, "unstatable": unstatable, "accelerator_groups": len(groups)}


def _align(ask_sites: Sequence[Ask], admitted: Sequence[Admission]) -> dict[str, Any]:
    """Pair sites in program order, VERIFIED by the extents both sides state independently.

    Order is what makes the pairing; the extents are what makes it checkable. Where both sides state
    a window, the two must agree -- a single disagreement refuses the whole alignment rather than
    pairing the rest, because a pairing that is wrong in one place is not evidence about any place.
    Sites left over on either end are reported, never dropped: a buffer site with no admission cannot
    be judged, and an admission with no buffer site is work that has no store path at all.
    """
    pairs: list[tuple[Ask, Admission]] = []
    checked = 0
    for ask, admit in zip(ask_sites, admitted):
        if ask.extents is not None and admit.extents is not None:
            if ask.extents != admit.extents:
                return {
                    "verified": False,
                    "why": (
                        f"site {ask.index} states extents {ask.extents} and the grouping's site "
                        f"{admit.index} states {admit.extents}; the two are not the same program"
                    ),
                    "pairs": [],
                    "checked": checked,
                    "unmatched_asks": [],
                    "unmatched_admissions": [],
                }
            checked += 1
        pairs.append((ask, admit))
    return {
        "verified": True,
        "why": None,
        "pairs": pairs,
        "checked": checked,
        "unmatched_asks": [a.index for a in ask_sites[len(pairs) :]],
        "unmatched_admissions": [a.index for a in admitted[len(pairs) :]],
    }


def compare(
    ask_sites: Sequence[Ask],
    admitted: Sequence[Admission],
    *,
    licensed: Sequence[str] | None,
    declined: Any = None,
    unstatable: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Score ``ask_sites`` against ``admitted``. ``licensed`` is the target's own stage vocabulary.

    ``licensed`` is ``None`` when the target's readout could not license an epilogue capability at
    all; the comparison is then ``incomplete``, because "the hardware cannot" and "nobody derived
    what the hardware holds" are different facts and only one of them is about the compiler.
    """
    unstatable = dict(unstatable or {})
    if licensed is None:
        return incomplete("the target's readout licenses no epilogue capability", ask_sites, admitted, unstatable)
    alignment = _align(ask_sites, admitted)
    if not alignment["verified"]:
        return incomplete(alignment["why"], ask_sites, admitted, unstatable)
    if alignment["unmatched_asks"]:
        return incomplete(
            f"{len(alignment['unmatched_asks'])} store-path site(s) of the buffer have no admission "
            f"to be judged against; the two statements are not about the same program",
            ask_sites,
            admitted,
            unstatable,
        )

    rows: list[dict[str, Any]] = []
    admissible_stages = asked_stages = 0
    for ask, admit in alignment["pairs"]:
        # Only what the target LICENSES counts as admissible. A grouper that grew past the licensed
        # vocabulary would otherwise inflate the denominator with work no readout can do.
        admissible = tuple(stage for stage in admit.stages if stage in set(licensed))
        carried = tuple(stage for stage in admissible if stage in ask.stages)
        omitted = tuple(stage for stage in admissible if stage not in ask.stages)
        admissible_stages += len(admissible)
        asked_stages += len(carried)
        rows.append(
            {
                "site": ask.index,
                "opcode": ask.opcode,
                "group": admit.index,
                "op": admit.op,
                "admissible": list(admissible),
                "asked": list(ask.stages),
                "omitted": list(omitted),
                "declared_empty_epilogue": ask.declared_empty,
            }
        )
    absent = [
        {
            "site": None,
            "opcode": "",
            "group": admitted[index].index,
            "op": admitted[index].op,
            "admissible": list(admitted[index].stages),
            "asked": [],
            "omitted": list(admitted[index].stages),
            "declared_empty_epilogue": False,
        }
        for index in range(len(alignment["pairs"]), len(admitted))
    ]
    for row in absent:
        admissible_stages += len(row["admissible"])
    rows += absent

    if not admissible_stages:
        return incomplete(
            "no store-path site of this capture admits any licensed stage, so a buffer that asks for "
            "nothing and a capture that licenses nothing are the same number",
            ask_sites,
            admitted,
            unstatable,
        )

    omitted_sites = [row for row in rows if row["omitted"]]
    # A site whose omission the buffer DECLARED is not silent. `declined` is the ABI's place to say
    # "could not lower this"; its absence beside an omitted admissible stage is the silent fallback.
    silent = (
        []
        if declined
        else [row["site"] if row["site"] is not None else f"group_{row['group']}" for row in omitted_sites]
    )
    return {
        "schema": SCHEMA,
        "verdict": VERDICT_GAP if omitted_sites else VERDICT_CLEAN,
        "why": None,
        "licensed_stages": list(licensed),
        "sites": rows,
        "alignment": {
            "paired": len(alignment["pairs"]),
            "extent_checked": alignment["checked"],
            "absent_sites": len(absent),
            "unstatable": unstatable,
        },
        "epilogue_share_on_store_path": round(asked_stages / admissible_stages, 6),
        "stages": {"admissible": admissible_stages, "asked": asked_stages, "omitted": admissible_stages - asked_stages},
        "omitted_sites": [row["site"] if row["site"] is not None else f"group_{row['group']}" for row in omitted_sites],
        "silent_fallbacks": silent,
    }


def incomplete(why: str, ask_sites: Sequence[Ask], admitted: Sequence[Admission], unstatable) -> dict[str, Any]:
    """An undecided comparison. Carries the numbers it DID reach, and no score it did not earn."""
    return {
        "schema": SCHEMA,
        "verdict": VERDICT_INCOMPLETE,
        "why": why,
        "licensed_stages": None,
        "sites": [],
        "alignment": {
            "paired": 0,
            "extent_checked": 0,
            "absent_sites": 0,
            "unstatable": dict(unstatable or {}),
            "buffer_sites": len(ask_sites),
            "admissions": len(admitted),
        },
        # NOT 0.0. A share nobody could compute is not a share of zero, and the difference is the
        # whole reason `incomplete` is a status rather than a failing score.
        "epilogue_share_on_store_path": None,
        "stages": {"admissible": None, "asked": None, "omitted": None},
        "omitted_sites": [],
        "silent_fallbacks": [],
    }


def gap(
    command_buffer: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    module: Any,
    target: str,
    *,
    weight_args: Any = None,
    oracle: Any = None,
) -> dict[str, Any]:
    """The whole comparison for one emission: what it asked, what the target admits, the difference.

    ``command_buffer`` is a buffer mapping (``commands`` / ``declined`` are read) or a bare sequence
    of commands. ``module`` is the model the buffer was built from, parsed.
    """
    if isinstance(command_buffer, Mapping):
        commands = command_buffer.get("commands") or ()
        declined = command_buffer.get("declined")
    else:
        commands, declined = list(command_buffer), None
    licensed = licensed_stages(target)
    admitted = admissions(module, target, weight_args=weight_args, oracle=oracle)
    report = compare(
        asks(commands),
        admitted["sites"],
        licensed=licensed,
        declined=declined,
        unstatable=admitted["unstatable"],
    )
    report["target"] = target
    return report


def licensed_stages(target: str) -> tuple[str, ...] | None:
    """The ABI stage names ``target``'s own readout licenses, or ``None`` when it licenses none.

    Two conditions, both derived, and neither one is enough alone:

    * a readout DECLARES it applies the stage (``ReadoutFacet.readouts[*]["applies"]``, extracted
      from the target's RTL), and
    * that readout's facet produces a quantized-epilogue capability
      (:func:`merlin.targetgen.readout_facet.epilogue_capability`), which is what says the
      accumulator width, the scale format, the rounding and the clamp are known.

    The second is the fail-closed half. A facet that declares ``acc_scale`` and cannot say what
    format its scale is held in has not been shown to license anything -- and scoring a compiler
    against limits nobody derived is how a gate comes to fail submissions for a missing toolchain.
    A declared stage the ABI's own vocabulary has no name for is dropped: no buffer could ask for it,
    so it can be neither carried nor omitted.
    """
    from merlin.runtime.commandbuffer import EPILOGUE_STAGE_SET
    from merlin.targetgen import readout_facet

    stages: list[str] = []
    for facet in readout_facet.for_target(target):
        try:
            readout_facet.epilogue_capability(facet)
        except ValueError:
            continue  # this unit's readout is not derived; another unit's may be
        for readout in facet.readouts:
            for stage in readout.get("applies") or ():
                if stage in EPILOGUE_STAGE_SET and stage not in stages:
                    stages.append(str(stage))
    return tuple(stages) or None


def agent_view(report: Mapping[str, Any], *, detail: bool = False) -> dict[str, Any]:
    """The redaction a compiler-writing agent may be handed: the GAP, never the grouping.

    The experiments this feeds exist to measure whether an agent can build the compiler. Handing it
    the grouping -- which operations close with which contraction, at what geometry, with what
    multiplier -- would hand it the answer and destroy the measurement. What is safe to disclose is
    the SHORTFALL: which store-path sites carry readout the target can demonstrably absorb and this
    emission did not ask for, how much of it, and the target's own licensed vocabulary (already
    public in its contract and in the ABI schema).

    ``detail`` additionally names the omitted stages per site. It is off by default because the
    stage names are most of the answer at a site that has one candidate; a caller turning it on is
    making that trade deliberately.
    """
    verdict = report.get("verdict")
    view = {
        "schema": SCHEMA,
        "target": report.get("target"),
        "verdict": verdict,
        "why": report.get("why"),
        "licensed_stages": report.get("licensed_stages"),
        "epilogue_share_on_store_path": report.get("epilogue_share_on_store_path"),
        "stages": report.get("stages"),
        "silent_fallbacks": list(report.get("silent_fallbacks") or ()),
    }
    view["sites"] = [
        {
            "site": row["site"],
            "opcode": row["opcode"],
            "omitted_stage_count": len(row["omitted"]),
            "has_store_path_command": row["site"] is not None,
            **({"omitted": list(row["omitted"])} if detail else {}),
        }
        for row in report.get("sites") or ()
        if row["omitted"]
    ]
    return view
