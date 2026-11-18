"""Put a WHOLE WINDOW of dataflow-connected groups to a package as ONE region capsule.

:mod:`.region_legality` says which runs of consecutive groups a package MAY be offered together; this
module is what makes the offer and reads the answer. It reuses exactly the pieces
:func:`merlin.compiler.lower.device.whole_program._ask_package` already uses for one group -- the same
``corpus_spec.build`` per-op writers, the same ``capsule_common.lower_interface`` call, the same
``_spliced`` role-based binding -- so a region is not a second grammar: it is several of the SAME
single-op capsules, textually joined at the one seam each pair of neighbours shares, put to the
package in one call instead of several.

**The seam, and why nothing has to be renamed to make it.** Each member's own per-op capsule text
already reads its "activation" operand by whatever name its entry declares (``lhs`` for a
contraction, ``ifm`` for a convolution, either operand of a residual add). Before any text is
written, this module sets a NEIGHBOUR PAIR's shared boundary to one name: the producer's own ``out``
and the consumer's one operand that the whole-program statement already resolved to the producer's
tensor (found structurally, off the SAME generic-to-ABI rename `_as_declared` already performs --
never a second table of "which key names the activation for op X"). Two per-op bodies that agree on
one SSA name splice into one region exactly as :func:`corpus_spec.build_residual_seam` already writes
a contraction and a residual add by hand, except the members here are found, not authored.

**A region is FUSED: one kernel, graded at its boundary.** The package answers the whole window with
ONE kernel that reads the window's external operands and commits the BOUNDARY member's tensor (the
last member's -- the only one anything outside the window reads, which is what made the window legal).
Every internal member's tensor is the kernel's own business: the package may stage it in the program
buffer it is bound to, keep it on the accelerator, or never form it at all -- how is the package's to
decide, and nothing here prefers one way. What the region computes is judged at its boundary, against
the oracle, as one claim
(:func:`merlin.analysis.whole_model.build.stages.bind_groups` links it as one call; the target driver restates
each internal member from the inputs the device held and grades the boundary from them).

**Coverage is earned at the boundary, not by opcode.** A reply is accepted for the window only when
the boundary's own program tensor is committed BY NAME after splicing; an internal member's output is
bound to its program buffer when the reply declares it and is never required. There is deliberately no
check that each member's opcode appears among the commands: a fused kernel need not spell a member as
its own command, and a kernel that silently skipped one computes a wrong boundary, which the region's
correctness check catches -- a by-opcode check would refuse honest fusion and pass nothing a wrong
answer could not also pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

# ---- region_capsule ----------------------------------------------------------------------------


def region_output_name(position: int) -> str:
    """The capsule-level output name of a region's member at ``position`` -- zero-padded and strictly
    increasing, so the names sort in member order and a reply is bound by them, never by guess."""
    return f"__region_out_{position:04d}"


def split_region_commands(
    commands: Sequence[Mapping[str, Any]], member_dst_names: Sequence[str]
) -> list[list[Mapping[str, Any]]] | None:
    """``commands`` partitioned into one contiguous, in-order share per name in ``member_dst_names``.

    Each share but the last ends at (and includes) the first command that commits its member's own
    ``dst``; the last share absorbs everything from there to the end of ``commands``, so the shares
    always sum back to the whole list -- the invariant every per-group grader
    (:class:`merlin.analysis.whole_model.build.stages.LocalReference`) already holds every group to. ``None`` when
    some member's own ``dst`` is never committed by any command at or after its share's start: the
    region did not truly produce that member, and the caller must refuse it whole rather than credit a
    share that would otherwise be empty.
    """
    shares: list[list[Mapping[str, Any]]] = []
    start = 0
    for name in member_dst_names[:-1]:
        split_at = None
        for position in range(start, len(commands)):
            operands = commands[position].get("operands") if isinstance(commands[position], Mapping) else None
            if isinstance(operands, Mapping) and operands.get("dst") == name:
                split_at = position
                break
        if split_at is None:
            return None
        shares.append(list(commands[start : split_at + 1]))
        start = split_at + 1
    last_name = member_dst_names[-1]
    committed_last = any(
        isinstance(c, Mapping) and isinstance(c.get("operands"), Mapping) and c["operands"].get("dst") == last_name
        for c in commands[start:]
    )
    if not committed_last:
        return None
    shares.append(list(commands[start:]))
    return shares


def _declared_tensor_name(line: str) -> str | None:
    """The tensor a ``merlin_iface.tensor {name = "X", ...}`` declaration line names, else ``None``.

    Structural: a fixed marker string is located and the name is read between the quotes that follow
    it, never a pattern matched against the whole line.
    """
    marker = 'merlin_iface.tensor {name = "'
    if marker not in line:
        return None
    _before, _, rest = line.partition(marker)
    name, _, _after = rest.partition('"')
    return name or None


def _member_body(text: str) -> tuple[str, list[str]]:
    """``(the module-attributes prelude line, the body lines between it and the closing brace)``."""
    lines = text.splitlines()
    start = 1 if lines and lines[0].startswith("//") else 0
    return lines[start], lines[start + 1 : -1]


def _linking_key(next_opcode: str, next_operands: Mapping[str, str], producer_dst: str) -> list[str]:
    from merlin.compiler.lower.device import whole_program as WP

    declared = WP._as_declared(next_opcode, dict(next_operands))
    return [key for key, value in declared.items() if key != "dst" and value == producer_dst]


def ask_package_region(
    package,
    members: Sequence[Mapping[str, Any]],
    device: str,
    work: str | Path,
    timeout: int,
    run,
    *,
    shapes: Mapping[str, Mapping] | None,
    views: list | None,
    record: dict | None,
    binder=None,
) -> tuple[list[dict], dict, str, str, dict | None]:
    """Put ``members`` (>= 2 groups, in dataflow order) to ``package`` as one region capsule.

    Each of ``members`` carries the same four keys the whole-program statement already computed for
    that single group before ever asking anyone: ``tag``, ``entry`` (the op's own geometry, as
    :func:`merlin.compiler.partition.groups.group_command.program` states it), ``operands`` (THIS
    PROGRAM's own buffer names for the group's operands -- never the capsule's local ones), and
    ``opcode`` (the command-buffer opcode the entry states).

    Returns exactly what :func:`merlin.compiler.lower.device.whole_program._ask_package` returns for one group --
    ``(commands, scratch, why, cause)`` -- with a fifth element: ``None`` when the region was not
    asked or not accepted (the caller then asks each member on its own, exactly as an unspliced
    single group always has), else the region's own record (its id, its members in order, and which
    one is the BOUNDARY -- the only member whose output is still checked as this program's own named
    tensor).
    """
    from merlin.compiler.lower.device import whole_program as WP
    from merlin.development.targetgen.corpus import capsule_io as CC
    from merlin.development.targetgen.packaging import loading as OR

    def refuse(cause: str, why: str) -> tuple[list[dict], dict, str, str, None]:
        commands, scratch, why_, cause_ = WP._refuse(cause, why)
        return commands, scratch, why_, cause_, None

    members = list(members)
    if len(members) < 2:
        return refuse(WP.NOT_ASKED, "a region needs at least two members")

    # EVERY MEMBER'S OWN CAPSULE-LEVEL OUTPUT NAME, assigned here rather than left to corpus_spec's
    # per-op default -- a zero-padded, strictly increasing string per position, so that sorting these
    # names ALPHABETICALLY recovers member order exactly. That is what lets the OUTPUT BINDING below
    # pair the capsule's declared tensors with this program's own buffers by POSITION rather than by
    # `_spliced`'s ordinary (and here ambiguous) alphabetical-then-positional rule, which assumes a
    # single tensor per role -- a region binds one PER MEMBER.
    entries = [dict(m["entry"]) for m in members]
    out_names = [region_output_name(position) for position in range(len(members))]
    for entry, name in zip(entries, out_names, strict=True):
        entry["out"] = name

    # THE SEAM NAMES: one per neighbour pair, found by the SAME rename `_as_declared` already
    # performs for a single group's ABI-facing operands -- never a second, per-op table of "which key
    # is the activation". Reusing each producer's own OUTPUT NAME as the seam IS the link: the
    # consumer's re-declaration of the same name is dropped below, so its later ops read the
    # producer's own committed value.
    for i in range(len(members) - 1):
        producer_dst = str(members[i]["operands"]["dst"])
        keys = _linking_key(str(members[i + 1]["opcode"]), members[i + 1]["operands"], producer_dst)
        if len(keys) != 1:
            return refuse(
                WP.NOT_ASKED,
                f"member {members[i + 1]['tag']} does not read member {members[i]['tag']}'s output "
                f"through exactly one declared operand (found {keys}); a region is only askable when "
                f"its internal seam is unambiguous",
            )
        entries[i + 1][keys[0]] = out_names[i]
    # A seam is every OUT NAME but the last: the last member's own output is not read by anything
    # inside this region and is the one this deliverable's correctness rule checks at the boundary.
    seams = set(out_names[:-1])

    # EACH MEMBER'S OWN CAPSULE TEXT, exactly as a single-group ask builds it -- the seam names above
    # are the only thing distinguishing this from N independent asks.
    bodies: list[list[str]] = []
    declared_opcodes: list[str] = []
    head: str | None = None
    for member, entry in zip(members, entries, strict=True):
        # THE SAME WIDTH DECLARATION A SINGLE-GROUP ASK MAKES: left undeclared, the capsule builder
        # derives a commit width from the epilogue alone, and a member that leaves as the accumulator
        # would be asked for a narrow commit into a buffer this program declares wider.
        capsule_entry = dict(entry) if "output_dtype" in entry else {**entry, "output_dtype": member["output_dtype"]}
        try:
            # THE ONE GROUP CAPSULE PATH (see `whole_program.group_interface`): the member is stated
            # exactly as the corpus states it, under the corpus binding -- never a binding built here.
            text = WP.group_interface(capsule_entry, binder, source_reference=f"region member {member['tag']}")
        except Exception as error:  # noqa: BLE001 -- recorded per region, never swallowed
            return refuse(WP.CAPSULE_FAILED, f"region member {member['tag']}: {type(error).__name__}: {error}")
        prelude, body = _member_body(text)
        bodies.append(body)
        declared_opcodes.append(str(member["opcode"]))
        if head is None:
            head = prelude

    # STITCHED: one module, one prelude, every member's body in order -- except a member's own
    # re-declaration of a name the PREVIOUS member already produced, which would define that SSA name
    # twice and, read the other way, would describe the seam as an external leaf it is not.
    merged: list[str] = [str(head)]
    for body in bodies:
        for line in body:
            declared_name = _declared_tensor_name(line)
            if declared_name is not None and declared_name in seams:
                continue
            merged.append(line)
    merged.append("}")

    tag = f"region_{members[0]['tag']}_{members[-1]['tag']}"
    work = Path(work)
    source = work / f"{tag}.iface.mlir"
    source.write_text("\n".join(merged) + "\n", encoding="utf-8")
    if record is not None:
        record["interface"] = str(source)

    try:
        emitted, artifact = CC.lower_interface(
            package,
            source,
            work / f"{tag}.generated",
            contract=None,
            timeout=timeout,
            invoke=run or OR.run_entrypoint,
            artifact_name=f"{tag}.artifact.txt",
        )
    except OR.BackendDeclined as declined:
        return refuse(WP.PACKAGE_DECLINED, f"the package declined this region: {str(declined)[:200]}")
    except OR.CertFailure as failure:
        cause = WP.MALFORMED if str(getattr(failure, "plane", "")) == "command_buffer_schema" else WP.PACKAGE_FAILED
        return refuse(cause, f"the package did not lower this region: {str(failure)[:300]}")
    except Exception as error:  # noqa: BLE001
        return refuse(WP.PACKAGE_FAILED, f"the package did not run: {type(error).__name__}: {error}")
    from merlin.verification.lean.emission_custody import write_emitted_artifact

    write_emitted_artifact(work / f"{tag}.artifact.txt", artifact)
    if record is not None:
        # THE SAME TWO PATHS A SINGLE-GROUP ASK RECORDS (`whole_program._ask_package`), in the SAME
        # place -- `bind_groups` reads `asked["command_buffer"]` for EVERY package-answered row,
        # region member or not, to resolve its kernel's own argument order. Left unset here, a region
        # that succeeds crashes `bind_groups` with `KeyError: 'command_buffer'` for every member --
        # including a build where `allow_regions` is on but no package ever asked for one; the crash
        # depends only on some structurally legal window succeeding, never on a package's own opt-in.
        record["command_buffer"] = str(work / f"{tag}.generated" / "command_buffer.json")
        record["artifact"] = str(work / f"{tag}.artifact.txt")

    # BOUND ROLES ACROSS THE WHOLE WINDOW: every member's TRUE external operand -- never one consumed
    # from an EARLIER member INSIDE this region, which is this region's own private intermediate and
    # is supplied by the merged capsule's own internal SSA link, not by a program buffer. Checked
    # against every member's own `dst` (the PROGRAM's name for it), not against the capsule-level seam
    # strings this function assigned above: those live in a different namespace (the capsule's own
    # tensor names) and would never match a program buffer name, which is what silently let an
    # internal member's consumed operand through as if it were external in an earlier version of this
    # module -- caught by a region wrongly asking to bind a tensor no capsule text declares.
    internal_dsts = {str(m["operands"]["dst"]) for m in members[:-1]}
    bound: dict[str, list[str]] = {}
    for member in members:
        operands = member["operands"]
        weight_keys = [
            key
            for key in ("lhs", "rhs")
            if key in operands
            and operands[key] not in internal_dsts
            and str(((shapes or {}).get(operands[key]) or {}).get("role")) == "weight"
        ]
        input_keys = [
            key
            for key in ("lhs", "rhs")
            if key in operands and operands[key] not in internal_dsts and key not in weight_keys
        ]
        for key in weight_keys:
            bound.setdefault("weight", []).append(operands[key])
        for key in input_keys:
            bound.setdefault("input", []).append(operands[key])
        if "bias" in operands and operands["bias"] not in internal_dsts:
            bound.setdefault("bias", []).append(operands["bias"])
    # THE OUTPUTS ARE BOUND BY THE NAMES THE CAPSULE DECLARED (`region_output_name`), never by role
    # and position: the BOUNDARY's is required -- it is what the region commits and is graded on -- and
    # an internal member's is bound to that member's own program buffer only when the reply declares
    # it (the kernel may stage it there; it may equally keep it to itself and never form it at all).
    declared = emitted.get("tensors") if isinstance(emitted.get("tensors"), Mapping) else {}
    if out_names[-1] not in declared:
        return refuse(
            WP.ROLE_UNBOUND,
            f"the region's reply declares no {out_names[-1]!r}, the boundary member {members[-1]['tag']}'s "
            "output the capsule named; a region is accepted only for the boundary it commits",
        )
    pinned = {out_names[-1]: str(members[-1]["operands"]["dst"])}
    for name, member in zip(out_names[:-1], members[:-1], strict=True):
        if name in declared:
            pinned[name] = str(member["operands"]["dst"])
    deduped = {role: tuple(dict.fromkeys(values)) for role, values in bound.items()}

    commands, scratch, why, cause = WP._spliced(emitted, deduped, tag, shapes, views, record, pinned)
    if not commands:
        return commands, scratch, why, cause, None

    # SUFFICIENCY, AT THE BOUNDARY: the boundary member's own program tensor must be some command's
    # OWN committed `operands["dst"]`, by name, after splicing -- never merely declared. It is the one
    # tensor the rest of the program reads from this window, and the one its correctness is judged on.
    boundary_dst = str(members[-1]["operands"]["dst"])
    committed = {
        c["operands"]["dst"]
        for c in commands
        if isinstance(c, Mapping) and isinstance(c.get("operands"), Mapping) and "dst" in c["operands"]
    }
    if boundary_dst not in committed:
        return refuse(
            WP.NO_COMMANDS,
            f"the region's reply never commits {boundary_dst!r}, its boundary's output, by name; a region "
            "is accepted only for the boundary it commits",
        )
    region_info = {
        "id": tag,
        "members": [str(m["tag"]) for m in members],
        "boundary": str(members[-1]["tag"]),
        "declared_opcodes": declared_opcodes,
        # WHERE THIS REGION IS GRADED: its boundary's output, against the oracle. Its internal members
        # are answered by the same kernel and have no output of their own to grade.
        "graded_at": str(members[-1]["tag"]),
        # Which internal members' outputs the reply chose to stage in their own program buffers.
        "staged": [str(m["tag"]) for name, m in zip(out_names[:-1], members[:-1], strict=True) if name in pinned],
    }
    return commands, scratch, why, cause, region_info


# ---- region_legality ---------------------------------------------------------------------------
# Which consecutive device groups of a captured model may be answered as ONE region kernel.
#
# Today the whole-program statement (:mod:`.whole_program`) puts every device group to a package one
# at a time: a package answers a matmul, or a residual add, and never both together, so it cannot fuse
# a residual add onto the contraction that feeds it or keep an intermediate on the accelerator between
# two groups. This module answers the one question that has to be answered BEFORE any of that is
# offered: which runs of groups even COULD be one region, from the statement's own dataflow alone.
#
# **The rule, and why it is the builder's to apply, not the package's.** A run of consecutive groups
# ``g_i, g_{i+1}, ..., g_j`` may be asked as one region exactly when every internal boundary in it is a
# SINGLE-CONSUMER INTERMEDIATE THAT IS NOT A MODEL OUTPUT: group ``g_k``'s committed tensor (``k`` in
# ``(i, j]``... rather ``[i, j)``) is read by no group other than ``g_{k+1}``, and it is not the
# model's own result. Both halves matter. A tensor two groups both read cannot be folded into one
# kernel's private intermediate -- the OTHER reader still needs it as a named buffer. A tensor that IS
# the model's result cannot disappear into a region either: something has to read it out, and the
# statement already names the last device group's product as that result
# (:func:`merlin.compiler.lower.device.whole_program.whole_program_buffer`'s own convention, restated here rather
# than re-derived, so the two cannot disagree about which group produces the output).
#
# The rule is a fact about the model's dataflow, never about a name, a shape, an op spelling, or a
# group count -- a two-layer synthetic model and a seventy-one-group capture are found by the same
# walk, and adding a second target requires no change here at all.
#
# **What legality does NOT decide.** A legal window is an OFFER, not a fusion: :mod:`.region_capsule`
# still has to ask a package for it, the package may decline (or answer group by group, which reads as
# declining), and a region that the package does not answer falls back to asking its members one at a
# time -- exactly as an unaccepted single-group ask always has.


def _sink(group: Any) -> Any | None:
    """The one tensor a device group commits, as the whole-program statement reads it: the last
    member's first result. ``None`` for a member sequence with no result at all (a residency
    backend's own ``EVICT`` tail), which cannot be anyone's producer."""
    if not group.members or not group.members[-1].results:
        return None
    return group.members[-1].results[0]


def legal_regions(groups: Sequence[Any]) -> list[tuple[int, int]]:
    """Maximal runs of two or more consecutive device groups this program may put to a package as
    ONE region, as ``(first group index, last group index)`` pairs in ascending, non-overlapping
    order.

    Only DEVICE groups are considered (a group's placement is checked, never assumed); the one host
    region a closed model keeps is not a candidate for anything here. A model whose groups admit no
    legal boundary at all -- every intermediate is read twice, or every group but the last is a
    fan-out -- returns an empty list, and every group is still asked on its own, exactly as before
    this existed.
    """
    from merlin.compiler.partition.groups import compute_groups as CG

    device = [g for g in groups if g.placement != CG.HOST]
    if len(device) < 2:
        return []
    sinks: dict[int, Any] = {g.index: _sink(g) for g in device}
    consumers: dict[int, set[int]] = {index: set() for index, value in sinks.items() if value is not None}
    for group in device:
        for member in group.members:
            for operand in member.operands:
                for index, value in sinks.items():
                    if index != group.index and value is not None and operand is value:
                        consumers[index].add(group.index)
    order = [g.index for g in device]
    # THE MODEL'S OWN RESULT is the last device group's product -- the same convention
    # `whole_program_buffer` uses to declare the buffer's output, restated rather than re-derived so
    # the two cannot silently disagree about which group may not disappear into a region.
    result_group = order[-1]
    regions: list[tuple[int, int]] = []
    start = order[0]
    for position in range(len(order) - 1):
        this_index, next_index = order[position], order[position + 1]
        legal_edge = this_index in consumers and consumers[this_index] == {next_index} and this_index != result_group
        if not legal_edge:
            if this_index != start:
                regions.append((start, this_index))
            start = next_index
    if order[-1] != start:
        regions.append((start, order[-1]))
    return regions


# ---- whole_program_attribution -----------------------------------------------------------------
# Which groups of a whole-program buffer the SUBMISSION's own compiler answered, and which fell back.
#
# Split out of :mod:`merlin.compiler.lower.device.whole_program` (which re-exports both names) to keep that module
# under the repository's module-size limit.

#: WHAT A SPLIT MEASURED BEFORE 2026-09-24 IS WORTH. The splice bound a package's tensors to this
#: program's without checking their shapes whenever the role counts happened to agree, so a group
#: could be counted as "the submission's" while its kernel read a wrongly-ordered buffer. Measured on
#: a captured ResNet-50 the moment the classification existed: of 123 bindings, 2 were identical and
#: 121 were permutations -- so the 35-of-71 split reported all session described bindings that were
#: not correct, and the honest answer to "how much of this model does the submission author" was
#: UNKNOWN, not 35. Any earlier figure is void rather than superseded.
SPLIT_UNMEASURED_BEFORE = "2026-09-24: splits reported before the binding shape check are void"


def attribution(buffer: Mapping[str, Any] | None) -> dict[str, Any]:
    """Which groups of this program the SUBMISSION's own compiler answered, and which fell back.

    Evidence about the emission as it stands, for a reader who would otherwise have no way to tell.
    The distinction is invisible in the command list -- a reference command and a spliced one sit
    side by side and both ran -- so without this, work spent on a path the compiler is never asked
    about looks exactly like work spent on one it answers. Measured on a live campaign: an authoring
    agent's first edit went into the convolution lowering, every convolution group falls back, and
    the emission was correctly unchanged.

    States the attribution and nothing about what to do with it: fallback reasons are the package's
    OWN words, quoted, and the roll-up is a count rather than a recommendation.
    """
    # At call time: that module re-exports this function, so it imports this one.
    from merlin.compiler.lower.device.whole_program import FROM_REFERENCE, FROM_SUBMISSION

    route = (buffer or {}).get("whole_program") if isinstance(buffer, Mapping) else None
    if not isinstance(route, Mapping) or not isinstance(route.get("per_group"), list):
        return {
            "schema": "whole_program_attribution_v1",
            "status": "unavailable",
            "why": "this buffer carries no whole-program route record, so nothing says which compiler answered what",
        }
    rows = [row for row in route["per_group"] if isinstance(row, Mapping)]
    # A ROW THAT NAMES NO OPERATION IS NOT A ROW ABOUT "UNKNOWN". Measured on a live campaign: every
    # row bucketed to UNKNOWN and the per-operation split never appeared, because the buffer had been
    # RETAINED from an emission by an older harness that did not record the operation. Bucketing that
    # under a name reads as a measured fact about seventy-one groups; saying the field is absent says
    # what is true, and points at the retained buffer instead of at the model.
    by_op: dict[str, dict[str, int]] | None = {}
    if not any(row.get("op") for row in rows):
        by_op = None
    else:
        for row in rows:
            op = str(row.get("op") or "UNKNOWN")
            side = str(row.get("on") or "UNKNOWN")
            by_op.setdefault(op, {FROM_SUBMISSION: 0, FROM_REFERENCE: 0})
            by_op[op][side] = by_op[op].get(side, 0) + 1
    answered = sum(1 for row in rows if row.get("on") == FROM_SUBMISSION)
    # FALLBACKS GROUPED BY THE PACKAGE'S OWN REASON. The per-op roll-up says WHICH operations fall
    # back; this says WHY, and the two answer different questions. Measured on a live campaign, the
    # 36 fallbacks are three causes of very different kinds -- an operation the package has no
    # lowering for at all, a layout it states differently from this program, and a capacity argument
    # that is correct and should stay a refusal -- and a reader who saw only "36" would not be able
    # to tell the one worth closing from the one that must not be.
    reasons: dict[str, dict[str, Any]] = {}
    for row in rows:
        why = str(row.get("why") or "")
        if row.get("on") == FROM_SUBMISSION or not why:
            continue
        # KEYED ON THE CAUSE, NOT THE SENTENCE. The sentence carries the package's own shapes and
        # tensor names, so nineteen refusals of one cause read as nineteen causes of one -- measured,
        # and it buried the largest single cause beneath a smaller one whose text happened to repeat.
        key = str(row.get("cause") or why)
        seen = reasons.setdefault(
            key, {"cause": row.get("cause"), "groups": 0, "ops": {}, "examples": [], "why": why, "_sentences": {}}
        )
        seen["groups"] += 1
        seen["_sentences"][why] = seen["_sentences"].get(why, 0) + 1
        if row.get("op"):
            seen["ops"][str(row["op"])] = seen["ops"].get(str(row["op"]), 0) + 1
        if len(seen["examples"]) < 3:
            seen["examples"].append(row.get("group"))
    # THE QUOTED EXAMPLE IS THE ONE THE BUCKET IS MOSTLY MADE OF. A cause collects refusals whose
    # sentences differ, and quoting whichever arrived first states the wrong reason over the right
    # count: measured on a live campaign, a 17-group `package_declined` bucket quoted a stem
    # convolution's capacity argument while 16 of its 17 rows were an operation the package had no
    # lowering for at all -- so the one refusal that should STAY was presented as the whole bucket.
    # The full distribution travels beside it, because a bucket of several sentences is a different
    # fact from a bucket of one and a single quote cannot distinguish them.
    for detail in reasons.values():
        sentences = detail.pop("_sentences")
        detail["why"] = max(sentences.items(), key=lambda item: item[1])[0]
        detail["distinct_sentences"] = [
            {"count": count, "why": sentence}
            for sentence, count in sorted(sentences.items(), key=lambda item: -item[1])
        ]
    return {
        "schema": "whole_program_attribution_v1",
        "status": "derived",
        "groups": len(rows),
        "on_submission": answered,
        "on_reference": len(rows) - answered,
        "by_op": by_op,
        # `why` is ONE verbatim sentence -- the one MOST of the bucket's rows gave, never a paraphrase
        # and never a merge; `distinct_sentences` carries the rest with their counts.
        "fallback_by_reason": [
            dict(detail) for _key, detail in sorted(reasons.items(), key=lambda item: -item[1]["groups"])
        ],
        "grouping": (
            "by cause"
            if any(row.get("cause") for row in rows)
            else "by message text; these rows predate the cause field, so refusals differing only in "
            "their quoted shapes appear as separate causes"
        ),
        # The package's own words, unparaphrased, and absent rather than empty when the group was
        # answered -- "no reason given" and "a reason nobody recorded" are different facts.
        "per_group": [
            {
                "group": row.get("group"),
                "op": row.get("op"),
                "answered_by": row.get("on"),
                **({"why": row["why"]} if row.get("why") else {}),
            }
            for row in rows
        ],
        "means": (
            "a group answered by the submission was lowered by the compiler under test; one answered "
            "by the reference was stated by the harness because the submission did not lower it, for "
            "the quoted reason. Both run; the emission does not distinguish them."
        ),
    }


# ---- whole_program_splice ----------------------------------------------------------------------
# How ONE group's package-emitted command buffer is bound into a whole program's namespace.
#
# Split out of :mod:`merlin.compiler.lower.device.whole_program` (which re-exports every cause token and helper
# here) so that module stays under the repository's module-size gate. The seam is the one the
# statement already names: STATING a model as one buffer is one concern; binding a package's reply to
# the buffers that statement declared -- by role and shape, refusing a permutation by name -- is
# another. Nothing here names a target.

#: WHY a group fell back, as a token rather than a sentence. The sentence quotes the package's own
#: words and carries its shapes and tensor names, so two groups refused for the SAME cause read as two
#: different causes -- measured on a live campaign, nineteen layout refusals became nineteen buckets of
#: one while a sixteen-group bucket sat above them, and the largest single cause was invisible AS a
#: cause. The token is what a reader groups by; the sentence is what they then read.
NO_COMMANDS = "package_emitted_no_command"
MALFORMED = "package_buffer_malformed"
ROLE_UNBOUND = "operand_not_bound_by_shape"
ROLE_AMBIGUOUS = "operand_shape_ambiguous"
NO_SHAPE = "operand_shape_unknown"
PACKAGE_DECLINED = "package_declined"
PACKAGE_FAILED = "package_invocation_failed"
CAPSULE_FAILED = "interface_capsule_failed"
DTYPE_MISMATCH = "operand_dtype_mismatch"
NOT_ASKED = "no_package_named"
#: The CALLER routed this group to the reference/library statement (``decline=``), not the package.
#: Same string :data:`merlin.analysis.whole_model.build.stages.CALLER_DECLINED` uses -- two modules, one cause,
#: never cross-imported (``llvmlower`` sits below ``perf``).
CALLER_DECLINED = "caller_declined"


#: Roles whose tensor may be READ under another shape when it is provably the same bytes. Only the
#: data ones: a weight's device layout is a PERMUTATION that `group_prepack.device_weight` performs,
#: and admitting a reshape there would bind a correctly-sized buffer holding the wrong element order.
_RESHAPABLE_ROLES = ("input", "output")


def _elements(shape: Sequence[int]) -> int:
    """How many elements a shape holds; 0 for an unreadable one, which never matches."""
    if not shape:
        return 0
    total = 1
    for extent in shape:
        total *= int(extent)
    return total


def _same_bytes(mine: list[int], theirs: list[int]) -> str:
    """Why two shapes hold the SAME BYTES IN THE SAME ORDER, or "" when they do not.

    Derived, never assumed from a pair that happens to look right. The rule is the one both sides
    actually obey: these tensors are channel-minor, so they are the same bytes exactly when their
    trailing extent agrees and the product of everything before it agrees. A matmul writing
    ``[3136, 64]`` and a convolution reading ``[1, 56, 56, 64]`` pass -- 3136 = 56x56, 64 = 64, and
    the element order is identical -- while a weight ``[64, 64, 3, 3]`` against ``[576, 64]`` fails,
    because that is a PERMUTATION and the prepack, not a reshape, is what performs it.

    A rule rather than a convention: it either holds of the two tensors or it refuses. A convention
    about which groups may sit together holds only while everyone remembers to apply it.
    """
    if not mine or not theirs:
        return ""
    if mine == theirs:
        return "identical"
    # A UNIT AXIS HOLDS ONE INDEX, so adding or dropping one cannot move an element: [2048, 1] and
    # [1, 2048] are one vector. Measured on a captured ResNet-50: the classifier reads the pooled
    # [2048, 1] as its [1, 2048] row, and the channel-minor rule below refused it as a permutation.
    squeezed_mine = [int(e) for e in mine if int(e) != 1]
    squeezed_theirs = [int(e) for e in theirs if int(e) != 1]
    if squeezed_mine == squeezed_theirs:
        return f"the same {squeezed_mine or [1]} up to unit axes, which hold one index each"
    if mine[-1] != theirs[-1]:
        return ""
    lead_mine, lead_theirs = 1, 1
    for extent in mine[:-1]:
        lead_mine *= int(extent)
    for extent in theirs[:-1]:
        lead_theirs *= int(extent)
    if lead_mine != lead_theirs:
        return ""
    return f"same elements in the same order: {lead_mine} positions x {mine[-1]} channel-minor"


def _refuse(cause: str, why: str):
    """One group's fallback: the token to group by, and the package's own words to read."""
    return [], {}, why, cause


def _decline_key(item: Any) -> tuple[str, Any]:
    """``("group", index)`` for a group named by index (``33`` or ``"g33"``), else ``("op", name)``.

    The SAME rule :func:`merlin.analysis.whole_model.build.stages._decline_key` applies to a caller's ``decline``
    list -- duplicated rather than imported, because ``llvmlower`` sits below ``perf`` and importing
    upward would invert that layering. A change to one must be checked against the other.
    """
    if isinstance(item, bool):
        raise ValueError(f"decline entry {item!r} is neither an op name nor a group index")
    if isinstance(item, int):
        return ("group", item)
    text = str(item)
    if text[:1] == "g" and text[1:].isdigit():
        return ("group", int(text[1:]))
    if text.isdigit():
        return ("group", int(text))
    return ("op", text)


def _is_declined(group_index: int, op: str, decline: Sequence[Any]) -> bool:
    """Whether the CALLER named this group -- by index or by every group of this op -- in ``decline``."""
    for item in decline or ():
        kind, value = _decline_key(item)
        if kind == "group" and int(value) == int(group_index):
            return True
        if kind == "op" and str(value) == str(op):
            return True
    return False


def _record_binding(name, target_name, role, declared_shape, shapes, views, tag, rename) -> tuple | None:
    """Bind one of the package's tensors to this program's, checking the shapes. ``None`` when bound.

    Three outcomes and they are three different claims. IDENTICAL shapes bind silently. A shape that
    holds the SAME BYTES IN THE SAME ORDER binds and is recorded as a view, with the derivation that
    admitted it. Anything else is a PERMUTATION: it binds only provisionally, marked, and the caller
    refuses it outright -- because equal element count proves nothing about order, and a
    correctly-sized buffer in the wrong one links, runs and passes. Performing that permutation is
    separate work (`group_prepack.device_weight` is what does it) and this buffer does not state it.
    """
    rename[name] = target_name
    read_as = list(declared_shape.get(name) or [])
    want = list(((shapes or {}).get(target_name) or {}).get("shape") or [])
    if not want or read_as == want:
        return None
    if _elements(read_as) != _elements(want):
        return _refuse(
            ROLE_UNBOUND,
            f"the package declares {role!r} tensor {name!r} as {read_as} where this program binds "
            f"{target_name!r} as {want}; those are different numbers of elements",
        )
    order = _same_bytes(want, read_as)
    if not order:
        # EQUAL ELEMENT COUNT PROVES NOTHING ABOUT ORDER. This is a permutation -- measured on a
        # captured ResNet-50, every one of its nineteen convolutions wants NHWC where the program
        # holds NCHW -- and binding it would hand the kernel a correctly-sized buffer in the wrong
        # element order, which links, runs and passes. Refused by name; performing that transpose is
        # separate work that something must do and this buffer must state.
        return _refuse(
            ROLE_UNBOUND,
            f"the package declares {role!r} tensor {name!r} as {read_as} where this program binds "
            f"{target_name!r} as {want}; the element counts agree but the order does not, so this is "
            f"a permutation rather than a reshape",
        )
    if views is not None:
        # STATED, never implied: the buffer says this tensor is read under another shape and on what
        # basis, so a reader is never left to infer that the bytes line up.
        views.append(
            {
                "tensor": target_name,
                "read_as": read_as,
                "declared": want,
                "by": tag,
                "basis": order,
            }
        )
    return None


def _spliced(
    buffer: dict,
    bound: dict,
    tag: str,
    shapes: Mapping[str, Mapping] | None = None,
    views: list | None = None,
    record: dict | None = None,
    pinned: Mapping[str, str] | None = None,
) -> tuple[list[dict], dict[str, dict], str, str]:
    """One group's package-emitted buffer, renamed into the whole program's namespace.

    ``pinned`` binds tensors BY THE NAME an offer declared (``{package name: program tensor}``) ahead
    of any role matching. A negotiated group carries two data operands whose roles are the same and
    whose shapes can coincide -- the activation and the skip tensor -- so pairing them by role and
    order would be a guess; the offer named them, and the reply is bound by those names or refused.

    ``bound`` maps the group's ROLES to the names this program already uses for them, so the
    submission's ``A0``/``W``/``B``/``Y0`` become the buffers its neighbours read and write. Every
    other name the package used is a scratch handle private to that group and is prefixed, because two
    groups both calling their accumulator ``acc0`` would otherwise alias into one buffer.

    ``record``, when given, receives the BINDING this made -- each of the package's tensors, the role
    it declared, the shape it declared and the program tensor it now names. It is the one statement
    of which buffer a kernel argument is, and a caller that calls the package's kernel reads it here
    rather than re-deriving it from roles and extents a second time.
    """
    declared = buffer.get("tensors")
    commands = buffer.get("commands")
    if not isinstance(declared, Mapping) or not isinstance(commands, Sequence) or not commands:
        return _refuse(NO_COMMANDS, "the package emitted no command for this group")
    rename: dict[str, str] = {}
    by_role: dict[str, list[str]] = {}
    for name, spec in declared.items():
        if not isinstance(spec, Mapping):
            return _refuse(MALFORMED, f"the package declared tensor {name!r} as something other than an object")
        by_role.setdefault(str(spec.get("role") or ""), []).append(str(name))
    declared_shape = {str(name): list((spec or {}).get("shape") or []) for name, spec in declared.items()}
    for name, target_name in (pinned or {}).items():
        spec = declared.get(name)
        if not isinstance(spec, Mapping):
            return _refuse(
                ROLE_UNBOUND,
                f"the offer declared tensor {name!r} for {target_name!r} and the package's buffer does not "
                f"declare it; a negotiated group binds by the names it was offered in",
            )
        refusal = _record_binding(
            str(name), target_name, str(spec.get("role") or ""), declared_shape, shapes, views, tag, rename
        )
        if refusal:
            return refusal
    for role, wanted in bound.items():
        found = sorted(by_role.get(role, ()))
        if len(found) == len(wanted):
            # PAIRED BY ORDER, BUT STILL CHECKED. The declaration order is meaningful when the counts
            # agree -- an operand sum's two activations are stated in the order their scales are --
            # so the pairing is positional here. The SHAPES are not thereby agreed: this path used to
            # bind without looking at them, so a weight the package declares transposed bound
            # silently, with no view recorded and nothing to notice it. That is the wrong element
            # order in a correctly-sized buffer, which links, runs and passes.
            for name, target_name in zip(found, wanted, strict=True):
                refusal = _record_binding(name, target_name, role, declared_shape, shapes, views, tag, rename)
                if refusal:
                    return refusal
            continue
        # MORE TENSORS OF THIS ROLE THAN THE GROUP BINDS. A package legitimately declares its own
        # staging beside the operand -- a convolution's im2col buffer is an `input` in its capsule and
        # scratch to this program -- so the extra ones are not an error. What binds them is the SHAPE,
        # which both sides state; position would be a guess, and a mis-bound operand is a numerically
        # wrong program that still passes.
        pool = list(found)
        for target_name in wanted:
            want = list(((shapes or {}).get(target_name) or {}).get("shape") or [])
            if not want:
                return _refuse(NO_SHAPE, f"this program declares no shape for {target_name!r}, so nothing can bind it")
            # A DATA tensor may be READ under a different shape when it is provably the same bytes --
            # a convolution reads its producer's [positions, channels] as [1, H, W, channels]. A
            # WEIGHT may not: its device layout is a permutation, which a reshape cannot perform. So
            # the relaxation is per role, and it is a derived check either way.
            reshapable = role in _RESHAPABLE_ROLES
            match = [
                name
                for name in pool
                if declared_shape.get(name) == want
                or (reshapable and _same_bytes(want, declared_shape.get(name) or []))
            ]
            if len(match) != 1:
                return _refuse(
                    ROLE_AMBIGUOUS if len(match) > 1 else ROLE_UNBOUND,
                    (
                        f"the package declares {len(match)} {role!r} tensor(s) of shape {want} for "
                        f"{target_name!r} (it declares "
                        + ", ".join(f"{name}{declared_shape.get(name)}" for name in found)
                        + "); binding by position would be a guess"
                    ),
                )
            pool.remove(match[0])
            refusal = _record_binding(match[0], target_name, role, declared_shape, shapes, views, tag, rename)
            if refusal:
                return refusal
    for name in declared:
        rename.setdefault(str(name), f"{tag}_{name}")
    # THE ELEMENT TYPE IS PART OF WHICH BUFFER A DATA TENSOR IS. The shape checks above count elements,
    # and a count is blind to width: a kernel committing i8 into a buffer this program declares i32 (a
    # classifier that leaves as the accumulator) writes a quarter of the bytes and every reader of it
    # reads garbage -- with every shape agreeing. Checked for the data roles, whose dtype both sides
    # state; a weight or bias is laid out by the prepack and is not the capture's declaration.
    held_names = {str(target) for targets in bound.values() for target in targets} | {
        str(t) for t in (pinned or {}).values()
    }
    for name, spec in declared.items():
        target_name = rename[str(name)]
        role = str((spec or {}).get("role") or "")
        if role not in _RESHAPABLE_ROLES or target_name not in held_names:
            continue
        theirs = str((spec or {}).get("dtype") or "")
        mine = str(((shapes or {}).get(target_name) or {}).get("dtype") or "")
        if theirs and mine and theirs != mine:
            return _refuse(
                DTYPE_MISMATCH,
                f"the package declares {role!r} tensor {name!r} as {theirs} where this program holds "
                f"{target_name!r} as {mine}; the element counts agree and the bytes do not",
            )

    def renamed(value: str) -> str:
        return rename.get(str(value), f"{tag}_{value}")

    out: list[dict] = []
    for command in commands:
        if not isinstance(command, Mapping) or not command.get("opcode"):
            return _refuse(MALFORMED, "the package emitted a command with no opcode")
        row: dict[str, Any] = {"opcode": str(command["opcode"])}
        operands = command.get("operands")
        if isinstance(operands, Mapping):
            row["operands"] = {str(key): renamed(value) for key, value in operands.items()}
        attributes = command.get("attributes")
        if isinstance(attributes, Mapping):
            # A tensor NAME can appear in an attribute too (the commit's `bias`), and leaving that one
            # unrenamed points a command at a buffer from another group with nothing to notice it.
            row["attributes"] = {
                key: (renamed(value) if isinstance(value, str) and value in rename else value)
                for key, value in attributes.items()
            }
        out.append(row)
    scratch = {
        rename[name]: dict(spec)
        for name, spec in declared.items()
        if rename[name] not in bound.get("__bound__", ()) and rename[name].startswith(f"{tag}_")
    }
    if record is not None:
        held = {str(target) for targets in bound.values() for target in targets} | {
            str(t) for t in (pinned or {}).values()
        }
        record["binding"] = {
            str(name): {
                "program": rename[str(name)],
                "role": str((spec or {}).get("role") or ""),
                "declared": list(declared_shape.get(str(name)) or []),
                # A tensor the program holds (an operand, a weight, a neighbour's output) as against
                # one private to this group's kernel, which nothing outside the group reads.
                "bound": rename[str(name)] in held,
            }
            for name, spec in declared.items()
        }
    return out, scratch, "", ""


# ---- device_routing_placement ------------------------------------------------------------------
# Derive a whole-model build's ``DeviceRouting`` from a placement, rather than declaring it.
#
# Split out of :mod:`merlin.compiler.lower.device.build` (which re-exports both names) to keep that module
# under the repository's module-size limit.

if TYPE_CHECKING:
    from merlin.compiler.lower.device.device_build import DeviceRouting
    from merlin.target.contract.build.object_admission import LinkedElfAdmissionService


def routing_for_placement(
    placement,
    device: str,
    package_dir: str | Path,
    *,
    numeric_policy=None,
    granularity: str = "contraction",
    capture: str | Path | None = None,
    model: str = "",
    linked_elf_admission: LinkedElfAdmissionService | None = None,
) -> DeviceRouting:
    """The ``DeviceRouting`` a whole-model build needs, derived from a placement rather than declared.

    This is the step that made the fused single-ELF path unreachable in production. Every piece of it
    exists -- the rewrite, the shim, the per-extent kernel build, the link -- and all of it is inert
    until a caller supplies ``select``, which nothing did. So a "compiled" whole model ran its
    contractions through the Python interpreter while the artifact that would have run them on the
    device was never asked for.

    The operand and accumulate formats are read off the placement, NOT defaulted: they are what the
    router matched the contraction against, and a build that assumed them would emit kernels in a
    precision the placement never chose. Two consequences, both deliberate:

    * a placement that put nothing on this device raises, because a routing with no work is a caller
      error rather than an empty build;
    * device placements that disagree about the datapath raise too. One ELF carries one device
      datapath; picking the first and dropping the rest is how half a model gets computed in a
      precision nobody selected.

    The accumulate format has a SECOND derivation because a unit legitimately has no accumulate rule to
    match: a contract that declares its formats without an accumulate matrix routes fine and reports
    ``acc=None`` (measured: the reference systolic mesh does exactly this). That is a gap in the
    contract, not in the hardware, so the format is then read off the device's own RTL datapath facts
    for this operand pair -- the same source :mod:`merlin.target.system` uses -- and it is still an
    error when those facts name none, or name more than one.
    """
    # At call time: that module re-exports this function, so it imports this one.
    from merlin.compiler.lower.device.device_build import DeviceRouting
    from merlin.compiler.planning.placement import device_selector

    on_dev = [p for p in placement.placed if p.on_device and p.device == device]
    if not on_dev:
        raise ValueError(f"this placement puts no work on {device!r}; there is nothing to build")
    operands = {getattr(p.demand, "in_fmt", None) for p in on_dev}
    weights = {getattr(p.demand, "weight_fmt", None) or getattr(p.demand, "in_fmt", None) for p in on_dev}
    accums = {p.acc for p in on_dev}
    if len(operands) != 1 or len(weights) != 1 or len(accums) != 1:
        raise ValueError(
            f"{device!r} placements disagree about the datapath (operands={sorted(map(str, operands))}, "
            f"accumulate={sorted(map(str, accums))}); one image carries one device datapath, so the "
            f"placement has to be split before it can be built"
        )
    operand, weight, accum = operands.pop(), weights.pop(), accums.pop()
    if not operand:
        raise ValueError(
            f"{device!r} placements carry no operand format; the kernel precision is "
            f"underivable and assuming one emits the wrong datapath"
        )
    accum = accum or _accum_from_facts(device, operand, weight)
    return DeviceRouting(
        device=device,
        package_dir=package_dir,
        operand_dtype=str(operand),
        accum_dtype=str(accum),
        select=device_selector(placement),
        numeric_policy=numeric_policy,
        granularity=str(granularity),
        capture=capture,
        model=str(model),
        linked_elf_admission=linked_elf_admission,
    )


def _accum_from_facts(device: str, operand: str, weight: str) -> str:
    """The accumulate format this device's RTL datapath declares for ``operand`` x ``weight``.

    Fails closed in both directions. No matching triple means the device does not declare what it
    accumulates this pair into, and more than one means it declares several -- and a build that picked
    among them would be choosing the model's arithmetic on the strength of dictionary order.
    """
    from merlin.target.capabilities.routing import _fmt_ok  # noqa: PLC2701 -- one format-equality predicate
    from merlin.target.system import device_dtype_triples

    found = {a for i, w, a in device_dtype_triples(device) if _fmt_ok(operand, (i,)) and _fmt_ok(weight, (w,))}
    if len(found) != 1:
        raise ValueError(
            f"{device!r} declares {len(found)} accumulate format(s) for {operand} x {weight} "
            f"({sorted(found) or 'none'}); the unit matched no accumulate rule either, so the kernel "
            f"precision is underivable and assuming one emits the wrong datapath"
        )
    return found.pop()


__all__ = [
    "ask_package_region",
    "legal_regions",
    "region_output_name",
    "split_region_commands",
]
