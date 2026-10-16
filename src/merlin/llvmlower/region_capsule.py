"""Put a WHOLE WINDOW of dataflow-connected groups to a package as ONE region capsule.

:mod:`.region_legality` says which runs of consecutive groups a package MAY be offered together; this
module is what makes the offer and reads the answer. It reuses exactly the pieces
:func:`merlin.llvmlower.whole_program._ask_package` already uses for one group -- the same
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
(:func:`merlin.perf.whole_model_build.bind_groups` links it as one call; the target driver restates
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
from typing import Any

__all__ = ["ask_package_region", "region_output_name", "split_region_commands"]


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
    (:class:`merlin.perf.whole_model_build.LocalReference`) already holds every group to. ``None`` when
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
    from merlin.llvmlower import whole_program as WP

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
    :func:`merlin.xdsl_dialects.lowering.group_command.program` states it), ``operands`` (THIS
    PROGRAM's own buffer names for the group's operands -- never the capsule's local ones), and
    ``opcode`` (the command-buffer opcode the entry states).

    Returns exactly what :func:`merlin.llvmlower.whole_program._ask_package` returns for one group --
    ``(commands, scratch, why, cause)`` -- with a fifth element: ``None`` when the region was not
    asked or not accepted (the caller then asks each member on its own, exactly as an unspliced
    single group always has), else the region's own record (its id, its members in order, and which
    one is the BOUNDARY -- the only member whose output is still checked as this program's own named
    tensor).
    """
    from merlin.llvmlower import whole_program as WP
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen import package_runtime as OR

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
    from merlin.verify.lean_emission_context import write_emitted_artifact

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
