"""How ONE group's package-emitted command buffer is bound into a whole program's namespace.

Split out of :mod:`merlin.compiler.lower.model.whole_program` (which re-exports every cause token and helper
here) so that module stays under the repository's module-size gate. The seam is the one the
statement already names: STATING a model as one buffer is one concern; binding a package's reply to
the buffers that statement declared -- by role and shape, refusing a permutation by name -- is
another. Nothing here names a target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
#: Same string :data:`merlin.analysis.perf.whole_model_build.CALLER_DECLINED` uses -- two modules, one cause,
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

    The SAME rule :func:`merlin.analysis.perf.whole_model_build._decline_key` applies to a caller's ``decline``
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
