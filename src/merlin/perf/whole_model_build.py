"""A compiler package and a model capsule, built into ONE runnable whole-model program and its oracle.

    record = build(package_dir, model_capsule, target=target)

is the whole route, with no hand step in it. Until this existed every whole-model number that ran a
package's own kernels came from scratch scripts driven by hand: the package asked once per group, its
LLVM re-lowered through a second entrypoint call with ad-hoc ``llc`` flags, each kernel's arguments
re-derived from roles and extents the driver recomputed, and the oracle run separately. Each of those
was a second spelling of a fact the compiler route already states, free to disagree with it.

ONE PATH, THE CAPSULE'S. A capsule is a one-group model: every compute group of the model is put to the
package as a ``merlin_iface`` capsule through the SAME call the capsule runner makes
(:func:`merlin.targetgen.capsule_common.lower_interface`, reached through
:func:`merlin.llvmlower.whole_program.whole_program_buffer`), and each reply's target artifact is
compiled to an object by the SAME function the capsule runner compiles one with
(:func:`merlin.targetgen.contract.compile.llvm_mlir_to_object`). Nothing about a group is restated:

* the output extents are ``group_command.device_output_shape`` (through the whole-program statement);
* the kernel's argument ORDER is the contract's ``arg_order_by_command_shape`` row that the package's
  own buffer matches (the selected support's ``rtl_checks.resolve_kernel_arg_order``);
* which program buffer each argument IS comes from the binding the splice made and recorded, checked
  there against the declared shapes (a permutation is refused, never bound);
* a weight's device layout is ``group_prepack.device_weight``'s, carried with its content digest, and
  the driver's copy of it is compared byte for byte;
* a buffer private to a kernel (an im2col matrix) is built from the package's own declared recipe.

A group the package does not answer keeps the TARGET's library call, with the package's own reason.
The C program around the calls -- vendor library, timing brackets, the UART protocol -- is the target's
software environment and is loaded from the target's backend (``whole_model_driver``); this module
writes none of it.

THE ORACLE IS INDEPENDENT OF EVERY ARM. The model is stated a second time with NO package, and
:func:`merlin.runtime.reference.reference_outputs` recomputes every committed buffer from the leaf
inputs, supplied by name and refused if any is missing (a missing leaf is otherwise filled with
stimulus, which computes a different model and still runs). Each group's expectation is the FNV-1a
digest the program prints; a group whose op declares a tolerance (``bound_lsb``) is graded on the
program's own bounded check instead, and the digests DOWNSTREAM of such a group are exact only for an
arm that rounds as the reference does -- recorded, because a reader would otherwise read a mismatch
there as a wrong kernel.

Nothing here names a target: the target is a parameter, and everything target-specific -- the harness
build recipe, the entry symbol, the driver -- is read through it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from merlin.common import compile_trace as _trace

# The memory map and the host-side grade live in their own module; re-exported here for callers.
from merlin.perf.whole_model_memory import MEMORY_MAP_SCHEMA, grade_memory, memory_map

from . import whole_model_object_cache as _OC
from . import whole_model_partial as _partial

__all__ = [
    "MEMORY_MAP_SCHEMA",
    "refuse_if_every_group_declined",
    "require_datapath_facts",
    "SCHEMA",
    "ModelCapsule",
    "WholeModelBuildError",
    "bind_groups",
    "build",
    "decline_ops",
    "grade",
    "grade_memory",
    "LocalReference",
    "load_model_capsule",
    "machine_header",
    "memory_map",
    "prune_object_cache",
    "state",
]

SCHEMA = "whole_model_build_v1"

#: Who answered a group. Three different claims, never summed: the compiler under test lowered it, the
#: target's own library stands in for it, or it is the one host region a closed model keeps.
ON_PACKAGE = "package"
ON_VENDOR = "vendor"
ON_HOST = "host"

#: Why a group the package DID answer still cannot be linked, as tokens a reader groups by. The splice's
#: own causes (`whole_program`'s) cover a package that did not answer at all.
ABI_UNDERIVABLE = "kernel_abi_order_underivable"
ARGUMENT_UNBOUND = "kernel_argument_unbound"
PRIVATE_WITHOUT_RECIPE = "private_operand_without_recipe"
OBJECT_FAILED = "kernel_object_build_failed"
#: The splice reported ``FROM_SUBMISSION`` but recorded no path to read the package's own reply from
#: -- a defect in whatever asked the package (a missing ``record["command_buffer"]``/``["artifact"]``),
#: never a property of the submission's bytes. Refused BY NAME rather than a bare ``KeyError``: the
#: latter reads as this build crashing, when the fact is "something upstream claimed a splice and
#: left nothing to read it back from".
SUBMISSION_UNREADABLE = "submission_reply_unreadable"
#: The CALLER sent this op to the target's library (``build(decline=...)``). Not the package's refusal
#: and not a failure of this build: a statement by whoever asked for the program, recorded as theirs.
CALLER_DECLINED = "caller_declined"


from .whole_model_capsule import (
    ModelCapsule as ModelCapsule,
)
from .whole_model_capsule import (
    WholeModelBuildError as WholeModelBuildError,
)
from .whole_model_capsule import (
    _model_golden as _model_golden,
)
from .whole_model_capsule import (
    load_model_capsule as load_model_capsule,
)

# --------------------------------------------------------------------------------- package responses


def package_digest(package_dir: str | Path) -> str:
    """A compiler package's content identity -- the ONE digest a build keys and records it by.

    :func:`merlin.common.tree_hash.hash_tree` over the package directory, the same digest a measurement
    store keys a package on and a harness-owned package repository's commit tree recomputes, so a
    build record, a reply cache and a stored candidate all name one package the same way.
    """
    from merlin.common.tree_hash import hash_tree

    identity = hash_tree(Path(package_dir))
    if not identity["present"]:
        raise WholeModelBuildError(f"the package directory {package_dir} does not exist")
    return str(identity["sha256"])


_package_digest = package_digest


from .whole_model_replies import (  # noqa: E402  (re-exported)
    REPLIES_DEFAULT_MAX_BYTES as REPLIES_DEFAULT_MAX_BYTES,
)
from .whole_model_replies import (  # noqa: E402
    REPLIES_MAX_BYTES_ENV as REPLIES_MAX_BYTES_ENV,
)
from .whole_model_replies import (  # noqa: E402
    Asker,
    _ReplyCache,
    prune_replies,
)
from .whole_model_replies import (  # noqa: E402
    drop_package_replies as drop_package_replies,
)
from .whole_model_replies import (  # noqa: E402
    replies_folder as replies_folder,
)

# --------------------------------------------------------------------------------------- statement


def state(
    capsule: ModelCapsule,
    *,
    target: str,
    package_dir: str | Path | None = None,
    work: str | Path | None = None,
    timeout: int = 600,
    cache: bool = True,
    jobs: int | None = None,
    allow_regions: bool = False,
    decline: Sequence[Any] = (),
    open_model: bool = False,
    binder=None,
    prewarm_objects: bool = False,
    region_internal_ops: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The model as ONE command buffer, each group put to the package when one is named.

    ``prewarm_objects`` compiles each answered group's artifact into the persistent object cache in
    the worker that asked for it, so a build's object stage finds every object already compiled.

    ``binder`` (:func:`corpus_binder`) is the corpus binding every group is stated under when it is
    put to the package; see :func:`merlin.llvmlower.whole_program.group_interface`.

    This is :func:`whole_program_buffer` over the capsule's own files, nothing more; with no package it
    is the reference statement the oracle is computed from. ``cache`` records the package's replies by
    content (see :class:`_ReplyCache`) under ``cache_dir("package-replies")``. ``allow_regions`` is
    OFF BY DEFAULT and passed straight through to :func:`whole_program_buffer`; a caller that never
    sets it gets exactly today's per-group statement. ``region_internal_ops`` is the TARGET's own
    statement of which ops its driver can link as a fused region's internal member
    (:func:`region_facts`); ``None`` reads it from the target's driver.

    ``decline`` (op names / group indices) is also passed straight through: a declined group is never
    asked, so it is stated exactly as a group the package itself refused would be, with a real
    committed buffer, and it is never pulled into a region window either.
    ``open_model`` states only the DEVICE part of a model whose host regions compute between its
    groups (see :func:`merlin.llvmlower.whole_program.whole_program_buffer`).
    """
    from merlin.common import mlir_query as mq
    from merlin.common.ir_lock import IR_LOCK
    from merlin.llvmlower import whole_program as WP
    from merlin.xdsl_dialects.lowering import stream_plan as SP

    manifest = json.loads(capsule.weights_manifest.read_text(encoding="utf-8"))
    if region_internal_ops is None:
        region_internal_ops = region_facts(target)["internal_ops"] if allow_regions else ()
    replies = None
    package = None
    work = Path(work) if work is not None else None
    # One package, one build: an interface the package answered is answered the same way for every
    # group that states it, in either pass (see capsule_common.lower_interface's ``memo``).
    lowering_memo: dict = {}

    def statement(run) -> dict[str, Any]:
        with IR_LOCK:
            # By PATH: the statement only reads the module (a test holds its printed form unchanged), so
            # every pass, and the build's own read of the capture's groups, share one parse of it.
            return WP.whole_program_buffer(
                mq.parse(Path(capsule.interface)),
                target,
                weight_args=SP.weight_args_beside(capsule.interface),
                manifest=manifest,
                model=capsule.name,
                package=package,
                capture=capsule.interface,
                workdir=work,
                timeout=timeout,
                run=run,
                allow_regions=allow_regions,
                decline=decline,
                open_model=open_model,
                binder=binder,
                lowering_memo=lowering_memo,
                region_internal_ops=tuple(region_internal_ops or ()),
            )

    buffer = None
    asker = None
    if package_dir is not None:
        from merlin.common.artifacts import cache_dir
        from merlin.targetgen import oot_runner as OR

        package = OR.load_package(str(package_dir))
        replies = _ReplyCache(_package_digest(package_dir), cache_dir("package-replies") if cache else None)
        if work is not None and replies.directory is not None and jobs != 1:
            # FIRST PASS FROM RECORDED REPLIES ONLY. Every interface is written and every recorded reply
            # replayed; a question with no recorded answer starts being asked in the background the
            # moment its interface is written (and its object compiled once answered), and is refused
            # for now. When none was missing, that pass IS the statement. Otherwise the statement is made
            # again, in order, each group waiting only for its own ask.
            compile_artifact = None
            if prewarm_objects:

                def compile_artifact(text: str) -> None:
                    # Started and left running: the statement does not wait for it, the object stage does.
                    _OC.prewarm_async(text, target=target, jobs=jobs or min(16, os.cpu_count() or 1))

            asker = Asker(
                package,
                replies,
                Path(work),
                timeout=timeout,
                jobs=jobs or min(16, os.cpu_count() or 1),
                compile_artifact=compile_artifact,
            )
            first = statement(asker.recorded)
            if not asker.missing:
                buffer = first
    try:
        if buffer is None:
            buffer = statement(asker.invoke if asker is not None else (replies.invoke if replies is not None else None))
    finally:
        if asker is not None:
            asker.close()
    if replies is not None:
        buffer["whole_program"]["package_replies"] = {
            "package_digest": replies.package_digest,
            "replayed": replies.hits,
            "asked": replies.misses,
            "pruned": prune_replies(replies.cache_root) if replies.cache_root is not None else None,
        }
    return buffer


#: The numeric extents each op's own entry vocabulary declares (``merlin.xdsl_dialects.lowering.
#: group_command``), for a diagnostic MAC/byte estimate -- never for binding a kernel argument. An
#: op not listed here (a reduction, a residual add) gets no shape facts and no derived bound: those
#: have no MAC-bearing contraction this estimate covers, and fabricating one would be worse than
#: omitting it.
_SHAPE_FIELDS_BY_OP: Mapping[str, tuple[str, ...]] = {
    "conv2d": ("N", "ci", "Himg", "Wimg", "kh", "kw", "stride", "padding"),
    "matmul": ("M", "K", "N"),
}


def _shape_facts(op: Any, entry: Any) -> dict[str, Any] | None:
    """The numeric shape extents ``entry`` declares for ``op``, or ``None`` when ``op`` is not one
    this estimate covers, or ``entry`` lacks a field it needs (never a fabricated default)."""
    fields = _SHAPE_FIELDS_BY_OP.get(str(op))
    if fields is None or not isinstance(entry, Mapping):
        return None
    facts = {name: entry.get(name) for name in fields}
    if any(facts[name] is None for name in fields):
        return None
    for key in ("operand_dtype", "output_dtype"):
        value = entry.get(key)
        if isinstance(value, str):
            facts[key] = value
    return facts


def _kernel_arg_order(target: str, command_buffer: Mapping[str, Any]) -> tuple[list[str], str, str]:
    """``(argument names in call order, contract shape, reason)`` -- the SELECTED support's answer.

    The order is the target contract's ``arg_order_by_command_shape`` row the buffer matches, resolved
    by the target's own RTL-check capability (``rocc_semantics.rtl_checks.resolve_kernel_arg_order``).
    A provider that declares no resolver leaves the order underivable, by name, never guessed.
    """
    from merlin.targetgen.rtl_checks import RtlChecksUnavailable, selected_checks

    try:
        resolve = getattr(selected_checks(target), "resolve_kernel_arg_order", None)
    except RtlChecksUnavailable as missing:
        return [], "", str(missing)
    if not callable(resolve):
        return [], "", f"the selected support for {target!r} declares no kernel argument order resolver"
    return resolve(dict(command_buffer))


#: A fused region whose kernel could not be linked: every member keeps the target's library call.
REGION_UNLINKED = "region_unlinked"
#: The target's driver refused to link a region the statement accepted (its own steps disagree).
DRIVER_REGION_REFUSED = "driver_region_refused"


def region_facts(target: str) -> dict[str, Any]:
    """What ``target``'s whole-model driver says about FUSED REGIONS: whether it links one
    (``links``) and which statement ops it can hold as a region's internal member (``internal_ops``).

    Read from the driver, never assumed: a driver that declares nothing links no region, and then no
    region is ever offered to a package -- a region the program cannot run as one call is not one."""
    from merlin.runtime.backends import base as backends

    try:
        kernels = backends.whole_model_driver(target).kernels
    except Exception as error:  # noqa: BLE001 -- no driver, no region; said, never assumed
        return {"links": False, "internal_ops": (), "why": f"{type(error).__name__}: {error}"}
    links = bool(getattr(kernels, "LINKS_FUSED_REGIONS", False))
    ops = tuple(str(o) for o in (getattr(kernels, "FUSED_REGION_INTERNAL_OPS", ()) or ())) if links else ()
    return {
        "links": links,
        "internal_ops": ops,
        "why": "" if links and ops else "the target's whole-model driver declares no fused-region linking",
    }


def linked_regions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``[{"members", "boundary"}]`` -- every fused region whose kernel the package answers here."""
    out = []
    for row in rows:
        region = row.get("region")
        if isinstance(region, Mapping) and region.get("role") == "boundary" and row.get("on") == ON_PACKAGE:
            out.append({"members": [int(g) for g in region["member_groups"]], "boundary": int(row["group"])})
    return out


def settle_regions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A region's INTERNAL members stand or fall with its boundary, in place.

    They have no kernel of their own -- the boundary's row carries the region's one kernel -- so when
    that kernel is not linked (a binding, an object, or the driver refused it) every member keeps the
    target's library call, with the boundary's reason. Never the other way round: an internal member
    credited to the package while its region's kernel never ran would be coverage nothing earned."""
    by_group = {int(r["group"]): r for r in rows}
    for row in rows:
        region = row.get("region")
        if not isinstance(region, Mapping) or region.get("role") != "internal" or row.get("on") != ON_PACKAGE:
            continue
        boundary = by_group.get(int(row["graded_at"]))
        if boundary is None or boundary.get("on") != ON_PACKAGE:
            row.update(
                {
                    "on": ON_VENDOR,
                    "cause": REGION_UNLINKED,
                    "why": f"its region's kernel (answered at group {row['graded_at']}) was not linked: "
                    f"{(boundary or {}).get('cause')}: {str((boundary or {}).get('why') or '')[:300]}",
                }
            )
    return rows


def bind_groups(buffer: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per group: who answers it, and -- for the package -- its kernel's arguments in ABI order.

    The ORDER is the contract row the package's own buffer resolves to; WHICH BUFFER each argument is
    comes from the binding the splice recorded. A private buffer (one the program does not hold) is
    admitted only with the package's own recipe for building it. Anything else is a refusal with a
    cause, and the group keeps the target's library call.
    """
    orders: dict[str, tuple[list[str], str, str]] = {}
    from merlin.llvmlower.whole_program import FROM_SUBMISSION

    route = buffer.get("whole_program") or {}
    prepacked = route.get("prepacked") or {}
    rows: list[dict[str, Any]] = []
    domain = route.get("input_domain")
    if isinstance(domain, Mapping) and domain.get("group") is not None:
        rows.append(
            {
                "group": int(domain["group"]),
                "op": "quantize",
                "on": ON_HOST,
                "why": "the model argument put on the integer grid: the one host region a closed model "
                "keeps, performed by the caller before the first command reads it",
            }
        )
    stated = list(route.get("per_group") or ())
    # Every member's own operands, by region: the boundary's kernel reads any of them.
    member_operands: dict[str, dict[str, dict[str, Any]]] = {}
    for row in stated:
        region = row.get("region")
        if isinstance(region, Mapping) and row.get("on") == FROM_SUBMISSION:
            member_operands.setdefault(str(region.get("id")), {})[str(row["group"])] = dict(row.get("operands") or {})
    for row in stated:
        base = {
            "group": int(row["group"]),
            "op": row.get("op"),
            "operands": dict(row.get("operands") or {}),
        }
        region = row.get("region")
        if isinstance(region, Mapping):
            # THE WHOLE REGION IS ONE CLAIM, RECORDED ON EVERY MEMBER -- never dropped, so attribution
            # can say "these groups were answered together, as one region kernel". A region member is
            # otherwise an ORDINARY package-answered row: it carries its own share of the region's
            # commands (`whole_program.emit_region` split them by name, ending at its own committed
            # dst) and binds its own kernel arguments exactly like any other group below, because a
            # linked program still expects one sized, committed buffer per device group.
            base["region"] = dict(region)
            if region.get("role") == "internal" and row.get("on") == FROM_SUBMISSION:
                # AN INTERNAL MEMBER HAS NO KERNEL OF ITS OWN: the region's one kernel is its
                # boundary's, which answers it. Package-answered while that kernel is linked
                # (:func:`settle_regions` holds it to that), and graded at the boundary.
                rows.append({**base, "on": ON_PACKAGE, "graded_at": int(region["member_groups"][-1])})
                continue
            if region.get("role") == "boundary":
                base["member_operands"] = member_operands.get(str(region.get("id")), {})
        shape_facts = _shape_facts(row.get("op"), row.get("entry"))
        if shape_facts is not None:
            # DIAGNOSTIC ONLY -- never a binding. Which buffer a kernel argument is comes from
            # ``args``/``command_buffer`` below, entirely separately; this is a MAC/byte estimate a
            # reader derives against the target's own facts, kept whether the group answers on the
            # package, the host or the library, so a gap holder's headroom can be read against a
            # bound regardless of who ran it.
            base["shape_facts"] = shape_facts
        asked = row.get("asked") or {}
        if asked.get("interface"):
            base["interface"] = asked["interface"]
        if row.get("absorbs"):
            # A NEGOTIATED group: the package took this group and the sum(s) listed here as one kernel,
            # so those groups have no call of their own and their output is this one's.
            base["absorbs"] = [int(g) for g in row["absorbs"]]

        def refuse(cause: str, why: str, base=base) -> dict[str, Any]:
            return {**base, "on": ON_VENDOR, "cause": cause, "why": why}

        if row.get("on") != FROM_SUBMISSION:
            rows.append(refuse(str(row.get("cause") or ""), str(row.get("why") or "")))
            continue
        cb_path, artifact_path = asked.get("command_buffer"), asked.get("artifact")
        if not cb_path or not artifact_path:
            rows.append(
                refuse(
                    SUBMISSION_UNREADABLE,
                    f"the splice reports this group answered by the submission, but records no "
                    f"{'command_buffer' if not cb_path else 'artifact'} path to read its reply from",
                )
            )
            continue
        cb_text = Path(cb_path).read_text(encoding="utf-8")
        command_buffer = json.loads(cb_text)
        # The order is a function of the buffer (and the selected support, fixed for this call): groups
        # that share an interface share a buffer, so it is resolved once per distinct buffer.
        if cb_text not in orders:
            orders[cb_text] = _kernel_arg_order(str(buffer.get("target") or ""), command_buffer)
        order, shape, why = orders[cb_text]
        order = list(order)
        if not order:
            rows.append(refuse(ABI_UNDERIVABLE, why))
            continue
        binding = asked.get("binding") or {}
        recipes = {
            str(r.get("target")): r
            for r in ((command_buffer.get("params") or {}).get("im2col_recipes") or ())
            if isinstance(r, Mapping)
        }
        args: list[dict[str, Any]] = []
        refusal = None
        for name in order:
            bound = binding.get(name)
            if bound is None:
                refusal = refuse(
                    ARGUMENT_UNBOUND, f"the ABI calls with {name!r}, which the package's buffer never declared"
                )
                break
            arg = {"tensor": name, "role": bound["role"], "declared": list(bound["declared"])}
            if bound["bound"]:
                arg["program"] = bound["program"]
                digest = (prepacked.get(bound["program"]) or {}).get("sha256")
                if digest:
                    arg["sha256"] = digest
                args.append(arg)
                continue
            recipe = recipes.get(name)
            source = binding.get(str((recipe or {}).get("source")))
            if recipe is None or source is None or not source["bound"]:
                refusal = refuse(
                    PRIVATE_WITHOUT_RECIPE,
                    f"the kernel reads {name!r}, a buffer private to it, and the package declares no recipe "
                    f"that builds it from a buffer this program holds",
                )
                break
            arg["gather"] = {"recipe": dict(recipe), "source": source["program"], "source_declared": source["declared"]}
            args.append(arg)
        if refusal is not None:
            rows.append(refusal)
            continue
        rows.append(
            {
                **base,
                "on": ON_PACKAGE,
                "shape": shape,
                "args": args,
                "command_buffer": cb_path,
                "artifact": artifact_path,
            }
        )
    return settle_regions(sorted(rows, key=lambda r: r["group"]))


def _prohibited(target: str, roles: Sequence[str]) -> dict[int, str]:
    """The selectors of every instruction the target's ISA facts give one of ``roles`` (derived)."""
    from .isa_prohibition import prohibited_instructions

    return prohibited_instructions(target, roles)


def pointee_row_padding(target: str) -> dict[str, Any]:
    """The row padding the backend contract declares for a kernel's pointer arguments, for ``target``.

    The contract (``kernel_abi.pointee_layout``) states every pointee is row-major with its rows
    zero-padded to the device's tile edge; the edge itself is the target's, derived from its RTL facts
    the same way the capsule shim derives it (:func:`merlin.llvmlower.device_shim.tile_edge_for`).
    ``multiple`` is None when the edge cannot be derived -- a caller must then refuse to hand a kernel
    a buffer whose rows would need padding, never hand it the dense one."""
    from merlin.llvmlower import device_shim

    abi = device_shim.kernel_abi_for(target)
    edge = device_shim.tile_edge_for(target)
    return {
        "layout": abi.pointee_layout if abi is not None else None,
        "multiple": int(edge) if edge else None,
        "source": "kernel_abi.pointee_layout (mlir_oot_backend_contract.yaml), tile edge from the RTL facts",
    }


def _decline_key(item: Any) -> tuple[str, Any]:
    """``("group", index)`` for a group named by index (``33`` or ``"g33"``), else ``("op", name)``."""
    if isinstance(item, bool):
        raise WholeModelBuildError(f"decline entry {item!r} is neither an op name nor a group index")
    if isinstance(item, int):
        return ("group", item)
    text = str(item)
    if text[:1] == "g" and text[1:].isdigit():
        return ("group", int(text[1:]))
    if text.isdigit():
        return ("group", int(text))
    return ("op", text)


def decline_ops(rows: list[dict[str, Any]], decline: Sequence[Any]) -> list[dict[str, Any]]:
    """Route package-answered groups the caller names to the target's library. In place.

    ``decline`` names OPS (``"residual_add"``: every group of that op) and GROUPS (``33`` or ``"g33"``:
    that one group, by its index in the model's own group order). For an experiment that must hold
    groups constant across two packages -- e.g. to measure a single lever with known-defective groups
    kept out of both programs -- each such group keeps the target's library call with a
    ``caller_declined`` cause, and ``declined_as`` says whether the op or the group was named. A group
    the package already did not answer keeps its own reason: the caller's decline never overwrites the
    package's. A declined op or group the model does not have is REFUSED -- a misspelled entry would
    otherwise decline nothing and the two programs would silently differ.
    """
    keys = [_decline_key(item) for item in decline]
    if not keys:
        return rows
    ops = {v for k, v in keys if k == "op"}
    groups = {v for k, v in keys if k == "group"}
    present_ops = {str(r.get("op")) for r in rows}
    present_groups = {int(r["group"]) for r in rows}
    unknown = sorted(ops - present_ops) + [f"g{g}" for g in sorted(groups - present_groups)]
    if unknown:
        raise WholeModelBuildError(
            f"decline names {unknown}, which this model does not have; its ops are {sorted(present_ops)} "
            f"and its groups g{min(present_groups)}..g{max(present_groups)}"
        )
    named = sorted(ops) + [f"g{g}" for g in sorted(groups)]
    for row in rows:
        by_group, by_op = int(row["group"]) in groups, str(row.get("op")) in ops
        if row.get("on") != ON_PACKAGE or not (by_group or by_op):
            continue
        for key in ("shape", "args", "command_buffer", "artifact"):
            row.pop(key, None)
        what = f"group g{row['group']}" if by_group else f"every {row.get('op')!r} group"
        row.update(
            {
                "on": ON_VENDOR,
                "cause": CALLER_DECLINED,
                "declined_as": "group" if by_group else "op",
                "why": f"the caller routed {what} to the target's library (decline={named}); the "
                f"package's kernel for it was not linked",
            }
        )
    return rows


# ------------------------------------------------------------------------------------------ objects

#: The per-group compiled-object cache lives in its own module (:mod:`merlin.perf.whole_model_object_cache`)
#: -- everything about WHAT decides a compiled object's bytes and how the cache is keyed, bounded and
#: pruned. These names are re-exported so existing callers of this module keep working unchanged.
OBJECT_CACHE_NAMESPACE = _OC.OBJECT_CACHE_NAMESPACE
OBJECT_CACHE_DISABLE_ENV = _OC.OBJECT_CACHE_DISABLE_ENV
OBJECT_CACHE_MAX_BYTES_ENV = _OC.OBJECT_CACHE_MAX_BYTES_ENV
OBJECT_CACHE_DEFAULT_MAX_BYTES = _OC.OBJECT_CACHE_DEFAULT_MAX_BYTES
prune_object_cache = _OC.prune
_object_cache_root = _OC.object_cache_root
_object_cache_fingerprint = _OC.object_cache_fingerprint
_object_cache_key = _OC.object_cache_key
_object_cache_store = _OC.store


def _kernel_objects(rows: list[dict[str, Any]], *, target: str, out: Path, jobs: int) -> dict[str, int]:
    """Compile each answered group's artifact to an object whose kernel has its own symbol. In place.

    The object is built by the capsule runner's own function for the target's own ISA; the symbol the
    contract declares is then renamed per group so seventy kernels can share one program, and the
    rename is CHECKED (the new symbol defined, the old one gone) rather than assumed. A group whose
    artifact text is byte-identical to a previously compiled one (same target, same toolchain) reuses
    that PRE-rename object from :data:`OBJECT_CACHE_NAMESPACE` instead of recompiling; the rename and
    the symbol check still run on every group, fresh or cached, so the linked symbol is always verified.

    A REGION'S MEMBERS SHARE ONE COMPILE, in memory, for the life of this one build. Every member of a
    region answered as one kernel (:mod:`merlin.llvmlower.region_capsule`) records the identical
    artifact path -- it is the SAME reply -- so without this a region's N members would each ask the
    (possibly disk-cached) compile step, N times, in the same build. Keyed by artifact TEXT,
    thread-safely; whichever member asks first still goes through the persistent on-disk cache below,
    every later member in this build just reads that first member's already-compiled object.
    """
    from merlin.llvmlower import toolchain
    from merlin.targetgen.contract.compile import llvm_mlir_to_object
    from merlin.targetgen.contract.harness_abi import for_target

    entry = for_target(target).entry_symbol
    cache_root = _object_cache_root()
    fingerprint = None
    if cache_root is not None:
        try:
            fingerprint = _object_cache_fingerprint(target)
        except Exception:  # noqa: BLE001 -- an unfingerprintable toolchain disables the cache, not the build
            cache_root = None
    compiled_by_artifact: dict[str, tuple[Path, str]] = {}
    # ONE LOCK PER DISTINCT ARTIFACT, never one for the build: identical artifacts (a region's members,
    # two groups with one signature) compile once, and different ones compile concurrently. A single
    # lock around the compile serialized every group's object behind the slowest.
    text_locks: dict[str, threading.Lock] = {}
    table_lock = threading.Lock()

    def compiled_object(artifact_text: str, work: Path) -> tuple[Path, str]:
        """(compiled ``.o`` path, ``object_cache`` state) for this artifact TEXT -- ``"hit"`` off the
        persistent disk cache, ``"shared"`` off this build's own in-memory memo (another group with the
        same artifact already compiled it), or ``"miss"`` (compiled here, and disk-cached for later)."""
        with table_lock:
            lock = text_locks.setdefault(artifact_text, threading.Lock())
        with lock:
            cached = compiled_by_artifact.get(artifact_text)
            if cached is not None:
                return cached[0], "shared"
            cached_path = None
            if cache_root is not None and fingerprint is not None:
                key = _object_cache_key(artifact_text, target=target, fingerprint=fingerprint)
                cached_path = cache_root / key[:2] / f"{key}.o"
            # A HIT is re-verified against the digest recorded when it was written (`_OC.load`), so a
            # corrupted or truncated entry is a miss -- recompiled -- never a wrong answer served with
            # confidence.
            if cached_path is not None:
                _OC.await_inflight(key)  # a prewarm of this text still compiling: its object, not a second compile
            verified = _OC.load(cached_path) if cached_path is not None else None
            if verified is not None:
                work.mkdir(parents=True, exist_ok=True)
                compiled = work / "kernel.o"
                shutil.copyfile(verified, compiled)
                state = "hit"
            else:
                compiled = llvm_mlir_to_object(artifact_text, work, target=target)
                if cached_path is not None:
                    _object_cache_store(cached_path, compiled)
                state = "miss"
            compiled_by_artifact[artifact_text] = (compiled, state)
            return compiled, state

    def one(row: dict[str, Any]) -> None:
        group = int(row["group"])
        work = out / f"g{group}"
        symbol = f"{entry}_g{group}"
        try:
            artifact_text = Path(row["artifact"]).read_text(encoding="utf-8")
            compiled, object_cache = compiled_object(artifact_text, work)
            renamed = out / f"g{group}.o"
            subprocess.run(
                [str(toolchain.objcopy()), f"--redefine-sym={entry}={symbol}", str(compiled), str(renamed)],
                check=True,
                capture_output=True,
                text=True,
            )
            listed = subprocess.run(
                [str(toolchain.nm()), "--defined-only", str(renamed)], check=True, capture_output=True, text=True
            ).stdout
            names = {line.split()[-1] for line in listed.splitlines() if line.split()}
            if symbol not in names or entry in names:
                raise WholeModelBuildError(f"the renamed object defines {sorted(names & {entry, symbol})}")
        except Exception as error:  # noqa: BLE001 -- a group that will not build is a named fallback
            detail = getattr(error, "stderr", "") or str(error)
            row.update(
                {"on": ON_VENDOR, "cause": OBJECT_FAILED, "why": f"{type(error).__name__}: {str(detail)[-400:]}"}
            )
            return
        row.update(
            {
                "object": str(renamed),
                "symbol": symbol,
                "object_sha256": _sha256(renamed),
                "object_cache": object_cache,
                "compiled_signature": hashlib.sha256(artifact_text.encode("utf-8")).hexdigest(),
            }
        )

    # A fused region's internal members carry no kernel of their own: their region's is the boundary's.
    answered = [row for row in rows if row["on"] == ON_PACKAGE and row.get("artifact")]
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        list(pool.map(one, answered))
    if cache_root is not None:
        with contextlib.suppress(Exception):  # noqa: BLE001 -- pruning is an optimization, never a build dependency
            prune_object_cache(cache_root)
    # WHAT THE DEDUPE BOUGHT: a repeated model block emits byte-identical artifacts, compiled once each.
    return {"unique_signatures": len(compiled_by_artifact), "total_groups": len(answered)}


# ------------------------------------------------------------------------------------ corpus binding


@dataclasses.dataclass(frozen=True)
class CorpusBinder:
    """The binder a build states its groups under, and the record of what it was derived from."""

    binder: Any
    record: dict[str, Any]


def corpus_binder(
    target: str, *, phase0_recipe: str | Path | None, descriptor: str | Path | None = None
) -> CorpusBinder:
    """The corpus binding a package build states every group under, from EXPLICIT inputs only.

    ``phase0_recipe`` is the Phase 0 recipe whose ``datapath`` block the corpus is generated under --
    the same block :func:`merlin.targetgen.group_capsule_entries.group_binding` takes -- so a group a
    build asks a package for and the capsule that certifies it are the same bytes. ``descriptor`` is
    the target descriptor the experiment loads; without it only the target name is carried (the
    recipe must then declare its oracle tiers, which is the only thing a descriptor adds). Nothing is
    discovered: a missing recipe is a refusal, not a default.
    """
    import types

    import yaml

    from merlin.llvmlower import whole_program as WP

    if phase0_recipe is None:
        raise WholeModelBuildError(
            "a package build states each group under the corpus binding, which the Phase 0 recipe's "
            "datapath block declares; name the recipe (phase0_recipe=)"
        )
    path = Path(phase0_recipe)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    datapath = doc.get("datapath") if isinstance(doc, Mapping) else None
    if not isinstance(datapath, Mapping):
        raise WholeModelBuildError(f"{path} declares no datapath block, so no corpus binding can be derived")
    if descriptor is not None:
        from merlin.targetgen.target_experiment import load_target_experiment

        te = load_target_experiment(descriptor)
        if te.target != target:
            raise WholeModelBuildError(f"{descriptor} describes {te.target!r}, not {target!r}")
    else:
        te = types.SimpleNamespace(target=target, sim_via="")
    record = {
        "phase0_recipe": str(path.resolve()),
        "phase0_recipe_sha256": _sha256(path),
        "descriptor": str(Path(descriptor).resolve()) if descriptor is not None else None,
        "descriptor_sha256": _sha256(descriptor) if descriptor is not None else None,
        "datapath": dict(datapath),
    }
    return CorpusBinder(binder=WP.group_binder(te, dict(datapath)), record=record)


# ------------------------------------------------------------------------------------------- oracle
#
# The independent oracle and the LOCAL grade -- `LocalReference`, `_oracle`, `grade`, the merged-group
# restatement and `verify_passes_preserve_semantics` -- live in `whole_model_oracle.py` (this
# module's own size gate), and are re-exported here so every existing caller of
# `whole_model_build._oracle` / `.grade` / `.memory_map` / `.LocalReference` / ... needs no change.
from .whole_model_oracle import (  # noqa: E402
    LocalReference,
    OracleJob,
    grade,
    verify_passes_preserve_semantics,
)
from .whole_model_oracle import (  # noqa: E402
    _entry_array as _entry_array,
)

# -------------------------------------------------------------------------------------------- build


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


from .whole_model_headers import _headers_read, _with_header, machine_header  # noqa: E402  (re-exported)


def _apply_package_passes(
    capsule: ModelCapsule, *, target: str, package_dir: str | Path, out: Path, timeout: int
) -> dict[str, Any]:
    """Run ``package_dir``'s own declared whole-model passes over ``capsule``'s interface, verify the
    result computes the same function, and say what happened either way. Never raises: a package that
    declares no passes, or one whose manifest cannot even name them correctly, is recorded and this
    build proceeds on the capsule's OWN interface exactly as it would have without this existing.
    """
    from . import whole_model_passes as WMPass

    try:
        package = _load_pass_package(package_dir)
        names = WMPass.declared_passes(package)
    except WMPass.PassDeclarationError as error:
        return {"applied": False, "declared": (), "why": str(error)}
    if not names:
        return {"applied": False, "declared": ()}
    transformed, records, stopped = WMPass.apply_passes(
        capsule.interface, package, work=out / "passes", timeout=timeout
    )
    if stopped or transformed == Path(capsule.interface):
        return {"applied": False, "declared": list(names), "pass_records": records, "why": stopped or "no pass ran"}
    verdict = verify_passes_preserve_semantics(capsule, target=target, transformed_interface=transformed)
    if not verdict["ok"]:
        return {
            "applied": False,
            "declared": list(names),
            "pass_records": records,
            "interface": str(transformed),
            "verification": verdict,
            "why": f"the transformed model failed semantic verification: {verdict['why']}",
        }
    return {
        "applied": True,
        "declared": list(names),
        "pass_records": records,
        "interface": str(transformed),
        "verification": verdict,
    }


def _load_pass_package(package_dir: str | Path) -> Any:
    from merlin.targetgen import oot_runner as OR

    return OR.load_package(str(package_dir))


def _pins_for(target: str) -> dict[str, Any]:
    from merlin.common import provenance as PROV

    verified: dict[str, Any] = {}
    for name, pin in PROV.load_pins().items():
        if target in (pin.targets or ()):
            try:
                verified[name] = PROV.verify(name)
            except Exception:  # noqa: BLE001 -- an unverifiable pin is recorded by name, never omitted
                continue
    return verified


def _default_out(target: str, capsule: ModelCapsule) -> Path:
    from merlin.common.artifacts import git_sha7, utc_stamp
    from merlin.common.paths import artifacts_dir

    return artifacts_dir() / "perf-bench" / target / "whole-model-build" / capsule.name / f"{utc_stamp()}_{git_sha7()}"


#: The build's stages, in the order :func:`build` runs them; each is timed by :class:`_StageClock` and is a
#: compile-trace stage whose products are the files it wrote under the build's directory.
BUILD_STAGES = _trace.declare(
    "whole-model",
    ("passes", "statement", "group_objects", "extract", "render_kernels", "program_build", "memory_map", "oracle_join"),
    entry="merlin.perf.whole_model_build.build",
    summary="package passes -> per-group statement (lower/gN.*: interface, command buffer, target IR) -> "
    "objects -> kernels -> program (C + ELF) -> memory map -> oracle",
)


class _StageClock:
    """Wall seconds per named build stage, in the order the stages ran. Under an open compile trace each
    stage also reports the files it wrote below ``root`` and is a point the build can stop at."""

    def __init__(self) -> None:
        import time

        self._clock = time.monotonic
        self._started = self._clock()
        self._last = self._started
        self._stages: dict[str, float] = {}
        self.root: Path | None = None

    @contextlib.contextmanager
    def __call__(self, name: str):
        began = self._clock()
        before = _trace.snapshot(self.root) if self.root is not None else None
        try:
            yield
        finally:
            self._last = self._clock()
            self._stages[name] = round(self._stages.get(name, 0.0) + self._last - began, 3)
        if before is not None:  # only a stage that completed reports, and only under an open trace
            written = _trace.written_since(self.root, before)
            _trace.artifact(name, written, pipeline="whole-model", seconds=self._last - began)
            _trace.stop_if_reached()

    def mark(self, name: str) -> None:
        """Close a stage that began where the previous stage (or mark) ended, without a block."""
        now = self._clock()
        self._stages[name] = round(self._stages.get(name, 0.0) + now - self._last, 3)
        self._last = now

    def record(self) -> dict[str, Any]:
        return {"stages": dict(self._stages), "total": round(self._clock() - self._started, 3)}


def require_datapath_facts(target: str) -> dict[str, Any]:
    """The target's RTL facts, refused by name when they carry no datapath.

    Grouping places a contraction's epilogue (its requantize, its scale) on the device only when the
    readout the facts derive can take it. Without datapath facts nothing can, every epilogue lands on
    the host, and the model silently becomes an OPEN one -- host code between every group -- with
    roughly twice the groups. That is a missing input, not a model, so the build refuses."""
    from merlin.targetgen.rtl import facts as F

    try:
        doc = F.load_facts(target)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise WholeModelBuildError(f"no datapath facts for {target}: {type(exc).__name__}: {exc}") from exc
    if not ((doc or {}).get("facts") or {}).get("datapaths"):
        raise WholeModelBuildError(
            f"no datapath facts for {target}: its RTL facts carry no extracted datapath, so no epilogue could "
            "be placed on the device and the model would build as an open one"
        )
    return doc


def refuse_if_every_group_declined(package_dir: str | Path | None, attribution: Sequence[Mapping[str, Any]]) -> None:
    """A PACKAGE build in which the package answers no group at all, for a reason other than the
    caller's own decline, is refused: its program is the library's (or the host's) and every cycle it
    measures would be credited to a package that did nothing.  The reasons are stated, most common first."""
    if package_dir is None or not attribution or any(r.get("on") == ON_PACKAGE for r in attribution):
        return
    reasons: dict[str, int] = {}
    for row in attribution:
        cause = row.get("declined_as") or row.get("cause")
        if cause == "caller_declined" or (row.get("on") == ON_HOST and cause is None):
            continue
        why = f"{cause}: {str(row.get('declined_why') or row.get('why') or '')[:200]}"
        reasons[why] = reasons.get(why, 0) + 1
    if not reasons:
        return  # the caller declined every group itself
    ranked = sorted(reasons.items(), key=lambda kv: -kv[1])
    raise WholeModelBuildError(
        f"the package answered none of {len(attribution)} group(s); "
        + "; ".join(f"{n} x {why}" for why, n in ranked[:3])
    )


def build(
    package_dir: str | Path | None,
    model_capsule: str | Path,
    *,
    target: str,
    machine: str,
    header: str | Path,
    header_sha256: str | None = None,
    out: str | Path | None = None,
    oracle: bool = True,
    verify: str = "on_target",
    timeout: int = 600,
    jobs: int | None = None,
    decline: Sequence[Any] = (),
    harness_overrides: Sequence[str | Path] = (),
    prohibited_roles: Sequence[str] = (),
    allow_passes: bool = False,
    allow_regions: bool = False,
    phase0_recipe: str | Path | None = None,
    descriptor: str | Path | None = None,
    only_groups: Sequence[Any] = (),
) -> dict[str, Any]:
    """Build ``model_capsule`` as one runnable program whose kernels are ``package_dir``'s, with its oracle.

    Inputs:
      * ``package_dir`` -- a compiler package (its ``manifest.yaml`` declares the entrypoints), or
        ``None`` for the target's library program: every group on the vendor call, attributed so;
      * ``model_capsule`` -- a model capsule directory (``capsule.yaml``, interface, weights, golden);
      * ``target`` -- the target name; everything target-specific is read through it;
      * ``machine`` -- the hardware-registry entry the program is built FOR (the device it will run on);
      * ``header`` -- the vendor parameter header for that machine, an explicit file whose sha256 must
        equal the ABI header the registry declares for ``machine`` (or ``header_sha256`` when the
        registry declares none -- recorded as the caller's assertion); see :func:`machine_header`;
      * ``verify`` -- where each group's output is checked: ``on_target`` digests it on the core
        after the measured window (right for FireSim, ruinous on an elaborated-RTL simulator), or
        ``host_dump``, which emits no on-target check and records ``<out>/memory_map.json`` -- every
        group's output buffer by symbol, address and size -- for :func:`grade_memory` over a dump; or
        ``local``, which is ``on_target`` plus each exact group's reference recomputed ON THE CORE from
        the inputs it actually held (one ``GM_LOCAL`` line per group; the gate :func:`grade` reads);
      * ``out`` -- the product directory; default
        ``out/artifacts/perf-bench/<target>/whole-model-build/<capsule>/<TS>_<sha7>/``;
      * ``decline`` -- op names (``bind_groups``' ``op``, e.g. ``residual_add``) and group indices
        (``33`` / ``"g33"``) whose groups keep the target's library call even where the package
        answered them, each recorded with the ``caller_declined`` cause (see :func:`decline_ops`);
      * ``harness_overrides`` -- further files of the target's harness tree (e.g. a patched vendor
        library header) replaced by namesake in the build's copy of it, exactly as ``header`` is; each
        is recorded by digest under ``program.harness_overrides``, and the program must have read it;
      * ``allow_passes`` -- OFF BY DEFAULT. When set and ``package_dir`` declares
        ``whole_model_passes`` in its manifest, each is run over the capsule's own interface MLIR, in
        order (:func:`merlin.perf.whole_model_passes.apply_passes`), and the result is used for the
        rest of this build ONLY if it verifies as computing the same function
        (:func:`verify_passes_preserve_semantics`) -- never trusted from the pass's own claim, and
        never applied at all for a package that declares none;
      * ``allow_regions`` -- OFF BY DEFAULT. When set, the target's driver links a FUSED REGION
        (:func:`region_facts`) AND the package's own manifest declares ``whole_model_regions: true``,
        it may be offered a legal run of consecutive groups as one kernel
        (:func:`merlin.llvmlower.region_legality.legal_regions`) in addition to each group alone. An
        answered region is ONE step of the program -- its kernel called once in its members' place,
        graded at its boundary, every member package-answered (``regions`` in the record says which);
        a package that never declares that opt-in sees no change at all, whatever this flag is (see
        :func:`merlin.llvmlower.whole_program.whole_program_buffer`);
      * ``phase0_recipe`` / ``descriptor`` -- the Phase 0 recipe whose ``datapath`` block the corpus is
        built under, and the target descriptor (see :func:`corpus_binder`). Required with a package:
        each group is put to it as the capsule the corpus would write for that group, never under a
        binding assumed here;
      * ``only_groups`` -- ask the package for these groups alone (``["g12"]``); every other group is
        declined to the target's library and the result is a PARTIAL build, marked so on the record,
        the oracle and ``PARTIAL_BUILD.json`` and refused as a whole model (:mod:`.whole_model_partial`).

    Returns the build record (also written to ``<out>/whole_model_build.json``, with the oracle in
    ``<out>/oracle.json``): the ELF and its digest; per-group ATTRIBUTION -- ``package`` (the kernel is
    the package's), ``vendor`` (the target's library call, with the package's or this build's named
    reason) or ``host`` -- and each linked object's digest; the program's compiler, flags and every
    header it read; and ``provenance`` (``merlin.common.provenance.record()``, with the package's
    content digest). Raises :class:`WholeModelBuildError` when the model cannot be one program at all.
    """
    from merlin.common import provenance as PROV
    from merlin.common.artifacts import dump_yaml
    from merlin.runtime.backends import base as backends

    stages = _StageClock()
    capsule = load_model_capsule(model_capsule)
    # BEFORE ANY WORK: a header that is not the machine's is a wrong program, not a slow one.
    abi_header = machine_header(machine, header, header_sha256)
    require_datapath_facts(target)
    out = Path(out) if out is not None else _default_out(target, capsule)
    out.mkdir(parents=True, exist_ok=True)
    stages.root = out
    jobs = jobs or min(16, os.cpu_count() or 1)
    if only_groups:
        decline = [*decline, *_partial.unasked(capsule, target=target, only=only_groups)]
    binding = corpus_binder(target, phase0_recipe=phase0_recipe, descriptor=descriptor) if package_dir else None

    passes_record: dict[str, Any] | None = None
    if allow_passes and package_dir is not None:
        with stages("passes"):
            passes_record = _apply_package_passes(
                capsule, target=target, package_dir=package_dir, out=out, timeout=timeout
            )
        if passes_record.get("applied"):
            capsule = dataclasses.replace(capsule, interface=Path(passes_record["interface"]))

    driver = backends.whole_model_driver(target)
    # A FUSED REGION IS OFFERED ONLY WHERE THE TARGET'S DRIVER LINKS ONE: its own declaration, never
    # assumed. Without it `allow_regions` changes nothing, and the record says why.
    regions_fact = region_facts(target) if allow_regions else {"links": False, "internal_ops": (), "why": "off"}
    # THE ORACLE DEPENDS ON NOTHING THE PACKAGE DOES: started now, in its own process (or read back by
    # its content key), and joined only when the record needs it.
    oracle_job = OracleJob(capsule, target=target) if oracle else None
    with stages("statement"):
        buffer = state(
            capsule,
            target=target,
            package_dir=package_dir,
            work=out / "lower",
            timeout=timeout,
            jobs=jobs,
            allow_regions=allow_regions and bool(regions_fact["internal_ops"]),
            decline=decline,
            binder=binding.binder if binding is not None else None,
            prewarm_objects=True,
            region_internal_ops=regions_fact["internal_ops"],
        )
    # decline_ops runs AFTER the buffer is already stated with every declined group's real reference
    # commands and committed buffer (via `state(..., decline=decline)` above): it validates the
    # declared names against the model's own ops/groups and is a no-op relabeling for a row already
    # `on: vendor` from that statement -- kept for that validation, not to do the routing itself, which
    # used to happen here instead and left the buffer holding the package's commands for a group with
    # neither a package kernel nor a library one.
    rows = decline_ops(bind_groups(buffer), decline)
    with stages("group_objects"):
        object_dedup = _kernel_objects(rows, target=target, out=out / "objects", jobs=jobs)
    settle_regions(rows)

    (value,) = capsule.inputs.values()
    (golden,) = capsule.outputs.values()
    with stages("extract"):
        model = driver.program.extract(
            None,
            target,
            sources={
                "linalg": capsule.interface,
                "weights_manifest": capsule.weights_manifest,
                "weights": capsule.weights,
                "input": value,
                "golden": golden,
            },
        )
    # EACH LINKED FUSED REGION BECOMES ONE STEP of the driver's program -- its kernel called once, in its
    # members' place, graded at its boundary. A region the driver's own steps refuse, or whose kernel
    # the driver cannot bind, keeps every member an ordinary library step with that reason: a region
    # is a step only while its kernel answers it, never half-linked.
    domain = buffer["whole_program"]["input_domain"]
    extracted = model

    def demote(boundaries: Mapping[int, tuple[str, str]]) -> None:
        for row in rows:
            boundary = int(row.get("graded_at") or row["group"])
            if row.get("region") and boundary in boundaries and row["on"] == ON_PACKAGE:
                cause, why = boundaries[boundary]
                row.update({"on": ON_VENDOR, "cause": cause, "why": why})
        settle_regions(rows)

    with stages("render_kernels"):
        while True:
            regions = linked_regions(rows)
            model, refused = driver.program.region_steps(extracted, regions) if regions else (extracted, {})
            if refused:
                demote({g: (DRIVER_REGION_REFUSED, why) for g, why in refused.items()})
                continue
            kernels = driver.kernels.render_kernels(
                model, rows, entry=domain["tensor"], row_padding=pointee_row_padding(target)["multiple"]
            )
            unbound = {
                int(r["group"]): (str(r.get("cause") or ""), str(r.get("why") or ""))
                for r in kernels["census"]
                if r.get("kind") == "region" and r.get("on") != "submission"
            }
            if not unbound:
                break
            demote(unbound)
    overrides = {str(Path(o).resolve()): _sha256(o) for o in harness_overrides}
    recipe = _with_header(
        backends.harness_build_recipe(target), Path(header), out / "harness", [Path(o) for o in overrides]
    )
    with stages("program_build"):
        receipt = driver.program.build(
            model,
            None,
            None,
            out / "program",
            sched_kernels=kernels,
            extra_objects=[Path(o) for o in kernels["objects"]],
            recipe=recipe,
            verify=verify,
            **(
                {}
                if not prohibited_roles
                else {"library_loops": False, "prohibited_selectors": sorted(_prohibited(target, prohibited_roles))}
            ),
        )
    headers = _headers_read(receipt, out / "program" / "group_model_program.c")
    if abi_header["sha256"] not in headers.values():
        raise WholeModelBuildError(
            f"the program did not read the asserted parameter header {abi_header['sha256'][:12]}; "
            f"it read {sorted(set(headers.values()))[:4]}"
        )
    unread = [path for path, digest in overrides.items() if digest not in headers.values()]
    if unread:
        raise WholeModelBuildError(f"the program did not read the harness override(s) {unread}")

    # ATTRIBUTION, ONE ROW PER GROUP. The driver can still refuse a package kernel it cannot bind (its
    # buffers disagree with the statement); that refusal is the final word for the group.
    driven = {int(r["group"]): r for r in kernels["census"]}
    on_core = {int(r["group"]): r for r in receipt.get("host_routed") or ()}
    library_paths = receipt.get("library_paths") or {}
    attribution = []
    for row in rows:
        # A fused region's internal member is answered by its boundary's kernel, and follows it.
        graded_at = int(row.get("graded_at") or row["group"])
        final = driven.get(graded_at)
        entry = {
            k: row.get(k)
            for k in ("group", "op", "on", "cause", "declined_as", "why", "shape")
            if row.get(k) is not None
        }
        if row.get("region"):
            entry["region"] = {k: row["region"].get(k) for k in ("id", "member_groups", "role", "graded_at")}
            entry["graded_at"] = graded_at
        if row["on"] == ON_PACKAGE and final is not None and final.get("on") != "submission":
            entry.update({"on": ON_VENDOR, "cause": final.get("cause"), "why": final.get("why")})
        if entry["on"] == ON_PACKAGE and graded_at != int(row["group"]):
            entry["answered_by"] = f"the region kernel linked at group {graded_at}"
        elif entry["on"] == ON_PACKAGE:
            entry.update(
                {
                    "arguments": [a["tensor"] for a in row["args"]],
                    "gather": bool(final and final.get("gather")),
                    "object": row.get("object"),
                    "object_sha256": row.get("object_sha256"),
                }
            )
        # THE DRIVER'S OWN ROUTING IS THE LAST WORD: a group the machine cannot read out the way its
        # call needs (a derived fact of the machine's header) runs on the core, with the driver's cause.
        routed = on_core.get(graded_at)
        if routed is not None and graded_at != int(row["group"]):
            # The machine routes this member's REGION to the core (its boundary's readout): the
            # region's kernel does not run, and this member runs the target's library call.
            entry = {k: v for k, v in entry.items() if k in ("group", "op", "shape", "region", "graded_at")}
            entry.update(
                {
                    "on": ON_VENDOR,
                    "cause": REGION_UNLINKED,
                    "why": f"its region is routed to the core: {routed.get('why')}",
                }
            )
        elif routed is not None:
            entry = {k: v for k, v in entry.items() if k in ("group", "op", "shape")}
            entry.update({"on": ON_HOST, "cause": routed.get("cause"), "why": routed.get("why")})
        elif entry["on"] == ON_VENDOR and library_paths:
            # A LIBRARY GROUP ON A LOOP-FREE PATH (the driver's own choice, with its reason): on the host
            # when that path is the library's host code, else still the library's, on the named path.
            chosen = library_paths.get(str(entry.get("op"))) or library_paths.get("matmul") or {}
            entry["library_path"] = chosen.get("path")
            if chosen.get("host"):
                entry.update(
                    {
                        "on": ON_HOST,
                        "cause": "library_loop_free_path",
                        "why": chosen.get("why"),
                        "declined_as": entry.get("cause"),
                        # Why the group reached the library at all (an object that failed to build, a
                        # package decline) -- kept, not overwritten by the routing's own reason.
                        "declined_why": entry.get("why"),
                    }
                )
            else:
                entry["library_path_why"] = chosen.get("why")
        attribution.append(entry)
    counts = {side: sum(1 for r in attribution if r["on"] == side) for side in (ON_PACKAGE, ON_VENDOR, ON_HOST)}
    refuse_if_every_group_declined(package_dir, attribution)
    object_cache_counts = {
        outcome: sum(1 for r in rows if r.get("object_cache") == outcome) for outcome in ("hit", "miss")
    }

    with stages("memory_map"):
        layout = memory_map(receipt["elf"], model, buffer)
    (out / "memory_map.json").write_text(json.dumps(layout, indent=1) + "\n", encoding="utf-8")

    expected = None
    if oracle:
        with stages("oracle_join"):
            expected, entry_array = oracle_job.result()
        expected["cache"] = {"key": oracle_job.key, "state": oracle_job.state}
        # The driver quantized and laid out the same argument on its own; the two must be the same bytes.
        expected["entry_matches_program"] = bool(entry_array.tobytes() == model["arrays"]["IMAGE_DATA"].tobytes())
        (out / "oracle.json").write_text(json.dumps(expected, indent=1) + "\n", encoding="utf-8")

    record = {
        "schema": SCHEMA,
        "target": target,
        "capsule": {
            "name": capsule.name,
            "directory": str(capsule.directory),
            "interface_sha256": _sha256(capsule.interface),
            "weights_sha256": _sha256(capsule.weights),
        },
        "package": (
            {
                "directory": str(Path(package_dir).resolve()),
                "digest": buffer["whole_program"].get("package_replies", {}).get("package_digest"),
                "replies": buffer["whole_program"].get("package_replies"),
            }
            if package_dir is not None
            else None
        ),
        "elf": receipt["elf"],
        "elf_sha256": receipt["elf_sha256"],
        "program": {
            "source_sha256": receipt["program_sha256"],
            "compiler": receipt["compiler"],
            "flags": receipt["flags"],
            "link_flags": receipt.get("link_flags"),
            "link_script": receipt.get("link_script"),
            "includes": receipt.get("includes"),
            "headers_read": headers,
            "abi_header": abi_header,
            "harness_overrides": overrides,
            "host_routed": receipt.get("host_routed") or [],
            "library_paths": receipt.get("library_paths"),
            # The library header's copy with every macro that would issue a prohibited instruction
            # trapped: its digests before and after, and the macros it trapped.
            "loop_free_header": receipt.get("loop_free_header"),
            "im2col_scratch_elements": kernels["im2col_scratch_elements"],
        },
        "linked_objects": receipt["linked_objects"],
        "object_cache": {
            **object_cache_counts,
            "namespace": OBJECT_CACHE_NAMESPACE,
            "disabled": bool((os.environ.get(OBJECT_CACHE_DISABLE_ENV) or "").strip()),
            **object_dedup,
        },
        "verify": verify,
        # WHERE THE BUILD'S WALL TIME WENT, per stage, in order: a slow build is fixed at the stage
        # that is slow, and a total alone names none of them.
        "stage_seconds": stages.record(),
        "declined_by_caller": sorted(f"g{v}" if k == "group" else str(v) for k, v in map(_decline_key, decline)),
        # FUSED REGIONS: whether this target links one (its driver's own declaration), and every region
        # a package kernel answers in THIS program -- each graded at its boundary, which is where the
        # expectations put its one check.
        "regions": {
            "offered": bool(allow_regions and regions_fact["internal_ops"]),
            "why_not_offered": "" if allow_regions and regions_fact["internal_ops"] else regions_fact["why"],
            "internal_ops": list(regions_fact["internal_ops"]),
            # The regions stated as ONE STEP of this program (each prints one line, at its boundary)...
            "stepped": regions,
            # ...and those whose kernel the package answers in it (a region the machine routes to the
            # core runs its members' library calls instead, still as one step).
            "linked": [
                r for r in regions if any(e["group"] == r["boundary"] and e["on"] == ON_PACKAGE for e in attribution)
            ],
        },
        "passes": passes_record,
        "corpus_binding": binding.record if binding is not None else None,
        "memory_map": str(out / "memory_map.json"),
        "attribution": {"counts": counts, "per_group": attribution},
        "oracle": (
            {k: expected[k] for k in ("argmax", "golden_argmax", "cosine_to_golden", "entry_matches_program")}
            | {"path": str(out / "oracle.json")}
            if expected is not None
            else None
        ),
        "provenance": PROV.record(
            pins=_pins_for(target),
            sources=[capsule.interface],
            artifacts={"elf": receipt["elf"], "weights": capsule.weights},
            extra={"compiler_sources": driver.program._compiler_provenance()},
        ),
    }
    if only_groups:
        _partial.mark(record, out, only_groups)
    (out / "whole_model_build.json").write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    (out / "manifest.yaml").write_text(
        dump_yaml(
            {
                "schema": SCHEMA,
                "target": target,
                "capsule": capsule.name,
                **({_partial.MARKER: record[_partial.MARKER]} if only_groups else {}),
                "git_sha": record["provenance"]["merlin"]["commit"],
                "elf_sha256": receipt["elf_sha256"],
                "artifacts": [
                    "whole_model_build.json",
                    "oracle.json",
                    "memory_map.json",
                    "program/group_model_program.elf",
                ],
            }
        ),
        encoding="utf-8",
    )
    return record


# ---------------------------------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m merlin.perf.whole_model_build`` (:mod:`.whole_model_build_cli`)."""
    from .whole_model_build_cli import main as _main

    return _main(argv)


if __name__ == "__main__":
    sys.exit(main())
