"""A compiler package and a model capsule, built into ONE runnable whole-model program and its oracle.

    record = build(package_dir, model_capsule, target=target)

is the whole route, with no hand step in it. Until this existed every whole-model number that ran a
package's own kernels came from scratch scripts driven by hand: the package asked once per group, its
LLVM re-lowered through a second entrypoint call with ad-hoc ``llc`` flags, each kernel's arguments
re-derived from roles and extents the driver recomputed, and the oracle run separately. Each of those
was a second spelling of a fact the compiler route already states, free to disagree with it.

ONE PATH, THE CAPSULE'S. A capsule is a one-group model: every compute group of the model is put to the
package as a ``merlin_iface`` capsule through the SAME call the capsule runner makes
(:func:`merlin.development.targetgen.corpus.capsule_io.lower_interface`, reached through
:func:`merlin.compiler.lower.device.whole_program.whole_program_buffer`), and each reply's target artifact is
compiled to an object by the SAME function the capsule runner compiles one with
(:func:`merlin.development.targetgen.execution.runner.compile.llvm_mlir_to_object`). Nothing about a group is restated:

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
:func:`merlin.runtime.engine.reference.reference_outputs` recomputes every committed buffer from the leaf
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

from merlin.analysis.whole_model.build import inputs as _partial
from merlin.analysis.whole_model.build import products as _OC
from merlin.analysis.whole_model.build.inputs import (
    ModelCapsule,
    WholeModelBuildError,
    _model_golden,
    load_model_capsule,
)
from merlin.analysis.whole_model.build.oracle import (
    LocalReference,
    OracleJob,
    _entry_array,
    grade,
    verify_passes_preserve_semantics,
)
from merlin.analysis.whole_model.build.products import (
    REPLIES_DEFAULT_MAX_BYTES,
    REPLIES_MAX_BYTES_ENV,
    Asker,
    _headers_read,
    _ReplyCache,
    _with_header,
    drop_package_replies,
    machine_header,
    prune_replies,
    replies_folder,
)
from merlin.analysis.whole_model.build.whole_model_memory import MEMORY_MAP_SCHEMA, grade_memory, memory_map
from merlin.compiler.ir import compile_trace as _trace

# ---- whole_model_build -------------------------------------------------------------------------

SCHEMA = "whole_model_build_v1"


ON_PACKAGE = "package"


ON_VENDOR = "vendor"


ON_HOST = "host"


ABI_UNDERIVABLE = "kernel_abi_order_underivable"


ARGUMENT_UNBOUND = "kernel_argument_unbound"


PRIVATE_WITHOUT_RECIPE = "private_operand_without_recipe"


OBJECT_FAILED = "kernel_object_build_failed"


SUBMISSION_UNREADABLE = "submission_reply_unreadable"


CALLER_DECLINED = "caller_declined"


def package_digest(package_dir: str | Path) -> str:
    """A compiler package's content identity -- the ONE digest a build keys and records it by.

    :func:`merlin.common.digest.hash_tree` over the package directory, the same digest a measurement
    store keys a package on and a harness-owned package repository's commit tree recomputes, so a
    build record, a reply cache and a stored candidate all name one package the same way.
    """
    from merlin.common.digest import hash_tree

    identity = hash_tree(Path(package_dir))
    if not identity["present"]:
        raise WholeModelBuildError(f"the package directory {package_dir} does not exist")
    return str(identity["sha256"])


_package_digest = package_digest


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
    put to the package; see :func:`merlin.compiler.lower.device.whole_program.group_interface`.

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
    groups (see :func:`merlin.compiler.lower.device.whole_program.whole_program_buffer`).
    """
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.ir.xdsl_io import IR_LOCK
    from merlin.compiler.lower.device import whole_program as WP
    from merlin.compiler.partition import memory_plan as SP

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
        from merlin.development.targetgen.packaging import loading as OR

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
    from merlin.target.isa.rocc.rtl_checks import RtlChecksUnavailable, selected_checks

    try:
        resolve = getattr(selected_checks(target), "resolve_kernel_arg_order", None)
    except RtlChecksUnavailable as missing:
        return [], "", str(missing)
    if not callable(resolve):
        return [], "", f"the selected support for {target!r} declares no kernel argument order resolver"
    return resolve(dict(command_buffer))


REGION_UNLINKED = "region_unlinked"


DRIVER_REGION_REFUSED = "driver_region_refused"


def region_facts(target: str) -> dict[str, Any]:
    """What ``target``'s whole-model driver says about FUSED REGIONS: whether it links one
    (``links``) and which statement ops it can hold as a region's internal member (``internal_ops``).

    Read from the driver, never assumed: a driver that declares nothing links no region, and then no
    region is ever offered to a package -- a region the program cannot run as one call is not one."""
    from merlin.runtime.backends import registry as backends

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
    from merlin.compiler.lower.device.whole_program import FROM_SUBMISSION

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
    from merlin.analysis.whole_model.grading.isa_prohibition import prohibited_instructions

    return prohibited_instructions(target, roles)


def pointee_row_padding(target: str) -> dict[str, Any]:
    """The row padding the selected kernel ABI declares for a kernel's pointer arguments, for ``target``.

    Under the logical kernel ABI (the default) every pointee is the dense logical tensor, so rows are
    never padded (``multiple`` 1). Under the version-1 resident ABI (``kernel_abi.pointee_layout``) rows
    are zero-padded to the device's tile edge, derived from its RTL facts the same way the capsule shim
    derives it (:func:`merlin.compiler.lower.device.shim.tile_edge_for`); ``multiple`` is then None when the
    edge cannot be derived -- a caller must refuse to hand a kernel a buffer whose rows would need
    padding, never hand it the dense one."""
    from merlin.compiler.lower.device import shim as device_shim

    abi = device_shim.kernel_abi_for(target)
    if abi is not None and getattr(abi, "version", 1) == 2:
        return {
            "layout": abi.pointee_layout,
            "multiple": 1,
            "source": "logical_kernel_abi.pointee (mlir_oot_backend_contract.yaml): dense, never padded",
        }
    edge = device_shim.tile_edge_for(target)
    return {
        "layout": abi.pointee_layout if abi is not None else None,
        "multiple": int(edge) if edge else None,
        "source": "kernel_abi.pointee_layout (legacy/kernel_abi_v1.yaml), tile edge from the RTL facts",
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
    region answered as one kernel (:mod:`merlin.compiler.lower.device.region`) records the identical
    artifact path -- it is the SAME reply -- so without this a region's N members would each ask the
    (possibly disk-cached) compile step, N times, in the same build. Keyed by artifact TEXT,
    thread-safely; whichever member asks first still goes through the persistent on-disk cache below,
    every later member in this build just reads that first member's already-compiled object.
    """
    from merlin.compiler.lower.model import toolchain
    from merlin.development.targetgen.execution.runner.compile import llvm_mlir_to_object
    from merlin.target.contract.interfaces.abi import for_target
    from merlin.verification.lean.emission_custody import (
        artifact_cache_key,
        is_emitted_artifact,
        read_emitted_artifact,
        requires_fresh_object,
    )

    entry = for_target(target).entry_symbol
    cache_root = _object_cache_root()
    fingerprint = None
    if cache_root is not None:
        try:
            fingerprint = _object_cache_fingerprint(target)
        except Exception:  # noqa: BLE001 -- an unfingerprintable toolchain disables the cache, not the build
            cache_root = None
    compiled_by_artifact: dict[object, tuple[Path, str]] = {}
    # Untraced bytes share their original lock/cache. Traced origins stay distinct;
    # repeated reads of one observed origin share only that origin's completed object.
    text_locks: dict[object, threading.Lock] = {}
    table_lock = threading.Lock()

    def compiled_object(artifact_text: str, work: Path) -> tuple[Path, str]:
        """(compiled ``.o`` path, ``object_cache`` state) for this original owner -- ``"hit"`` off the
        persistent disk cache, ``"shared"`` off this build's own in-memory memo (another group with the
        same artifact already compiled it), or ``"miss"`` (compiled here, and disk-cached for later)."""
        owner_key = artifact_cache_key(artifact_text)
        with table_lock:
            lock = text_locks.setdefault(owner_key, threading.Lock())
        with lock:
            fresh = requires_fresh_object(artifact_text)
            cached = None if fresh else compiled_by_artifact.get(owner_key)
            if cached is not None:
                return cached[0], "shared"
            cached_path = None
            if not fresh and cache_root is not None and fingerprint is not None:
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
            compiled_by_artifact[owner_key] = (compiled, state)
            return compiled, state

    def one(row: dict[str, Any]) -> None:
        group = int(row["group"])
        work = out / f"g{group}"
        symbol = f"{entry}_g{group}"
        fresh = is_emitted_artifact(row["artifact"])
        try:
            artifact_text = read_emitted_artifact(row["artifact"])
            fresh = fresh or requires_fresh_object(artifact_text)
            compiled, object_cache = compiled_object(artifact_text, work)
            renamed = out / f"g{group}.o"
            from merlin.compiler.lower.device.build import rename_object_command

            subprocess.run(
                rename_object_command(toolchain.objcopy(), entry, symbol, compiled, renamed),
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
            if fresh:
                raise WholeModelBuildError("original traced object production did not complete") from error
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
    from contextvars import copy_context

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(copy_context().run, one, row) for row in answered]
        for future in futures:
            future.result()
    if cache_root is not None:
        with contextlib.suppress(Exception):  # noqa: BLE001 -- pruning is an optimization, never a build dependency
            prune_object_cache(cache_root)
    # WHAT THE DEDUPE BOUGHT: a repeated model block emits byte-identical artifacts, compiled once each.
    return {
        "unique_signatures": len({row["compiled_signature"] for row in answered if "compiled_signature" in row}),
        "total_groups": len(answered),
    }


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
    the same block :func:`merlin.development.targetgen.corpus.group_entries.group_binding` takes -- so a group a
    build asks a package for and the capsule that certifies it are the same bytes. ``descriptor`` is
    the target descriptor the experiment loads; without it only the target name is carried (the
    recipe must then declare its oracle tiers, which is the only thing a descriptor adds). Nothing is
    discovered: a missing recipe is a refusal, not a default.
    """
    import types

    import yaml

    from merlin.compiler.lower.device import whole_program as WP

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
        from merlin.target.registry.descriptor_loading import load_target_experiment

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


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _apply_package_passes(
    capsule: ModelCapsule, *, target: str, package_dir: str | Path, out: Path, timeout: int
) -> dict[str, Any]:
    """Run ``package_dir``'s own declared whole-model passes over ``capsule``'s interface, verify the
    result computes the same function, and say what happened either way. Never raises: a package that
    declares no passes, or one whose manifest cannot even name them correctly, is recorded and this
    build proceeds on the capsule's OWN interface exactly as it would have without this existing.
    """
    from merlin.analysis.whole_model.build import inputs as WMPass

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
    from merlin.development.targetgen.packaging import loading as OR

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


BUILD_STAGES = _trace.declare(
    "whole-model",
    ("passes", "statement", "group_objects", "extract", "render_kernels", "program_build", "memory_map", "oracle_join"),
    entry="merlin.analysis.whole_model.build.whole_model_build.build",
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
    from merlin.target.rtl import facts as F

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


__all__ = [
    "LocalReference",
    "MEMORY_MAP_SCHEMA",
    "ModelCapsule",
    "SCHEMA",
    "WholeModelBuildError",
    "bind_groups",
    "build",
    "decline_ops",
    "grade",
    "grade_memory",
    "load_model_capsule",
    "machine_header",
    "memory_map",
    "prune_object_cache",
    "refuse_if_every_group_declined",
    "require_datapath_facts",
    "state",
]
