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
        order (:func:`merlin.analysis.whole_model.build.inputs.apply_passes`), and the result is used for the
        rest of this build ONLY if it verifies as computing the same function
        (:func:`verify_passes_preserve_semantics`) -- never trusted from the pass's own claim, and
        never applied at all for a package that declares none;
      * ``allow_regions`` -- OFF BY DEFAULT. When set, the target's driver links a FUSED REGION
        (:func:`region_facts`) AND the package's own manifest declares ``whole_model_regions: true``,
        it may be offered a legal run of consecutive groups as one kernel
        (:func:`merlin.compiler.lower.device.region.legal_regions`) in addition to each group alone. An
        answered region is ONE step of the program -- its kernel called once in its members' place,
        graded at its boundary, every member package-answered (``regions`` in the record says which);
        a package that never declares that opt-in sees no change at all, whatever this flag is (see
        :func:`merlin.compiler.lower.device.whole_program.whole_program_buffer`);
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
    from merlin.analysis.whole_model.build.stages import (
        DRIVER_REGION_REFUSED,
        OBJECT_CACHE_DISABLE_ENV,
        OBJECT_CACHE_NAMESPACE,
        ON_HOST,
        ON_PACKAGE,
        ON_VENDOR,
        REGION_UNLINKED,
        SCHEMA,
        _apply_package_passes,
        _decline_key,
        _default_out,
        _kernel_objects,
        _pins_for,
        _prohibited,
        _sha256,
        _StageClock,
        bind_groups,
        corpus_binder,
        decline_ops,
        linked_regions,
        pointee_row_padding,
        refuse_if_every_group_declined,
        region_facts,
        require_datapath_facts,
        settle_regions,
        state,
    )
    from merlin.common import provenance as PROV
    from merlin.common.artifacts import dump_yaml
    from merlin.runtime.backends import registry as backends

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


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m merlin.analysis.whole_model.build.stages`` (:mod:`.whole_model_build_cli`)."""
    from merlin.analysis.whole_model.build.whole_model_build_cli import main as _main

    return _main(argv)


if __name__ == "__main__":
    sys.exit(main())
