"""Stream verified Phase 1 archives; prepare, but never launch, Spike observations."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import statistics
import tarfile
from pathlib import Path, PurePosixPath

from . import contracts as C
from . import fitting as F
from . import validation as V

ARCHIVE = "merlin.phase1.capsule_timing_archive.v2"
BINDING = "merlin.phase1.spike_archive_binding.v1"
PLAN = "merlin.phase1.spike_observation_plan.v1"
MAX_ARCHIVE_BYTES = 16 * 1024**3
MAX_MEMBER_BYTES = 1024**3
MAX_MEMBERS = 100000
MAX_JSON_BYTES = 16 * 1024**2
CHUNK = 1024**2
MAX_METADATA_BYTES = 64 * 1024


class _BoundedTarInfo(tarfile.TarInfo):
    """Check extended headers before tarfile allocates their metadata payloads."""

    def _proc_member(self, archive):
        metadata = self.type in {tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK}
        depth = getattr(archive, "_timing_metadata_depth", 0)
        if metadata and (self.size > MAX_METADATA_BYTES or depth >= 16):
            raise C.ContractError("archive extended metadata exceeds its byte/depth bound")
        if metadata:
            total = getattr(archive, "_timing_metadata_bytes", 0) + self.size
            if total > MAX_JSON_BYTES:
                raise C.ContractError("archive cumulative extended metadata exceeds its byte bound")
            archive._timing_metadata_bytes = total
        if self.type == tarfile.GNUTYPE_SPARSE:
            raise C.ContractError("sparse archive members refuse")
        archive._timing_metadata_depth = depth + int(metadata)
        try:
            return super()._proc_member(archive)
        finally:
            archive._timing_metadata_depth = depth

    def _proc_gnusparse_00(self, *args, **kwargs):
        raise C.ContractError("sparse archive members refuse")

    _proc_gnusparse_01 = _proc_gnusparse_00
    _proc_gnusparse_10 = _proc_gnusparse_00


def _json(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise C.ContractError("duplicate archive JSON key")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise C.ContractError("archive JSON must be an object")
    # Reject NaN/Infinity and excessive nesting rather than canonicalizing them.
    C.digest(value)
    return value


def _path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 4096
        or "\\" in value
        or any(ord(char) < 32 for char in value)
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise C.ContractError("archive paths must be canonical relative POSIX paths")
    return value


def _artifacts(row: dict) -> list[dict]:
    value = row.get("artifacts")
    entries = [entry for entry in value.values() if entry is not None] if isinstance(value, dict) else value
    if not isinstance(entries, list) or len(entries) > 128:
        raise C.ContractError("archive row requires a bounded artifact roster")
    for entry in entries:
        if not isinstance(entry, dict) or not {"path", "sha256", "bytes"} <= set(entry):
            raise C.ContractError("indexed artifacts require path, sha256 and bytes")
        _path(entry["path"])
        C.sha(entry["sha256"], "indexed artifact")
        if type(entry["bytes"]) is not int or not 0 <= entry["bytes"] <= MAX_MEMBER_BYTES:
            raise C.ContractError("indexed artifact byte count exceeds its bound")
    return entries


def _positive_cycles(row: dict) -> bool:
    return type(row.get("cycles")) is int and row["cycles"] > 0 and row.get("cycles_measured") is True


def _passing(row: dict) -> bool:
    return row.get("engine") == "gsim" and row.get("status") == "pass"


def _complete(row: dict) -> bool:
    artifacts = row.get("artifacts") or {}
    return (
        _passing(row)
        and row.get("result_status") == "pass"
        and _positive_cycles(row)
        and row.get("derived_from_rtl") is True
        and row.get("cycle_accurate") is True
        and bool(row.get("cycle_scope"))
        and isinstance(row.get("timing_selection"), dict)
        and bool(row["timing_selection"])
        and isinstance(row.get("sim_provenance"), dict)
        and bool(row["sim_provenance"].get("sha256"))
        and bool(row.get("elf"))
        and row.get("console_cycle_verified") is True
        and isinstance(artifacts, dict)
        and all(
            artifacts.get(name)
            for name in ("harness", "input_interface", "command_buffer", "static_instruction_trace", "numeric_report")
        )
    )


def _caller_selection(raw: bytes) -> tuple[str, dict]:
    """Parse the producer's exact marker structurally, without a target grammar."""
    text = raw.decode("utf-8")
    prefix = "/* merlin.kernel_call_timing.v1 selection_sha256="
    if text.count(prefix) != 1:
        raise C.ContractError("original caller requires exactly one kernel timing marker")
    rest = text.split(prefix, 1)[1]
    digest, separator, rest = rest.partition(" boundary=entry-through-completion record_b64=")
    encoded, ending, _ = rest.partition(" */")
    if not separator or not ending:
        raise C.ContractError("original caller timing boundary marker is malformed")
    C.sha(digest, "original caller selection")
    try:
        selection = _json(base64.b64decode(encoded, validate=True))
    except (ValueError, UnicodeError) as exc:
        raise C.ContractError("original caller timing record is malformed") from exc
    counter = selection.get("counter_semantics")
    if (
        selection.get("schema") != "merlin.kernel_call_timing.v1"
        or selection.get("selection_sha256") != digest
        or selection.get("boundary") != "immediately before kernel entry through explicitly selected completion"
        or not isinstance(counter, dict)
        or counter.get("unit") != "cycles"
        or type(counter.get("width_bits")) is not int
        or counter["width_bits"] != 64
        or counter.get("wrap") != "modulo"
        or not counter.get("source_pins")
    ):
        raise C.ContractError("original caller lacks declared pinned cycle/clock semantics")
    return digest, selection


def _verify_contexts(path: str | Path, wanted: dict[str, list[tuple[str, dict]]], files: dict) -> None:
    """Re-read only bounded caller/console/result context, never extract the tar."""
    pending = set(wanted)
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|", tarinfo=_BoundedTarInfo) as archive:
            for member in archive:
                if member.name in pending:
                    if not member.isreg() or not 0 <= member.size <= MAX_JSON_BYTES:
                        raise C.ContractError("original caller/console/result context exceeds its bound")
                    raw = archive.extractfile(member).read(MAX_JSON_BYTES + 1)
                    if (
                        len(raw) != files[member.name]["bytes"]
                        or hashlib.sha256(raw).hexdigest() != files[member.name]["sha256"]
                    ):
                        raise C.ContractError("archive context changed between verification passes")
                    for kind, row in wanted[member.name]:
                        if kind == "harness":
                            digest, selection = _caller_selection(raw)
                            if (digest, selection) != (row["timing_selection_sha256"], row["timing_selection"]):
                                raise C.ContractError("indexed selection differs from its original caller marker")
                        elif kind == "console":
                            metrics = []
                            for line in raw.decode("utf-8").splitlines():
                                value = line.strip().removeprefix("METRIC cycles ")
                                if (
                                    line.strip() != value
                                    and value
                                    and value[0] != "0"
                                    and all("0" <= char <= "9" for char in value)
                                ):
                                    metrics.append(int(value))
                            if metrics != [row["cycles"]]:
                                raise C.ContractError(
                                    "original RTL console does not contain exactly one matching positive cycle marker"
                                )
                        else:
                            result = _json(raw)
                            tier = (result.get("tiers") or {}).get("L3") or {}
                            comparisons = {"status": row["result_status"], "capsule": row["capsule"]}
                            if any(result.get(key) != value for key, value in comparisons.items()) or any(
                                tier.get(key) != row.get(key)
                                for key in (
                                    "status",
                                    "engine",
                                    "cycles",
                                    "derived_from_rtl",
                                    "cycle_accurate",
                                    "sim_provenance",
                                    "timing",
                                )
                            ):
                                raise C.ContractError(
                                    "indexed verdict or cycles differ from original capsule_result.json"
                                )
                            compiler = result.get("original_compiler_child") or {}
                            buffer = result.get("command_buffer_artifact") or {}
                            tensors = (buffer.get("command_buffer") or {}).get("tensors") or {}
                            shape = {
                                name: {"dtype": spec.get("dtype"), "shape": spec.get("shape")}
                                for name, spec in sorted(tensors.items())
                            }
                            if (compiler.get("candidate_sha256"), buffer.get("artifact_sha256"), shape) != (
                                row["candidate_sha256"],
                                row["command_buffer_sha256"],
                                row["shape"],
                            ):
                                raise C.ContractError(
                                    "index candidate, command buffer or shape differs from original result"
                                )
                    pending.remove(member.name)
                archive.members.clear()
    if pending:
        raise C.ContractError("original context is missing from archive")


def inspect_archive(path: str | Path, expected_sha256: str) -> dict:
    """Hash a tar in bounded chunks, inspect without extraction, verify indexed files.

    The expected archive digest must come from the immutable Phase 1 handoff.
    Byte integrity does not establish source visibility or simulator provenance.
    """
    C.sha(expected_sha256, "expected archive")
    files, directories, index = {}, set(), None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_ARCHIVE_BYTES:
            raise C.ContractError("archive must be an ordinary bounded uncompressed tar")
        checksum = hashlib.sha256()
        while block := stream.read(CHUNK):
            checksum.update(block)
        if checksum.hexdigest() != expected_sha256:
            raise C.ContractError("archive bytes differ from the immutable handoff digest")
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode="r|", tarinfo=_BoundedTarInfo) as archive:
            count, total = 0, 0
            for member in archive:
                count += 1
                if count > MAX_MEMBERS:
                    raise C.ContractError("archive member count exceeds its bound")
                name = _path(member.name.rstrip("/") if member.isdir() else member.name)
                if name in files or name in directories:
                    raise C.ContractError("duplicate archive member")
                if name != "index.json" and name != "run" and not name.startswith("run/"):
                    raise C.ContractError("unexpected archive member outside run/")
                if member.isdir():
                    directories.add(name)
                else:
                    if not member.isreg() or member.issparse() or not 0 <= member.size <= MAX_MEMBER_BYTES:
                        raise C.ContractError("archive links, devices, sparse or oversized members refuse")
                    if name == "run" or name == "index.json" and member.size > MAX_JSON_BYTES:
                        raise C.ContractError("invalid archive index or root member")
                    total += member.size
                    if total > MAX_ARCHIVE_BYTES:
                        raise C.ContractError("archive payload exceeds its byte bound")
                    source = archive.extractfile(member)
                    digest, size, payload = hashlib.sha256(), 0, bytearray()
                    while block := source.read(CHUNK):
                        size += len(block)
                        digest.update(block)
                        if name == "index.json":
                            payload.extend(block)
                    if size != member.size:
                        raise C.ContractError("truncated archive member")
                    files[name] = {"sha256": digest.hexdigest(), "bytes": size}
                    if name == "index.json":
                        index = _json(payload)
                # Streaming TarFile otherwise retains every TarInfo object.
                archive.members.clear()
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise C.ContractError("archive changed during verification")
    if index is None or index.get("schema") != ARCHIVE:
        raise C.ContractError("expected a Phase 1 capsule timing v2 index")
    records = index.get("records")
    if not isinstance(records, list) or len(records) > C.MAX_RECORDS:
        raise C.ContractError("archive record count exceeds its bound")
    seen, verified, results, contexts = set(), set(), [], {}
    for row in records:
        if not isinstance(row, dict):
            raise C.ContractError("archive records must be objects")
        result = _path(row.get("result"))
        if result in seen or PurePosixPath(result).name != "capsule_result.json":
            raise C.ContractError("archive result paths must be unique capsule_result.json files")
        seen.add(result)
        if "run/" + result not in files:
            raise C.ContractError("indexed capsule result is absent")
        entries = _artifacts(row)
        local = set()
        for entry in entries:
            name = "run/" + entry["path"]
            if files.get(name) != {"sha256": entry["sha256"], "bytes": entry["bytes"]}:
                raise C.ContractError("indexed artifact is missing or its bytes differ")
            verified.add(name)
            local.add(name)
        if row.get("elf") is not None:
            elf = "run/" + _path(row["elf"])
            if elf not in local or files[elf]["sha256"] != C.sha(row.get("elf_sha256"), "original ELF"):
                raise C.ContractError("original ELF is not verified by the artifact roster")
            if isinstance(row["artifacts"], dict) and row["artifacts"].get("elf", {}).get("path") != row["elf"]:
                raise C.ContractError("original ELF differs from its role in this row")
        if _complete(row):
            if row.get("cycle_scope") != "entry-through-completion":
                raise C.ContractError("archive label is outside the kernel-call boundary")
            if row["timing_selection"].get("selection_sha256") != C.sha(
                row.get("timing_selection_sha256"), "selection"
            ):
                raise C.ContractError("counter selection differs from its declared selection identity")
            harness = row["artifacts"]["harness"]
            if PurePosixPath(harness["path"]).name != "harness.c" or harness["sha256"] != row.get("harness_sha256"):
                raise C.ContractError("complete timing row lacks its exact original caller")
            for field in ("candidate_sha256", "command_buffer_sha256", "harness_sha256"):
                C.sha(row.get(field), field)
            C.sha(row["sim_provenance"]["sha256"], "original simulator binary")
            for kind, artifact_name in (("harness", "harness"), ("console", "rtl_console")):
                reference = row["artifacts"].get(artifact_name)
                if not reference:
                    raise C.ContractError("complete timing row lacks original caller/console context")
                contexts.setdefault("run/" + reference["path"], []).append((kind, row))
            contexts.setdefault("run/" + result, []).append(("result", row))
        results.append(
            {
                "record_id": C.digest([expected_sha256, result]),
                "row": row,
                "result_sha256": files["run/" + result]["sha256"],
            }
        )
    counts = {
        "capsule_records": len(records),
        "gsim_pass_records": sum(map(_passing, records)),
        "gsim_pass_with_cycles": sum(_passing(row) and _positive_cycles(row) for row in records),
        "gsim_pass_complete": sum(map(_complete, records)),
    }
    if any(type(index.get(key)) is not int or index[key] != value for key, value in counts.items()):
        raise C.ContractError("archive summary counts disagree with its original record roster")
    ready = counts["gsim_pass_records"] > 0 and counts["gsim_pass_records"] == counts["gsim_pass_complete"]
    if type(index.get("timing_ready")) is not bool or index["timing_ready"] != ready:
        raise C.ContractError("archive timing_ready disagrees with its record roster")
    _verify_contexts(path, contexts, files)
    return {
        "schema": "merlin.phase1.spike_archive_inventory.v1",
        "archive_sha256": expected_sha256,
        "index_sha256": files["index.json"]["sha256"],
        "timing_ready": ready,
        **counts,
        "records": results,
        "members": len(files),
        "indexed_artifacts_verified": len(verified),
        "prediction_scope": "capsule kernel entry through completion wait only",
        "cache_warmth": "unknown",
        "static_trace_is_dynamic_timeline": False,
    }


def prepare(inventory: dict, binding: dict) -> dict:
    """Join externally reviewed sources/counter scopes; leave absent observations absent."""
    C.closed(binding, {"schema", "archive_sha256", "domain", "feature_names", "review", "cases"}, "archive binding")
    if binding["schema"] != BINDING or binding["archive_sha256"] != inventory["archive_sha256"]:
        raise C.ContractError("archive binding differs from the exact immutable archive")
    if inventory["timing_ready"] is not True:
        raise C.ContractError("incomplete timing archive cannot prepare calibration observations")
    C.domain(binding["domain"])
    if binding["domain"]["label_engine"] != "gsim":
        raise C.ContractError("Phase 1 archive labels require their original gSIM selection")
    review = C.closed(
        binding["review"],
        {"timing_selection", "cycle_scope", "sim_provenance_sha256", "counter", "cache_protocol"},
        "reviewed counter/simulator selection",
    )
    counter = C.closed(
        review["counter"],
        {"unit", "width_bits", "wrap", "clock", "max_delta_cycles", "sources"},
        "reviewed counter semantics",
    )
    if (counter["unit"], counter["width_bits"], counter["wrap"]) != ("cycles", 64, "modulo") or type(
        counter["width_bits"]
    ) is not int:
        raise C.ContractError("kernel-call labels require 64-bit modulo cycle semantics")
    C.text(counter["clock"], "reviewed clock")
    maximum = counter["max_delta_cycles"]
    if type(maximum) is not int or not 0 < maximum < 2**63:
        raise C.ContractError("explicit maximum counter delta must be positive and below 2^63")
    sources = counter["sources"]
    if not isinstance(sources, dict) or set(sources) != {"accessor", "isa", "clock"}:
        raise C.ContractError("counter review requires original accessor/ISA/clock source digests")
    for role, digest in sources.items():
        C.sha(digest, "counter source " + role)
    selected = review["timing_selection"]
    if not isinstance(selected, dict) or not isinstance(selected.get("counter_semantics"), dict):
        raise C.ContractError("original timing selection must include the reviewed counter semantics")
    semantics = selected["counter_semantics"]
    selected_clock = semantics.get("clock", semantics.get("clock_name"))
    if isinstance(selected_clock, dict):
        selected_clock = selected_clock.get("name")
    if (
        any(counter[key] != semantics.get(key) for key in ("unit", "width_bits", "wrap"))
        or selected_clock != counter["clock"]
    ):
        raise C.ContractError("review counter/clock contradicts the original timing selection")
    if type(selected.get("max_delta_cycles")) is not int or maximum != selected["max_delta_cycles"]:
        raise C.ContractError("review maximum delta differs from the original timing selection")

    def digests(value):
        if isinstance(value, str):
            return {value}
        if isinstance(value, dict):
            return set().union(*(digests(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(digests(item) for item in value))
        return set()

    if not set(sources.values()) <= digests(semantics.get("source_pins")):
        raise C.ContractError("review source pins are absent from original counter semantics")
    C.sha(review["sim_provenance_sha256"], "reviewed simulator selection")
    selection_sha = C.sha(review["timing_selection"].get("selection_sha256"), "reviewed selection")
    scope = {
        "schema": "merlin.kernel_call_scope.v1",
        "cycle_scope": review["cycle_scope"],
        "timing_selection_sha256": selection_sha,
        "selection_record_sha256": C.digest(review["timing_selection"]),
    }
    if not review["cycle_scope"] or C.digest(scope) != binding["domain"]["scope_sha256"]:
        raise C.ContractError("domain scope differs from the reviewed whole kernel-call label")
    warmth = {"cache_warmth": "unknown", "protocol": review["cache_protocol"]}
    if review["cache_protocol"] is not None:
        C.closed(review["cache_protocol"], {"state", "receipt_sha256"}, "separate cache protocol")
        C.text(review["cache_protocol"]["state"], "cache protocol state")
        C.sha(review["cache_protocol"]["receipt_sha256"], "cache protocol receipt")
        warmth["cache_warmth"] = review["cache_protocol"]["state"]
    if C.digest(warmth) != binding["domain"]["memory_regime_sha256"]:
        raise C.ContractError("memory regime must bind unknown warmth or a separate explicit protocol")
    by_result = {item["row"]["result"]: item for item in inventory["records"]}
    cases, observations, seen = [], [], set()
    if not isinstance(binding["cases"], list) or len(binding["cases"]) > C.MAX_CASES:
        raise C.ContractError("archive case binding exceeds its count bound")
    for item in binding["cases"]:
        C.closed(item, {"case", "results", "candidate_sha256", "shape_sha256", "counter_scope"}, "archive case binding")
        case = C.closed(item["case"], C.CASE_KEYS, "archive case")
        C.sha(item["candidate_sha256"], "candidate identity")
        C.sha(item["shape_sha256"], "logical shape identity")
        pcs = C.closed(
            item["counter_scope"], {"begin_pc", "end_pc", "occurrence", "evidence_sha256"}, "counter-PC review"
        )
        C.sha(pcs["evidence_sha256"], "original caller boundary review")
        if any(type(pcs[key]) is not int or not 0 < pcs[key] < 2**64 or pcs[key] % 2 for key in ("begin_pc", "end_pc")):
            raise C.ContractError("explicit aligned counter PCs are required")
        if pcs["begin_pc"] == pcs["end_pc"] or type(pcs["occurrence"]) is not int or not 0 <= pcs["occurrence"] < 65536:
            raise C.ContractError("distinct counter PCs and a bounded occurrence are required")
        results = item["results"]
        if not isinstance(results, list) or not results:
            raise C.ContractError("case binding must retain its original execution records")
        cases.append(case)
        for result in results:
            if not isinstance(result, str) or result not in by_result or result in seen:
                raise C.ContractError("bound results must exist and belong to exactly one case")
            seen.add(result)
            original = by_result[result]
            row = original["row"]
            if (row.get("elf_sha256"), row.get("candidate_sha256"), C.digest(row.get("shape"))) != (
                case["elf_sha256"],
                item["candidate_sha256"],
                item["shape_sha256"],
            ):
                raise C.ContractError("case ELF/candidate/logical shape differs from its original record")
            # This deliberately binds the complete caller input context. It does
            # not claim to hash just tensor bytes or infer an input from shape.
            input_sha = C.digest(
                {"schema": "merlin.embedded_input_context.v1", "harness_sha256": row.get("harness_sha256")}
            )
            if case["input_sha256"] != input_sha:
                raise C.ContractError("embedded input identity differs from the original caller context")
            eligible = _complete(row)
            if eligible and (
                row["timing_selection"] != review["timing_selection"]
                or row["cycle_scope"] != review["cycle_scope"]
                or C.digest(row["sim_provenance"]) != review["sim_provenance_sha256"]
                or row["cycles"] > maximum
            ):
                raise C.ContractError("label scope, clock selection, simulator or counter bound changed")
            if eligible and row["sim_provenance"]["sha256"] != binding["domain"]["label_engine_sha256"]:
                raise C.ContractError("domain label engine differs from original simulator binary")
            scope_binding = {
                "schema": "merlin.spike_scope_binding.v1",
                "elf_sha256": case["elf_sha256"],
                "input_sha256": case["input_sha256"],
                "scope_sha256": case["scope_sha256"],
                **pcs,
            }
            observations.append(
                {
                    "record_id": original["record_id"],
                    "case_id": case["case_id"],
                    "result": result,
                    "candidate_sha256": item["candidate_sha256"],
                    "shape_sha256": item["shape_sha256"],
                    "shape": row["shape"],
                    "elf_member": "run/" + row["elf"],
                    "scope_binding": scope_binding,
                    "label": {
                        "engine": "gsim",
                        "engine_sha256": binding["domain"]["label_engine_sha256"],
                        "receipt_sha256": C.digest(
                            {
                                "archive_sha256": inventory["archive_sha256"],
                                "result_sha256": original["result_sha256"],
                                "row": row,
                            }
                        ),
                        "execution_sha256": C.execution_identity(case),
                        "status": "pass" if eligible else "unknown",
                        "numeric": "pass" if eligible else "unknown",
                        "measured_now": eligible,
                        "cycles": row["cycles"] if eligible else None,
                    },
                }
            )
    # A known failed repeat cannot be removed to improve its execution's label.
    bound_elfs = {case["elf_sha256"] for case in cases}
    if any(
        row["row"]["result"] not in seen and (_passing(row["row"]) or row["row"].get("elf_sha256") in bound_elfs)
        for row in inventory["records"]
    ):
        raise C.ContractError("binding omitted a passing case or a repeat of a bound ELF")
    empty = {
        "schema": C.DATASET,
        "profile": "phase1-public",
        "domain": binding["domain"],
        "feature_names": binding["feature_names"],
        "cases": cases,
        "records": [],
    }
    C.dataset(empty)
    return {
        "schema": PLAN,
        "archive_sha256": inventory["archive_sha256"],
        "binding_sha256": C.digest(binding),
        "dataset_template": empty,
        "observations": observations,
        "review": review,
        "launch_authority": False,
        "launches": [],
        "cache_warmth": warmth["cache_warmth"],
        "scope": "capsule kernel entry through completion wait; no full-model or per-command labels",
        "static_trace_is_dynamic_timeline": False,
    }


def attach(plan: dict, reports: dict) -> dict:
    """Attach original reports to a verified plan; missing cases retain denominator."""
    C.closed(reports, {"schema", "plan_sha256", "reports"}, "original observation attachment")
    if (
        plan.get("schema") != PLAN
        or reports["schema"] != "merlin.phase1.spike_observations.v1"
        or reports["plan_sha256"] != C.digest(plan)
    ):
        raise C.ContractError("observation attachment differs from its exact frozen plan")
    if not isinstance(reports["reports"], dict):
        raise C.ContractError("original reports must be keyed by declared record identity")
    data = dict(plan["dataset_template"], records=[])
    C.dataset(data)
    by_id = {row["record_id"]: row for row in plan["observations"]}
    if set(reports["reports"]) - set(by_id):
        raise C.ContractError("observation attachment contains an undeclared record")
    cases = {case["case_id"]: case for case in data["cases"]}
    unavailable = {
        row["case_id"]
        for key, row in by_id.items()
        if key not in reports["reports"] or row["label"]["status"] != "pass"
    }
    for key, envelope in reports["reports"].items():
        original = by_id[key]
        report = envelope.get("report") if isinstance(envelope, dict) else None
        if not isinstance(report, dict) or report.get("scope_binding") != original["scope_binding"]:
            raise C.ContractError("native report does not reproduce its exact reviewed counter-PC binding")
        names = data["feature_names"]
        features = report.get("feature_counts")
        if not isinstance(features, dict) or not set(names) <= set(features):
            raise C.ContractError("original native report omits declared observed features")
        row = {
            "record_id": key,
            "case_id": original["case_id"],
            "features": {name: features[name] for name in names},
            "event_report": envelope,
            "label": original["label"],
            "feature_provenance": {
                "engine_sha256": data["domain"]["feature_engine_sha256"],
                "model_sha256": data["domain"]["target_model_sha256"],
                "trace_sha256": C.digest(report),
                "execution_sha256": original["label"]["execution_sha256"],
                "complete": report.get("timing_complete") is True,
            },
        }
        V.native_prediction(row, cases[row["case_id"]], data)
        if row["case_id"] not in unavailable:
            data["records"].append(row)
    C.dataset(data)
    return data


def file_identity(path: str | Path) -> str:
    """Hash a selected artifact in bounded chunks without copying it."""
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_ARCHIVE_BYTES:
            raise C.ContractError("selected handoff artifact must be an ordinary bounded file")
        checksum = hashlib.sha256()
        while block := stream.read(CHUNK):
            checksum.update(block)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise C.ContractError("selected handoff artifact changed while hashing")
        return checksum.hexdigest()


def handoff(plan: dict, dataset: dict, identities: dict) -> dict:
    """Freeze dataset, split, selected implementation and held-out error evidence."""
    C.closed(
        identities,
        {"spike_executable_sha256", "spike_configuration_sha256", "model_artifact_sha256", "training_code_sha256"},
        "handoff artifact identities",
    )
    for key, value in identities.items():
        C.sha(value, key)
    if identities["model_artifact_sha256"] != dataset["domain"]["target_model_sha256"]:
        raise C.ContractError("handoff model differs from the native predictions' exact model artifact")
    rebuilt = attach(
        plan,
        {
            "schema": "merlin.phase1.spike_observations.v1",
            "plan_sha256": C.digest(plan),
            "reports": {row["record_id"]: row["event_report"] for row in dataset["records"]},
        },
    )
    if rebuilt != dataset:
        raise C.ContractError("handoff dataset differs from the archive's original labels and native observations")
    report = V.validate(dataset, prediction_kind="event_timing")
    cases = {case["case_id"]: case for case in dataset["cases"]}
    contexts = {row["case_id"]: row for row in plan["observations"]}
    errors, families, shapes = [], {}, {}
    for key, point in report["predictions"].items():
        absolute = abs(point["predicted_cycles"] - point["measured_cycles"])
        error = {
            "case_id": key,
            **point,
            "absolute_cycle_error": absolute,
            "absolute_relative_cycle_error": absolute / point["measured_cycles"],
            "mechanism_family": cases[key]["mechanism_family"],
            "shape_sha256": contexts[key]["shape_sha256"],
            "shape": contexts[key]["shape"],
        }
        errors.append(error)
        families.setdefault(error["mechanism_family"], []).append(error)
        shapes.setdefault(error["shape_sha256"], []).append(error)

    def summarize(groups):
        return {
            key: {
                "cases": len(rows),
                "median_absolute_cycle_error": statistics.median(row["absolute_cycle_error"] for row in rows),
                "median_absolute_relative_cycle_error": statistics.median(
                    row["absolute_relative_cycle_error"] for row in rows
                ),
                "p95_absolute_relative_cycle_error": F.quantile(
                    [row["absolute_relative_cycle_error"] for row in rows], 0.95
                ),
            }
            for key, rows in groups.items()
        }

    return {
        "schema": "merlin.phase2.spike_timing_handoff.v1",
        "archive_sha256": plan["archive_sha256"],
        "plan_sha256": C.digest(plan),
        "dataset_sha256": C.digest(dataset),
        "artifact_identities": identities,
        "profile": dataset["profile"],
        "domain": dataset["domain"],
        "prediction_scope": plan["scope"],
        "counter_clock": plan["review"]["counter"],
        "cache_warmth": plan["cache_warmth"],
        "split_manifest": [
            {
                "case_id": case["case_id"],
                "split": case["split"],
                "source_group_sha256": case["source_group_sha256"],
                "workload_sha256": case["workload_sha256"],
                "shape_sha256": contexts[case["case_id"]]["shape_sha256"],
            }
            for case in dataset["cases"]
        ],
        "holdout_report": report,
        "held_case_errors": errors,
        "errors_by_mechanism_family": summarize(families),
        "errors_by_shape": summarize(shapes),
        "phase2_ranking_eligible": report["public_feedback_eligible"],
        "actual_rtl_timing_source": "gsim",
        "firesim_cycle_accuracy_established": False,
        "full_model_prediction_qualified": False,
        "limitations": [
            "artifact hashes record selections; external custody must establish actual executions and fitting",
            "whole-call labels do not measure individual command prices",
            "gSIM labels do not establish FireSim cycle accuracy or whole-model timing",
        ],
    }
