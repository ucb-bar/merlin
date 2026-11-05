"""Post-run freezing for a lean author already loaded before drain support.

This reads existing receipts; it does not recreate issued compiler capabilities
or grant qualification. The operator must first reap the author and its jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from . import lean_authoring as L
from .contracts import StageGateError, document_sha256, write_json
from .retained_revisions import source_inventory


def _read(path):
    path = Path(path)
    if path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
        raise StageGateError("lean finalization input must be a canonical real file")
    if path.stat().st_size > 16 << 20:
        raise StageGateError("lean finalization input exceeds its bounded document size")
    data = path.read_bytes()
    document = json.loads(data)
    if type(document) is not dict:
        raise StageGateError("lean finalization requires bounded mapping documents")
    return document, dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), n_bytes=len(data))


def finalize_lean_campaign(config_path, output, *, execution_closed=False, selected_revision=None):
    """Freeze exact persisted source after an operator-confirmed process shutdown."""
    if execution_closed is not True:
        raise StageGateError("the operator must confirm the author and evaluation processes have ended")
    config, config_pin = _read(config_path)
    if config.get("schema") != L.PROTOCOL:
        raise StageGateError("lean finalization requires the explicit lean campaign configuration")
    root, output = Path(config["output"]), Path(output)
    author, author_pin = _read(root / "author" / "result.json")
    if author.get("schema") != L.PROTOCOL:
        raise StageGateError("lean author result protocol changed")
    members = {row["name"]: row for row in config["members"]}
    retained = root / "author" / "retained"
    records, pins = {}, [config_pin, author_pin]
    leaves = sorted(retained.glob("*.json"))
    if not 1 <= len(leaves) <= 224:
        raise StageGateError("lean retained revision roster exceeds its declared budget")
    for leaf in leaves:
        record, pin = _read(leaf)
        if record.get("schema") != "merlin.retained_revision.v1" or document_sha256(record) != leaf.stem:
            raise StageGateError("lean retained revision declaration changed")
        if (not isinstance(record.get("source_sha256"), str) or len(record["source_sha256"]) != 64
                or set(record["source_sha256"]) - set("0123456789abcdef")):
            raise StageGateError("lean retained source identity is invalid")
        if leaf.stat().st_mode & 0o222:
            raise StageGateError("lean retained revision declaration became writable")
        records[leaf.stem] = record
        pins.append(pin)
    controls = [key for key, row in records.items() if row.get("parent_revision") is None
                and row.get("options") == {"hypothesis": "immutable original control"}]
    if len(controls) != 1:
        raise StageGateError("lean finalization requires one original retained control")
    control = controls[0]
    jobs = author.get("final_status", {}).get("jobs")
    if type(jobs) is not list or len(jobs) > config.get("max_jobs", 128):
        raise StageGateError("lean finalization lacks the persisted bounded job roster")
    public, observations, failed, seen = [], {}, set(), set()
    for job in jobs:
        identity, revision = job.get("job_id"), job.get("revision_id")
        if (not isinstance(identity, str) or Path(identity).name != identity or identity in {".", ".."}
                or identity in seen or revision not in records):
            raise StageGateError("lean persisted job identity is invalid")
        seen.add(identity)
        record = records[revision]
        names = job.get("members")
        if (type(names) is not list or not names or len(set(names)) != len(names)
                or not set(names) <= members.keys() or job.get("source_sha256") != record["source_sha256"]):
            raise StageGateError("lean persisted job source or member roster changed")
        directory = root / "jobs" / identity
        result_path = next((directory / name for name in ("worker-result.json", "private-result.json")
                            if (directory / name).exists()), None)
        view = dict(job_id=identity, revision_id=revision, source_sha256=record["source_sha256"], members=names)
        if result_path is None:
            public.append(dict(**view, state="PENDING", **L.DISCLAIMER))
            continue
        result, pin = _read(result_path)
        pins.append(pin)
        if result_path.name == "worker-result.json":
            request, request_pin = _read(directory / "worker-request.json")
            evaluation = request.get("evaluation", {})
            baseline_source = retained / "sources" / records[control]["source_sha256"]
            if (evaluation.get("candidate_source") != str(retained / "sources" / record["source_sha256"])
                    or evaluation.get("baseline_source") != str(baseline_source)
                    or [row.get("id", row.get("name")) for row in evaluation.get("selected_members", [])] != names):
                raise StageGateError("lean worker source or selected-member binding changed")
            pins.append(request_pin)
        rows = result.get("members")
        if type(rows) is not list or [row.get("id") for row in rows] != names:
            public.append(dict(**view, state="refused", error_type=result.get("error_type", "invalid_result"),
                               **L.DISCLAIMER))
            failed.add(revision)
            continue
        projected = []
        for row in rows:
            name = row["id"]
            arms = {arm: L._public_arm(row.get(arm), members[name].get("compute_floor_cycles"),
                                      config["frozen_model_sha256"]) for arm in ("baseline", "candidate")}
            projected.append(dict(id=name, **arms))
            if arms["candidate"]["complete_outputs_checked"]:
                observations.setdefault(revision, {}).setdefault(name, []).append(
                    arms["candidate"]["attainment_fraction"])
            else:
                failed.add(revision)
        public.append(dict(**view, state="completed", result=dict(members=projected, **L.DISCLAIMER),
                           **L.DISCLAIMER))
    successful = {key: rows for key, rows in observations.items() if key not in failed}
    target = config.get("target_fraction", .3)
    targets = [key for key, rows in successful.items() if set(rows) == set(members)
               and all(values and all(v is not None and v >= target for v in values) for values in rows.values())]
    reason = "partial exploratory selection; unmeasured obligations remain pending"
    if selected_revision is not None:
        if selected_revision not in records or selected_revision in failed:
            raise StageGateError("explicit lean selection is unavailable or numerically refused")
        selected, reason = selected_revision, "explicit operator selection; qualification remains deferred"
    elif targets:
        selected, reason = sorted(targets)[0], "estimated_structural_target_met; qualification remains deferred"
    elif successful:
        def score(key):
            rows = successful[key]
            estimates = [min(values) for values in rows.values() if values and all(v is not None for v in values)]
            return len(rows) == len(members), len(estimates), min(estimates, default=0), key
        selected = max(successful, key=score)
    else:
        selected, reason = control, "no completed candidate evidence; original control retained"
    record = records[selected]
    source = retained / "sources" / record["source_sha256"]
    inventory = source_inventory(source, max_bytes=32 << 20, max_files=8192, readonly=True)
    if (inventory["sha256"] != record["source_sha256"]
            or document_sha256(inventory) != record["source_inventory_sha256"]):
        raise StageGateError("selected retained compiler source inventory changed")
    output.mkdir()
    frozen = output / "frozen-compiler"
    shutil.copytree(source, frozen)
    if source_inventory(frozen, max_bytes=32 << 20, max_files=8192, readonly=True) != inventory:
        raise StageGateError("post-run frozen compiler differs from the selected retained source")
    final = dict(schema="merlin.phase2.lean_finalization.v1", execution_closed=True,
                 source_validation="exact_retained_record_and_inventory; no issued capabilities reconstructed",
                 selected_revision=dict(revision_id=selected, **record), frozen_compiler=str(frozen),
                 selection_reason=reason, estimated_structural_target_met=selected in targets,
                 jobs=public, inputs=pins, **L.DISCLAIMER)
    write_json(output / "result.json", final)
    (output / "result.json").chmod(0o444)
    output.chmod(0o555)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-revision")
    parser.add_argument("--execution-closed", action="store_true", required=True)
    args = parser.parse_args()
    finalize_lean_campaign(args.config, args.output, execution_closed=args.execution_closed,
                           selected_revision=args.selected_revision)


if __name__ == "__main__":
    main()
