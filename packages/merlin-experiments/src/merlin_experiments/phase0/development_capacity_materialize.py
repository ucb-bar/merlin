"""Materialize retained capacity controls without changing their cohort or roles.

Uses the existing public source factory and blocked independent reference owner.
No capture, target compilation, execution, publication or admission is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

from merlin.common.cpu_pool import shared_compute_pool

from . import development_blocked_reference as B
from . import development_capture_intake as I
from . import development_cohort as F
from . import development_joint_capacity as J
from . import development_sources as D
from . import workload_roster as W


def _pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("capacity input must be an ordinary selected file")
    return dict(path=str(path.absolute()), **F._identity(path))


def _check_reference_inputs(path, description, *, block):
    """Bound archive expansion before loading and compare every signed seed."""
    expected = {f"{kind}_{i}.npy" for kind in ("input", "output") for i in range(len(description[kind + "s"]))}
    payload = sum(
        int(np.prod(x["shape"])) * np.dtype(x["dtype"]).itemsize
        for kind in ("inputs", "outputs")
        for x in description[kind]
    )
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        if (
            len(items) != len(expected)
            or {x.filename for x in items} != expected
            or sum(x.file_size for x in items) > payload + 4096 * len(expected)
            or path.stat().st_size > payload + 4096 * len(expected)
        ):
            raise ValueError("retained capacity reference archive exceeds its complete typed payload")
    with np.load(path, allow_pickle=False) as archive:
        for ordinal, abi in enumerate(description["inputs"]):
            value = archive[f"input_{ordinal}"]
            if value.shape != tuple(abi["shape"]) or value.dtype != np.dtype(abi["dtype"]):
                raise ValueError("retained capacity input ABI changed")
            for i in range(0, value.shape[0], block["M"]):
                rows = np.arange(i, min(i + block["M"], value.shape[0]), dtype=np.int64)
                for j in range(0, value.shape[1], block["K"]):
                    columns = np.arange(j, min(j + block["K"], value.shape[1]), dtype=np.int64)
                    if not np.array_equal(
                        value[np.ix_(rows, columns)], B.seed_matrix(value.shape, ordinal, rows, columns)
                    ):
                        raise ValueError("retained capacity input differs from its complete original seed")


def materialize(
    sources,
    *,
    manifest_sha256,
    ordinals,
    facts_path,
    facts_sha256,
    operand_memory,
    accumulator_memory,
    axis_granules,
    reserved_bytes,
    output,
    budgets,
    block,
):
    """Create a new bounded source/data view, retaining every original member."""
    required = {
        "max_member_work",
        "max_total_work",
        "max_input_bytes",
        "max_output_bytes",
        "max_workspace_bytes",
        "member_timeout_s",
    }
    if (
        not isinstance(budgets, dict)
        or set(budgets) != required
        or any(type(x) is not int or x < 1 for x in budgets.values())
        or budgets["member_timeout_s"] > 120
        or not isinstance(ordinals, list)
        or not ordinals
        or any(type(x) is not int for x in ordinals)
        or len(set(ordinals)) != len(ordinals)
    ):
        raise ValueError("capacity materialization needs finite explicit reference budgets and unique ordinals")
    sources, output, facts_path = Path(sources), Path(output).absolute(), Path(facts_path)
    original = F.verify(sources, manifest_sha256=manifest_sha256)
    facts_pin = _pin(facts_path)
    if facts_pin["sha256"] != facts_sha256:
        raise ValueError("selected capacity facts differ from their independent pin")
    if any(not 0 <= x < len(original["members"]) for x in ordinals):
        raise ValueError("capacity selection is outside the retained source roster")
    stores = J.selected_stores(
        json.loads(facts_path.read_bytes()), operand_memory=operand_memory, accumulator_memory=accumulator_memory
    )
    if output.exists() or not output.parent.is_dir():
        raise ValueError("capacity source/data view needs a fresh destination")
    output.mkdir(mode=0o700)
    report = dict(
        schema="merlin.phase0.retained_joint_capacity_materialization.v1",
        original_manifest=dict(path=str((sources / "manifest.json").absolute()), sha256=manifest_sha256),
        original_source_cohort_sha256=original["source_cohort_sha256"],
        original_member_count=len(original["members"]),
        selected_ordinals=ordinals,
        facts=facts_pin,
        stores=stores,
        axis_granules=axis_granules,
        reserved_bytes=reserved_bytes,
        budgets=budgets,
        block=block,
        phase0_admitted=False,
        performance_objective_changed=False,
        roles_changed=False,
        physical_capacity_qualified=False,
        timing_qualified=False,
        target_jobs=0,
        members=[],
    )
    for member in original["members"]:
        report["members"].append(
            dict(
                ordinal=member["ordinal"],
                request_sha256=member["request_sha256"],
                original_status=member["status"],
                original_reason=member.get("reason"),
                original_request_role=member["request"].get("role"),
                numeric_policy=member.get("numeric_policy"),
                source_identity=member.get("loader_identity"),
                status="NOT_SELECTED",
            )
        )

    def save():
        (output / "index.json").write_bytes(F._bytes(report))

    reserved_work = 0
    save()
    for ordinal in ordinals:
        member, row = original["members"][ordinal], report["members"][ordinal]
        started, cpu = time.monotonic(), time.process_time()
        try:
            declaration, description = member["request"]["declaration"], member["description"]
            source_loader = sources / member["loader"]
            if D.describe(declaration) != description or F._identity(source_loader) != member["loader_identity"]:
                raise ValueError("retained capacity source, seed, ABI or arithmetic changed")
            # The independently pinned frozen manifest owns the original source
            # epoch. Never regenerate it after a factory helper/comment changes.
            row["current_factory_render_matches_retained"] = (
                hashlib.sha256(D.render(declaration).encode()).hexdigest() == member["loader_identity"]["sha256"]
            )
            row["joint_capacity"] = J.prove(
                declaration, stores=stores, axis_granules=axis_granules, reserved_bytes=reserved_bytes
            )
            cost = B.plan(
                description,
                block=block,
                max_workspace_bytes=budgets["max_workspace_bytes"],
                max_output_bytes=budgets["max_output_bytes"],
                max_dense_source_bytes=budgets["max_input_bytes"],
            )
            input_bytes = sum(int(np.prod(x["shape"])) * np.dtype(x["dtype"]).itemsize for x in description["inputs"])
            row["reference_cost_admission"] = dict(
                cost, input_bytes=input_bytes, scalar_products=description["scalar_products"]
            )
            if (
                not cost["reference_memory_admitted"]
                or input_bytes > budgets["max_input_bytes"]
                or description["scalar_products"] > budgets["max_member_work"]
                or reserved_work + description["scalar_products"] > budgets["max_total_work"]
            ):
                raise ValueError("explicit capacity reference cost admission denied; original member retained")
            if not member.get("reference"):
                raise ValueError("retained complete reference absent; select its original supplement separately")
            reserved_work += description["scalar_products"]
            reference = sources / member["reference"]
            selected = dict(kind="original_frozen_npz", path=str(reference), identity=member["reference_identity"])
            with shared_compute_pool(timeout_s=30):
                _check_reference_inputs(reference, description, block=block)
                expected = I.load_outputs(member, selected)
                destination = output / f"member-{ordinal:04d}"
                destination.mkdir()
                loader = destination / "loader.py"
                shutil.copyfile(sources / member["loader"], loader)
                shutil.copyfile(reference, destination / "original-inputs-and-reference.npz")
                result = B.materialize(
                    declaration,
                    description,
                    destination / "blocked-reference",
                    block=block,
                    timeout_s=budgets["member_timeout_s"],
                )
                for path, wanted in zip(result["outputs"], expected, strict=True):
                    if not np.array_equal(np.load(path, mmap_mode="r", allow_pickle=False), wanted):
                        raise ValueError("complete blocked output differs from the retained independent reference")
                if _pin(loader)["sha256"] != member["loader_identity"]["sha256"]:
                    raise ValueError("capacity source copy changed")
                roster = destination / "ordinary-source-roster.json"
                roster.write_bytes(
                    F._bytes(
                        dict(
                            schema=W.SCHEMA,
                            performance_applications={declaration["name"]: dict(loader=str(source_loader.absolute()))},
                        )
                    )
                )
                staged = W.stage(roster, destination / "ordinary-source-staging")
                staged_loader = (
                    Path(staged["performance_applications"][declaration["name"]]["workload_root"]) / "loader.py"
                )
                if F._identity(staged_loader) != member["loader_identity"]:
                    raise ValueError("ordinary source staging changed the retained loader epoch")
                (destination / "declaration.json").write_bytes(F._bytes(declaration))
                row.update(
                    status="complete_source_inputs_reference_materialized",
                    declaration_sha256=description["declaration_sha256"],
                    complete_outputs=len(expected),
                    complete_values=sum(x.size for x in expected),
                    computed_products=result["computed_scalar_products"],
                    files={str(p.relative_to(output)): _pin(p) for p in sorted(destination.rglob("*")) if p.is_file()},
                    capsule_entry=D.capsule_entry(declaration, staged_loader),
                    reused_reference_checked_against_fresh_complete_blocked_computation=True,
                    source_execution_agreement=member.get(
                        "source_reference_agreement", "UNKNOWN: capture not performed"
                    ),
                    boundary=member.get("performance_boundary"),
                    target_cycles=None,
                    target_cost="UNKNOWN: no target execution",
                )
        except (ValueError, KeyError, TypeError, TimeoutError, zipfile.BadZipFile) as error:
            row.update(status="PREPARATION_REFUSED", reason=str(error))
        row.update(wall_seconds=time.monotonic() - started, python_CPU_seconds=time.process_time() - cpu)
        save()
    F.verify(sources, manifest_sha256=manifest_sha256)
    if _pin(facts_path) != facts_pin:
        raise ValueError("capacity facts changed during materialization")
    report["original_sources_reverified_after"] = True
    report["facts_reverified_after"] = True
    save()
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--facts-sha256", required=True)
    parser.add_argument("--operand-memory", required=True)
    parser.add_argument("--accumulator-memory", required=True)
    parser.add_argument("--axis-granules", type=json.loads, required=True)
    parser.add_argument("--reserved-bytes", type=json.loads, required=True)
    parser.add_argument("--ordinals", nargs="+", type=int, required=True)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--block", type=json.loads, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    args["facts_path"] = args.pop("facts")
    args["budgets"] = json.loads(args["budgets"].read_bytes())
    materialize(**args)


if __name__ == "__main__":
    main()
