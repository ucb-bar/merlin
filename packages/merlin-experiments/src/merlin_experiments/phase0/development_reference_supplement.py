"""Separate cost-admitted reference evidence for an unchanged frozen source roster."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from merlin.common.cpu_pool import shared_compute_pool

from . import development_blocked_reference as B
from . import development_cohort as F
from . import development_sources as D


def _identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(512 * 1024):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def supplement(sources, *, manifest_sha256, output, ordinals, budgets, block):
    original = F.verify(sources, manifest_sha256=manifest_sha256)
    required = {
        "max_workspace_bytes",
        "max_output_bytes",
        "max_total_output_bytes",
        "max_member_work",
        "max_total_work",
        "max_dense_source_bytes",
        "member_timeout_s",
        "total_timeout_s",
    }
    if (
        not isinstance(budgets, dict)
        or set(budgets) != required
        or any(type(value) is not int or value < 1 for value in budgets.values())
        or not isinstance(ordinals, list)
        or len(set(ordinals)) != len(ordinals)
        or any(type(n) is not int or not 0 <= n < len(original["members"]) for n in ordinals)
    ):
        raise ValueError("reference supplement needs explicit complete-roster cost admissions")
    root = Path(output).absolute()
    if root.exists() or not root.parent.is_dir():
        raise ValueError("reference supplement destination must be fresh")
    root.mkdir()
    (root / "members").mkdir()
    selected, total_work, total_output, admitted_work, admitted_output = set(ordinals), 0, 0, 0, 0
    start = time.monotonic()
    report = {
        "schema": "merlin.phase0.public_development_reference_supplement.v1",
        "source_manifest_sha256": manifest_sha256,
        "source_cohort_sha256": original["source_cohort_sha256"],
        "original_source_budgets": original["budgets"],
        "supplement_budgets": budgets,
        "block": block,
        "producer_sources": {p.name: _identity(p) for p in [Path(__file__), Path(B.__file__)]},
        "phase0_admitted": False,
        "performance_objective_changed": False,
        "execution_order": ordinals,
        "members": [],
    }
    for member in original["members"]:
        ordinal = member["ordinal"]
        row = {
            "ordinal": ordinal,
            "request_sha256": member["request_sha256"],
            "original_status": member["status"],
            "numeric_policy": member.get("numeric_policy"),
            "status": "original_reference_available" if member.get("reference") else "unavailable_not_selected",
            "original_reason": member.get("reason"),
            "source_identity": member.get("loader_identity"),
        }
        report["members"].append(row)
        if ordinal in selected:
            row["status"] = "queued_for_explicit_cost_admission"
    (root / "index.json").write_bytes(F._bytes(report))
    for ordinal in ordinals:
        member = original["members"][ordinal]
        request, row = member["request"], report["members"][ordinal]
        t0 = time.monotonic()
        try:
            declaration, description = request["declaration"], member["description"]
            if declaration["pattern"] not in B.SUPPORTED:
                raise ValueError("source pattern remains unsupported by this exact integer reference supplement")
            if (
                D.describe(declaration) != description
                or hashlib.sha256(D.render(declaration).encode()).hexdigest() != (member["loader_identity"]["sha256"])
            ):
                raise ValueError("original source/seed/ABI producer identity changed")
            admission = B.plan(
                description,
                block=block,
                max_workspace_bytes=budgets["max_workspace_bytes"],
                max_output_bytes=budgets["max_output_bytes"],
                max_dense_source_bytes=budgets["max_dense_source_bytes"],
            )
            row["cost_admission"] = admission
            work, out_bytes = description["scalar_products"], admission["output_bytes"]
            if (
                not admission["reference_memory_admitted"]
                or work > budgets["max_member_work"]
                or work + admitted_work > budgets["max_total_work"]
                or out_bytes + admitted_output > budgets["max_total_output_bytes"]
                or time.monotonic() - start > budgets["total_timeout_s"]
            ):
                raise ValueError("new explicit supplement cost admission denied; original member remains required")
            admitted_work += work
            admitted_output += out_bytes
            row["reserved_scalar_products"] = work
            row["reserved_output_bytes"] = out_bytes
            queued = time.monotonic()
            remaining = budgets["total_timeout_s"] - (time.monotonic() - start)
            with shared_compute_pool(timeout_s=min(15, remaining)):
                compute_start = time.monotonic()
                cpus = sorted(os.sched_getaffinity(0))
                result = B.materialize(
                    declaration,
                    description,
                    root / "members" / f"m{ordinal:04d}",
                    block=block,
                    timeout_s=min(budgets["member_timeout_s"], budgets["total_timeout_s"] - (time.monotonic() - start)),
                )
            row.update(
                status="complete_exact_reference_materialized",
                cpus=cpus,
                lease_wait_seconds=compute_start - queued,
                reference_wall_seconds=time.monotonic() - compute_start,
                **result,
            )
            row["outputs"] = [
                {"path": str(Path(path).relative_to(root)), **_identity(Path(path)), "abi": abi}
                for path, abi in zip(result["outputs"], description["outputs"], strict=True)
            ]
            total_work += work
            total_output += out_bytes
        except (ValueError, TimeoutError, KeyError, TypeError) as error:
            row.update(status="supplement_refused", reason=str(error))
        row["wall_seconds"] = time.monotonic() - t0
        report.update(
            computed_scalar_products=total_work,
            complete_output_bytes=total_output,
            admitted_scalar_products_upper=admitted_work,
            admitted_output_bytes_upper=admitted_output,
            wall_seconds=time.monotonic() - start,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        (root / "index.json").write_bytes(F._bytes(report))
        print(f"reference {ordinal}: {row['status']} ({row['wall_seconds']:.2f}s)", flush=True)
    (root / "index.json").write_bytes(F._bytes(report))
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--ordinals", type=int, nargs="+", required=True)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    supplement(
        args.sources,
        manifest_sha256=args.manifest_sha256,
        output=args.output,
        ordinals=args.ordinals,
        budgets=json.loads(args.budgets.read_bytes()),
        block={"M": 32, "N": 256, "K": 1024},
    )


def verify(index_path, *, index_sha256, sources, manifest_sha256):
    """Reopen existing reference evidence; never relabel it a fresh execution."""
    index_path = Path(index_path)
    if index_path.is_symlink() or _identity(index_path)["sha256"] != index_sha256:
        raise ValueError("reference supplement differs from its independent host pin")
    report = json.loads(index_path.read_bytes())
    original = F.verify(sources, manifest_sha256=manifest_sha256)
    if (
        report.get("schema") != "merlin.phase0.public_development_reference_supplement.v1"
        or report.get("source_manifest_sha256") != manifest_sha256
        or report.get("source_cohort_sha256") != original["source_cohort_sha256"]
        or len(report.get("members", [])) != len(original["members"])
    ):
        raise ValueError("reference supplement changed its original complete source inventory")
    for row, member in zip(report["members"], original["members"], strict=True):
        if any(
            row.get(key) != value
            for key, value in {
                "ordinal": member["ordinal"],
                "request_sha256": member["request_sha256"],
                "numeric_policy": member.get("numeric_policy"),
                "source_identity": member.get("loader_identity"),
            }.items()
        ):
            raise ValueError("reference supplement changed a source, seed or numerical identity")
        if row["status"] != "complete_exact_reference_materialized":
            continue
        description = member["description"]
        if (
            len(row.get("outputs", [])) != len(description["outputs"])
            or row.get("computed_scalar_products") != description["scalar_products"]
        ):
            raise ValueError("reference supplement omitted required output or work")
        for output, abi in zip(row["outputs"], description["outputs"], strict=True):
            relative = Path(output["path"])
            path = index_path.parent / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or any(p.is_symlink() for p in [path, *path.parents])
                or _identity(path) != {key: output[key] for key in ("bytes", "sha256")}
                or output["abi"] != abi
            ):
                raise ValueError("reference supplement output bytes or ABI changed")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != abi["shape"] or array.dtype != np.dtype(abi["dtype"]):
                raise ValueError("reference supplement output storage changed")
    return report


if __name__ == "__main__":
    main()
