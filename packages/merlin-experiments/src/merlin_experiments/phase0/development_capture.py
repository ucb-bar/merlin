"""Bounded diagnostic connection from frozen public sources to ordinary writer.

This starts no compiler, simulator or search and grants no sealed admission.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen import capsule_source
from merlin.targetgen.corpus_spec import CorpusBinding

from . import development_capture_intake as I
from . import development_cohort as F
from . import development_reference as R
from . import development_sources as D


def capture(
    root,
    *,
    manifest_sha256,
    ordinals,
    output,
    max_work,
    max_elements,
    timeout_s=120,
    reference_supplements=(),
    frozen_references=False,
    max_seed_bytes=256 * 1024**2,
    max_contractions=256,
):
    if type(frozen_references) is not bool or (reference_supplements and not frozen_references):
        raise ValueError("selected reference supplements require explicit frozen reference reuse")
    if frozen_references:
        manifest, catalog = I.references(root, manifest_sha256=manifest_sha256, supplements=reference_supplements)
        intake = I.plan(
            manifest,
            catalog,
            max_work=max_work,
            max_elements=max_elements,
            max_seed_bytes=max_seed_bytes,
            max_contractions=max_contractions,
        )
    else:
        manifest = F.verify(root, manifest_sha256=manifest_sha256)
        catalog, intake = {}, None
    if (
        not ordinals
        or len(set(ordinals)) != len(ordinals)
        or len(ordinals) > 8
        or any(type(n) is not int or not 0 <= n < len(manifest["members"]) for n in ordinals)
        or any(type(n) is not int or n < 1 for n in (max_work, max_elements, timeout_s))
    ):
        raise ValueError("diagnostic capture needs explicit bounded source members and costs")
    output, root = Path(output).absolute(), Path(root)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("capture output must be fresh with an existing parent")
    output.mkdir(mode=0o700)
    source = capsule_source.PytorchRefSource(timeout=timeout_s)
    if not source.available():
        raise ValueError("selected original Model2MLIR capture runtime is unavailable")
    report = {
        "schema": "merlin.phase0.public_development_capture_diagnostic.v1",
        "source_manifest_sha256": manifest_sha256,
        "phase0_admitted": False,
        "members": [],
        **({"capture_intake": intake} if intake is not None else {}),
    }
    for ordinal in ordinals:
        original = manifest["members"][ordinal]
        row = {"ordinal": ordinal, "request_sha256": original["request_sha256"], "status": "UNKNOWN"}
        report["members"].append(row)
        (output / "report.json").write_bytes(F._bytes(report))
        start = time.monotonic()
        try:
            description = original.get("description", {})
            if (
                intake is not None
                and intake["members"][ordinal]["status"] != "capture_cost_screen_passed_not_admission"
            ):
                raise ValueError("; ".join(intake["members"][ordinal]["reasons"]))
            if (
                not description
                or description["scalar_products"] > max_work
                or description["logical_elements"] > max_elements
            ):
                raise ValueError("source unavailable or complete reference exceeds explicit diagnostic ceiling")
            declaration = original["request"]["declaration"]
            loader = root / original["loader"]
            row["loader_identity"] = F._identity(loader)
            with shared_compute_pool(timeout_s=timeout_s):
                independent_inputs = R.inputs(declaration)
                expected = (
                    I.load_outputs(original, catalog[ordinal])
                    if frozen_references
                    else R.evaluate(declaration, independent_inputs)
                )
                captured = source.capture_loader(
                    loader, "int8", workdir=output / f"capture-{ordinal}", already_quantized=True
                )
            abi = list(captured.meta.get("output_abi") or [])
            golden = list(captured.golden) if len(abi) > 1 else [captured.golden]
            if len(captured.inputs) != len(independent_inputs):
                raise ValueError("capture changed the independent source input roster")
            for actual, wanted in zip(captured.inputs, independent_inputs, strict=True):
                np.testing.assert_array_equal(actual, wanted)
            if (
                captured.meta["opaque"] != 0
                or captured.meta["materialized_contractions"] != description["contractions"]
                or len(golden) != len(expected)
                or len(abi) != len(expected)
            ):
                raise ValueError("fresh capture omitted complete outputs or required contractions")
            for actual, wanted, output_abi in zip(golden, expected, abi, strict=True):
                dtype = {"int8": "i8", "int32": "i32", "float32": "f32"}[str(wanted.dtype)]
                if output_abi["dtype"] != dtype or output_abi["shape"] != list(wanted.shape):
                    raise ValueError("fresh capture changed a complete output dtype")
                if not description.get("pending_numeric_policy"):
                    np.testing.assert_array_equal(actual, wanted)
            if description.get("pending_numeric_policy"):
                from .development_attention_diagnostic import retain

                row.update(retain(description, captured, expected, golden, output / f"pending-{ordinal}"))
                row.update(
                    capture_freshness="UNKNOWN: ordinary capture cache may be reused",
                    independent_reference_execution="fresh",
                    capture_metadata=captured.meta,
                    wall_seconds=time.monotonic() - start,
                )
                (output / "report.json").write_bytes(F._bytes(report))
                continue
            binding = CorpusBinding(
                target="source_only",
                tile_dim=None,
                operand_dtype="int8",
                accum_dtype="i32",
                integer=True,
                tiers=["L0"],
                compare="exact_int",
            )
            retained = SimpleNamespace(m2m_dir=source.m2m_dir, capture_loader=lambda *a, **kw: captured)
            directory = capsule_source.write_model_capsule(
                D.capsule_entry(declaration, loader), binding, output / "capsules", source=retained
            )
            row.update(
                status="complete_capture_and_independent_reference_match",
                capture_freshness="UNKNOWN: ordinary diagnostic cache may be reused; no sealed freshness claim",
                independent_reference_execution="reused_byte_verified_complete_outputs"
                if frozen_references
                else "fresh",
                **({"independent_reference": catalog[ordinal]} if frozen_references else {}),
                capsule=str(directory),
                capture_metadata=captured.meta,
                files={
                    str(p.relative_to(directory)): F._identity(p) for p in sorted(directory.rglob("*")) if p.is_file()
                },
            )
        except (ValueError, RuntimeError, AssertionError, capsule_source.M2MUnavailable) as error:
            row.update(status="capture_refused_or_failed", reason=str(error))
        row["wall_seconds"] = time.monotonic() - start
        (output / "report.json").write_bytes(F._bytes(report))
        print(f"public source {ordinal}: {row['status']} ({row['wall_seconds']:.2f}s)", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--ordinals", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-work", type=int, required=True)
    parser.add_argument("--max-elements", type=int, required=True)
    parser.add_argument("--frozen-references", action="store_true")
    parser.add_argument("--reference-supplement", action="append", default=[], metavar="INDEX@SHA256")
    parser.add_argument("--max-seed-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--max-contractions", type=int, default=256)
    args = parser.parse_args(argv)
    report = capture(
        args.sources,
        manifest_sha256=args.manifest_sha256,
        ordinals=args.ordinals,
        output=args.output,
        max_work=args.max_work,
        max_elements=args.max_elements,
        frozen_references=args.frozen_references,
        reference_supplements=[
            (Path(value.rsplit("@", 1)[0]), value.rsplit("@", 1)[1]) for value in args.reference_supplement
        ],
        max_seed_bytes=args.max_seed_bytes,
        max_contractions=args.max_contractions,
    )
    print(
        json.dumps(
            {"report": str(args.output / "report.json"), "statuses": [row["status"] for row in report["members"]]}
        )
    )


if __name__ == "__main__":
    main()
