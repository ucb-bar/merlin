"""Reprepare a reviewed corpus through its native owner under a new source context.

Preserves original generated lineage, public selections and reviewed operands.
This does not copy a seal or launch an author; review and sealing are separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from merlin_experiments.corpus import release


def prepare(previous, output, sdk, public_reference_manifest=None):
    previous, output, sdk = Path(previous), Path(output), Path(sdk)
    old = json.loads((previous / "private/preparation.json").read_bytes())
    assembly, scaffold = old["assembly"], old["scaffolding"]
    if assembly["mode"] != "generated_only" or assembly["baseline"] or assembly["retirements"]["members"]:
        raise ValueError("this refresh requires the reviewed generated-only source selection")
    scale = json.loads((previous / "payload/corpus/_public_scalability/selection.json").read_bytes())
    bundle = yaml.safe_load(
        (
            previous / "payload/experiment/input_bundles/merlin_assisted_rtlchecks_public_v0/input_bundle_manifest.yaml"
        ).read_text()
    )
    lean = selected_lean_manifest(bundle)

    def pair(row):
        return Path(row["path"]), row["sha256"]

    return release.prepare(
        Path(old["source_run"]),
        output,
        generated_only=True,
        capsule_scope=assembly["capsule_scope"]["kind"],
        phase1_policy_descriptor=Path(old["phase1_policy"]["source_path"]),
        lean_toolchain_manifest=lean,
        selected_levels=("EL4",),
        selected_variants=("public_v0",),
        common_substrate="merlin.common_compiler_substrate.v8",
        python_source_root=sdk,
        host_index_bits=64,
        native_timeout_s=7200,
        model_spike_timeout_s=3600,
        gsim_readback_transport="coherent_dump_v1",
        operand_protocol_selection=pair(scaffold["operand_protocol"]),
        operand_protocol_revision=pair(scaffold["operand_revision"]),
        public_supplement=Path(assembly["public_supplement"]["selection"]["path"]),
        public_scalability=Path(scale["source"]["selection"]["path"]),
        public_reference_manifest=(
            Path(public_reference_manifest)
            if public_reference_manifest is not None
            else Path(scaffold["public_reference_sources"]["path"])
        ),
    )


def selected_lean_manifest(bundle):
    """Preserve the explicit protected toolchain rather than silently omitting it."""
    expected = bundle["compiler_tool_profile"]["lean_toolchain"]
    if expected.get("selection") != "explicit_manifest":
        raise ValueError("source refresh requires the previously reviewed formal toolchain")
    selections = {
        row["closed_toolchain_root"]["manifest"]
        for row in bundle.get("host_inputs", [])
        if "closed_toolchain_root" in row
    }
    if len(selections) != 1:
        raise ValueError("source refresh requires one explicit closed formal toolchain manifest")
    path = Path(next(iter(selections)))
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected["manifest_sha256"]
    ):
        raise ValueError("selected formal toolchain manifest changed")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "output", "sdk"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--public-reference-manifest", type=Path)
    args = p.parse_args()
    report = prepare(**vars(args))
    (args.output.parent / "release-inspection.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("review_digest", "target") if k in report}))


if __name__ == "__main__":
    main()
