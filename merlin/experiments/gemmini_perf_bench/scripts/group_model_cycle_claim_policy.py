#!/usr/bin/env python3
"""Generate the v2 UART validation policy a whole-model CYCLE CLAIM is sealed against.

A cycle number for this program is citable only if the run that produced it can be shown to have
computed the model. FireSim job 730 is the measured counterexample: it returned 23,787,829 cycles,
cleared both criteria the v1 policy declared -- an argmax and a cosine -- and was WRONG, with
group 1's output summing to 5,652,929 against an oracle 5,663,048, while its argmax marker printed
BYTE-IDENTICALLY to a correct run's. The v1 shape cannot see that. The v2 shape
(:mod:`merlin.perf.firesim_batch`) can, because it declares what EVERY group's output must sum to.

THE COMPLETE SET, OR NOTHING. A v2 policy carrying some of the groups would reinstate the
under-constraint it exists to remove, quietly: a wrong group nobody declared reads exactly like a
correct one. So this tool derives one checksum for EVERY device group of the capture, from the same
numpy emulation :func:`group_model_program.emulate` runs against the capture's own golden, and the
policy it writes declares all of them. The emulation is CPU-only; no hardware is involved in
producing the oracle, which is the point -- an oracle measured on the device it is meant to check
would agree with it by construction.

WHY THE ORACLE IS THIS EMULATION, verifiably and not by assertion: run against the resnet50 capture
the emulation reproduces the three numbers independently on record from the job-730 post-mortem --
argmax 21 (want 21), cosine 997,981 ppm, and group 1 summing to 5,663,048, which is the single
oracle checksum that post-mortem published. Two of those three are the exact marker strings the v1
policy already declares, so the v2 document this tool writes constrains everything v1 did and 71
group checksums besides.

    group_model_cycle_claim_policy.py --capture <capture dir> --target <target> \\
        --policy <out>/resnet50_group_model_cycle_claim_v2.json

The provenance sidecar (``<policy stem>.provenance.json``) records what produced the numbers: the
capture's bytes, the RTL facts artifact that decided how the model was cut into groups, the merlin
sources Python actually resolved, and the canonical digest of the policy document itself. Without
the facts artifact the same capture forms 70 host regions instead of 71 device groups, so a policy
that did not name it could not be regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _script(name: str):
    """A sibling script, resolved from this file rather than from whatever ``sys.path`` holds."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, loaded)
    spec.loader.exec_module(loaded)
    return loaded


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markers(emulation: dict[str, Any]) -> tuple[str, ...]:
    """The two correctness lines the program prints, spelled from the emulation's own verdict.

    These are built here rather than copied from the v1 policy so that regenerating against a
    different capture cannot leave a stale argmax or cosine behind claiming to describe it. The
    cosine is truncated, not rounded, because the C prints ``(int)(cosine * 1000000.0)``.
    """
    got, want = int(emulation["argmax"]), int(emulation["want"])
    return (
        f"GM_ARGMAX got={got} want={want} agrees={int(got == want)}",
        f"GM_COSINE_PPM {int(emulation['cosine'] * 1000000.0)}",
    )


def checksums(model: dict[str, Any], emulation: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    """One ``(group, digest)`` per device group -- the ORDER-SENSITIVE digest, not the additive sum.

    Delegates to the program's own :func:`group_model_program.group_checksums`, which computes both
    numbers in one place precisely so a policy generator and the program it validates cannot drift
    into computing them differently.

    It must be the digest. An additive sum is PERMUTATION-BLIND: a transpose, a stride or a layout
    bug that reorders elements, and any pair of compensating +k/-k errors, leave it byte-identical.
    For a tensor compiler that is a first-class failure mode, and a policy minted on the sum would
    admit exactly those runs while looking like a correctness receipt. Measured: on one FPGA run all
    71 group checksums were wrong and the cosine moved under 1%.
    """
    program = _script("group_model_program")
    rows = [(group, int(values["fnv1a"])) for group, values in program.group_checksums(model, emulation).items()]
    return tuple(sorted(rows))


def build_policy(model: dict[str, Any], emulation: dict[str, Any], *, policy_id: str, workload: str, label: str):
    """The v2 document, assembled through the module that OWNS the shape rather than as raw JSON.

    Going through :class:`~merlin.perf.firesim_batch.BatchValidationPolicy` means this tool cannot
    emit a document that module would refuse -- in particular a window with no checksums, which is
    v1 wearing a new schema string and is rejected in the constructor.
    """
    from merlin.perf.firesim_batch import (
        WINDOW_KIND_SUM_OF_CALLS,
        BatchValidationPolicy,
        ChecksumLine,
        WindowPolicy,
    )

    return BatchValidationPolicy(
        policy_id=policy_id,
        workload=workload,
        # How this program SPELLS a group checksum:
        # `GM_GROUP <group> <kind> <cycles> sum=<additive> fnv1a=<digest>`.
        # Declared as data so the reader never learns the spelling. The claim is sealed against
        # `fnv1a`; `sum` is still printed only so a new build stays comparable with the runs
        # already on record, and must never be a policy's value_key again.
        checksum_line=ChecksumLine(prefix="GM_GROUP", value_key="fnv1a", group_token_offset=1),
        per_window=(
            WindowPolicy(
                label=label,
                markers=markers(emulation),
                checksums=checksums(model, emulation),
                # This program times each compute group separately and reports the SUM of those
                # spans, which is a different quantity from a single contiguous wall window around
                # the whole model -- it excludes every inter-group host step. Declaring the kind is
                # what lets a reader refuse a ratio between the two.
                window_kind=WINDOW_KIND_SUM_OF_CALLS,
            ),
        ),
    )


def portable(path: str | Path) -> str:
    """A path a reader on another machine can act on: relative to this repo, or to an ``out`` root.

    This record is TRACKED, and a tracked file may not name a personal directory
    (``build_tools/scripts/check_no_local_paths.py``) -- nor should it want to.  The digests beside
    each path are what identify the bytes; ``/scratch/<someone>`` identifies only the machine that
    happened to run the generator, and pins the record to a checkout nobody else has.
    """
    from merlin.common.paths import out_dir, repo_root

    resolved = Path(path).resolve()
    out_root = Path(out_dir()).resolve()
    for base, prefix in ((Path(repo_root()).resolve(), ""), (out_root, f"{out_root.name}/")):
        try:
            return prefix + resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    # Another checkout's generated root: keep the part that names the artifact, drop the machine.
    parts = resolved.parts
    for index in range(len(parts) - 1, 0, -1):
        if parts[index] == out_root.name:
            return Path(*parts[index:]).as_posix()
    return f"<outside this checkout>/{resolved.name}"


def provenance(capture: Path, target: str, script, policy_document: dict[str, Any], emulation) -> dict[str, Any]:
    """What produced these numbers, in enough detail to regenerate or to refute them."""
    from merlin.common import jsonio as _mjson
    from merlin.targetgen.rtl import facts as rtl_facts

    facts_path = rtl_facts.rtl_facts_path(target)
    # The sources PYTHON RESOLVED, so a shadowing checkout shows as a different digest.  The
    # interpreter path and the package directory are dropped: both name a machine, and neither adds
    # anything the per-module digests do not already say.
    compiler = {
        key: value for key, value in script._compiler_provenance().items() if key not in ("python", "merlin_package")
    }
    return {
        "schema": "merlin_cycle_claim_policy_provenance_v1",
        "generator": {
            "script": Path(__file__).name,
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "capture": {
            "path": portable(capture),
            **{
                f"{name}_sha256": _sha256(capture / name)
                for name in ("linalg.mlir", "weights.safetensors", "inputs.json", "golden.json")
                if (capture / name).is_file()
            },
        },
        # WHICH RTL FACTS CUT THE MODEL INTO GROUPS. Not decoration: without this artifact the same
        # capture forms 70 host regions and no device group at all, so the checksum set is a
        # property of these facts as much as of the capture.
        "rtl_facts": {
            "target": target,
            "path": portable(facts_path),
            "sha256": _sha256(facts_path) if facts_path.is_file() else "UNKNOWN: artifact absent",
        },
        "compiler_provenance": compiler,
        "emulation": {
            "argmax": int(emulation["argmax"]),
            "want": int(emulation["want"]),
            "cosine_ppm": int(emulation["cosine"] * 1000000.0),
        },
        "policy_document_sha256": _mjson.canonical_sha256(policy_document, allow_nan=True),
    }


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", required=True, type=Path, help="a capture directory (linalg.mlir, weights, ...)")
    parser.add_argument("--target", required=True)
    parser.add_argument("--policy", required=True, type=Path, help="where to write the v2 policy JSON")
    parser.add_argument("--policy-id", required=True, help="how a receipt will name this policy")
    parser.add_argument("--workload", required=True, help="the queue workload this policy is about")
    parser.add_argument(
        "--window-label",
        default=None,
        help="the measured window's label; defaults to the program renderer's own default, so the "
        "policy and an unflagged build agree without either side being told",
    )
    arguments = parser.parse_args(argv)

    script = _script("group_model_program")
    label = arguments.window_label or script.DEFAULT_WINDOW_LABEL
    try:
        model = script.extract(arguments.capture, arguments.target)
    except script.NotClosed as refusal:
        print(f"no policy written: {refusal}", file=sys.stderr)
        return 2
    emulation = script.emulate(model)
    policy = build_policy(
        model,
        emulation,
        policy_id=arguments.policy_id,
        workload=arguments.workload,
        label=label,
    )
    document = policy.to_dict()
    _write(arguments.policy, document)
    sidecar = arguments.policy.with_suffix(".provenance.json")
    _write(sidecar, provenance(arguments.capture, arguments.target, script, document, emulation))
    window = policy.per_window[0]
    print(
        f"wrote {arguments.policy}: window {window.label!r}, "
        f"{len(window.checksums)} group checksum(s), {len(window.markers)} marker(s); "
        f"provenance {sidecar}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
