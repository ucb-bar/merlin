"""Rebuild the selected Gemmini Verilator binary from pinned RTL and compare exact bytes.

This is a finite build-provenance check for one selected Chipyard configuration,
not a formal RTL correctness proof or an attestation of Spike's functional model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FIRTOOL_TIMEOUT_S = 180
VERILATOR_TIMEOUT_S = 180
BUILD_TIMEOUT_S = 600


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(argv: list[str], *, timeout_s: int, cwd: Path | None = None) -> dict:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=timeout_s, check=False)
    if result.returncode:
        stderr = result.stderr.decode("utf-8", errors="replace")
        diagnostics = [
            line.strip()[:500] for line in stderr.splitlines() if re.search(r"\b(?:error|fatal):", line, re.I)
        ]
        excerpt = "\n".join(diagnostics[:8]) if diagnostics else stderr[:1200]
        raise RuntimeError(f"tool exit {result.returncode}: {excerpt} [stderr_bytes={len(result.stderr)}]")
    return {
        "argv": argv,
        "returncode": result.returncode,
        "timeout_seconds": timeout_s,
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
    }


def ignored_output(path: Path) -> Path:
    root = path.resolve()
    if not root.is_relative_to(REPO / "out"):
        raise ValueError(f"output must resolve beneath {REPO / 'out'}")
    for candidate in (root / "receipt.json", root / "verilated/VTestDriver.cpp"):
        check = subprocess.run(["git", "check-ignore", "-q", "--", str(candidate)], cwd=REPO, timeout=5, check=False)
        if check.returncode:
            raise ValueError(f"output path is not git-ignored: {candidate}")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing evidence in {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def recorded_verilator_args(manifest: Path, *, original_binary: Path, original_model: Path) -> list[str]:
    line = next((line for line in manifest.read_text(encoding="utf-8").splitlines() if line.startswith("C ")), None)
    if line is None:
        raise ValueError("Verilator input manifest has no recorded command")
    fields = shlex.split(line[2:])
    if len(fields) != 1:
        raise ValueError("unexpected Verilator command encoding")
    raw = shlex.split(fields[0])
    # The timestamp file flattens the two quoted multiword compiler arguments.
    a, b, c = (raw.index(flag) for flag in ("-CFLAGS", "-LDFLAGS", "--threads"))
    if not a < b < c:
        raise ValueError("unexpected Verilator compiler/linker argument layout")
    argv = raw[: a + 1] + [" ".join(raw[a + 1 : b]), "-LDFLAGS", " ".join(raw[b + 1 : c])] + raw[c:]
    for flag, expected in (("-o", original_binary), ("-Mdir", original_model)):
        if argv.count(flag) != 1 or Path(argv[argv.index(flag) + 1]).resolve() != expected:
            raise ValueError(f"recorded Verilator {flag} does not name selected build")
    if (
        argv.count("-f") != 1
        or Path(argv[argv.index("-f") + 1]).resolve() != original_model.parent / "sim_files.common.f"
    ):
        raise ValueError("recorded Verilator filelist is not the selected build filelist")
    return argv


def relocate_hierarchy_annotations(original: Path, selected_dir: Path, artifact_dir: Path) -> tuple[Path, list[dict]]:
    """Project only FIRTOOL's two output-only hierarchy paths into this artifact."""
    rows = json.loads(original.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("selected annotations are not a list")
    destinations = {
        "sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation": "model_module_hierarchy.json",
        "sifive.enterprise.firrtl.ModuleHierarchyAnnotation": "top_module_hierarchy.json",
    }
    moves = []
    projected_rows = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("selected annotation is not a mapping")
        item = dict(row)
        annotation_class = item.get("class")
        if annotation_class in destinations:
            filename = destinations[annotation_class]
            before = str(selected_dir / filename)
            if item.get("filename") != before:
                raise ValueError(f"unexpected hierarchy output for {annotation_class}")
            after = str(artifact_dir / filename)
            item["filename"] = after
            moves.append({"class": annotation_class, "from": before, "to": after})
        elif "filename" in item:
            raise ValueError(f"unreviewed annotation filename: {annotation_class}")
        projected_rows.append(item)
    if len(moves) != 2 or {move["class"] for move in moves} != set(destinations):
        raise ValueError("selected annotations lack the exact two hierarchy output declarations")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    projected = artifact_dir / "annotations.json"
    projected.write_text(json.dumps(projected_rows, indent=2) + "\n", encoding="utf-8")
    return projected, moves


def attest(
    *,
    source_evidence: Path,
    chipyard: Path,
    firtool: Path,
    verilator: Path,
    kernel_receipt: Path,
    output_root: Path,
    jobs: int,
) -> dict:
    if not 1 <= jobs <= 16:
        raise ValueError("jobs must be between 1 and 16")
    source_evidence = source_evidence.resolve(strict=True)
    chipyard = chipyard.resolve(strict=True)
    firtool = firtool.resolve(strict=True)
    verilator = verilator.resolve(strict=True)
    kernel_receipt = kernel_receipt.resolve(strict=True)
    selection_path = source_evidence / "source-selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validation = json.loads((source_evidence / "validation.json").read_text(encoding="utf-8"))
    if selection["target"] != "gemmini" or selection["hierarchy_correspondence"]["status"] != "verified":
        raise ValueError("selected source is not a structurally verified Gemmini source")
    if validation["status"] != "verified":
        raise ValueError("selected source validation is not verified")
    config = selection["config"]
    if re.fullmatch(r"[A-Za-z0-9_]+", config) is None:
        raise ValueError("unsafe configuration name")
    stem = f"chipyard.harness.TestHarness.{config}"
    build = chipyard / "sims/verilator/generated-src" / stem
    original_binary = chipyard / "sims/verilator" / f"simulator-chipyard.harness-{config}"
    original_model = build / stem
    raw_firrtl = build / f"{stem}.fir"
    script_sha = sha(Path(__file__).resolve())
    raw_sha = sha(raw_firrtl)
    if raw_sha != selection["sources"]["firrtl"]["sha256"]:
        raise ValueError("selected FIRRTL bytes differ from Chipyard build input")
    old = json.loads(kernel_receipt.read_text(encoding="utf-8"))
    if old.get("schema") != "merlin.gemmini-kernel-numerical-qualification.v1":
        raise ValueError("not a Gemmini kernel numerical receipt")
    if old["selected_source"]["selection"]["sha256"] != sha(selection_path):
        raise ValueError("kernel receipt is bound to a different selected source")
    original_sha = sha(original_binary)
    if old["toolchain"]["verilator"]["sha256"] != original_sha:
        raise ValueError("kernel receipt used a different Verilator executable")
    annotations = build / f"{stem}.appended.anno.json"
    manifest = original_model / "VTestDriver__verFiles.dat"
    listed_sources = [
        Path(shlex.split(line)[-1]).resolve(strict=True)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.startswith("S ")
    ]
    if len(listed_sources) < 200 or any(not path.is_file() for path in listed_sources):
        raise ValueError("Verilator source-input manifest is incomplete")
    stable_inputs = {
        path: sha(path)
        for path in (
            Path(__file__).resolve(),
            selection_path,
            raw_firrtl,
            annotations,
            build / ".mfc_lowering_options",
            build / "sim_files.common.f",
            manifest,
            firtool,
            verilator,
            kernel_receipt,
            original_binary,
            *listed_sources,
        )
    }
    output_root = ignored_output(output_root)
    lowering = (build / ".mfc_lowering_options").read_text(encoding="utf-8").strip()
    # FIRTOOL's hierarchy export writes sidecars beside its input FIRRTL. The
    # selected Chipyard source may be read-only; preserve it and export from
    # an exact, hashed copy inside this new attestation artifact instead.
    firtool_input = output_root / "firtool" / raw_firrtl.name
    firtool_input.parent.mkdir(parents=True)
    shutil.copyfile(raw_firrtl, firtool_input)
    if sha(firtool_input) != raw_sha:
        raise ValueError("FIRTOOL input copy differs from selected FIRRTL")
    firtool_annotations, annotation_moves = relocate_hierarchy_annotations(annotations, build, firtool_input.parent)
    projected_annotations_sha = sha(firtool_annotations)
    fresh_rtl = output_root / "firtool/gen-collateral"
    fresh_rtl.parent.mkdir(parents=True, exist_ok=True)
    firtool_step = run_command(
        [
            str(firtool),
            "--format=fir",
            "--export-module-hierarchy",
            "--verify-each=true",
            "--warn-on-unprocessed-annotations",
            "--disable-annotation-classless",
            "--disable-annotation-unknown",
            f"--lowering-options={lowering}",
            "--repl-seq-mem",
            f"--repl-seq-mem-file={output_root / 'firtool/mems.conf'}",
            f"--annotation-file={firtool_annotations}",
            "--split-verilog",
            "-o",
            str(fresh_rtl),
            str(firtool_input),
        ],
        timeout_s=FIRTOOL_TIMEOUT_S,
    )
    original_rtl = build / "gen-collateral"
    core_names = selection["production"]["included_modules"]
    if len(core_names) < 100:
        raise ValueError("selected core closure unexpectedly small")
    core_hashes = {}
    for name in core_names:
        filename = name if name.endswith((".sv", ".v")) else f"{name}.sv"
        if name in ("plusarg_reader", "plusarg_reader_96"):
            # FIRRTL external-module aliases share one emitted blackbox definition.
            filename = "plusarg_reader.v"
        produced, selected = fresh_rtl / filename, original_rtl / filename
        if not produced.is_file() or sha(produced) != sha(selected):
            raise ValueError(f"selected core RTL differs from fresh FIRRTL lowering: {name}")
        core_hashes[filename] = sha(produced)

    args = recorded_verilator_args(manifest, original_binary=original_binary, original_model=original_model)
    fresh_model = output_root / "verilated"
    fresh_binary = output_root / "simulator-rebuild"
    args[args.index("-o") + 1] = str(fresh_binary)
    args[args.index("-Mdir") + 1] = str(fresh_model)
    verilator_step = run_command([str(verilator), *args], timeout_s=VERILATOR_TIMEOUT_S)
    model_files = sorted(p for p in fresh_model.iterdir() if p.suffix in (".cpp", ".h"))
    if len(model_files) < 200:
        raise ValueError("regenerated Verilator model unexpectedly small")
    for path in model_files:
        if sha(path) != sha(original_model / path.name):
            raise ValueError(f"regenerated Verilator model differs: {path.name}")
    make_executable = shutil.which("make")
    if make_executable is None:
        raise FileNotFoundError("make is not available")
    make = Path(make_executable).resolve(strict=True)
    compiler = chipyard / ".conda-env/bin/x86_64-conda-linux-gnu-c++"
    build_step = run_command(
        [str(make), "-C", str(fresh_model), "-f", "VTestDriver.mk", "-j", str(jobs)],
        timeout_s=BUILD_TIMEOUT_S,
    )
    if sha(fresh_binary) != original_sha:
        raise ValueError("rebuilt Verilator executable differs from kernel-tested executable")
    if (
        sha(firtool_input) != raw_sha
        or sha(firtool_annotations) != projected_annotations_sha
        or any(sha(path) != digest for path, digest in stable_inputs.items())
    ):
        raise ValueError("a selected build input changed during attestation")
    selected_tools = {
        "firtool": {"path": str(firtool), "sha256": sha(firtool)},
        "verilator": {"path": str(verilator), "sha256": sha(verilator)},
        "make": {"path": str(make), "sha256": sha(make)},
        "compiler": {"path": str(compiler), "sha256": sha(compiler)},
    }
    record = {
        "schema": "merlin.gemmini-verilator-provenance.v1",
        "status": "reproduced_exact_binary",
        "claim": (
            "selected Gemmini core RTL and current full Chipyard build inputs reproduce "
            "the kernel-tested Verilator executable byte-for-byte"
        ),
        "limits": [
            "This is build-byte provenance, not formal RTL correctness or all-input numerical proof.",
            "The full simulator includes SoC and external modules beyond the selected Gemmini core closure.",
            "Spike/libgemmini is a separately hashed functional model, not derived from the selected RTL.",
        ],
        "attester_source_sha256": script_sha,
        "source_selection_sha256": sha(selection_path),
        "selected_firrtl_sha256": raw_sha,
        "firtool_input_copy_sha256": sha(firtool_input),
        "annotations_sha256": sha(annotations),
        "annotation_output_projection": {
            "original_sha256": sha(annotations),
            "projected_sha256": projected_annotations_sha,
            "moves": annotation_moves,
            "scope": "output-only hierarchy filenames; regenerated RTL, C++ model and binary must match exact bytes",
        },
        "lowering_options_sha256": sha(build / ".mfc_lowering_options"),
        "verilator_input_manifest_sha256": sha(manifest),
        "verilator_filelist_sha256": sha(build / "sim_files.common.f"),
        "stable_verilator_source_count": len(listed_sources),
        "stable_build_inputs_sha256": hashlib.sha256(
            json.dumps({str(path): digest for path, digest in stable_inputs.items()}, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "kernel_receipt_sha256": sha(kernel_receipt),
        "kernel_tested_binary_sha256": original_sha,
        "rebuilt_binary_sha256": sha(fresh_binary),
        "core_rtl_file_count": len(core_hashes),
        "core_rtl_sha256": core_hashes,
        "verilated_cpp_header_count": len(model_files),
        "toolchain": selected_tools,
        "steps": {"firtool": firtool_step, "verilator": verilator_step, "make": build_step},
    }
    (output_root / "receipt.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--chipyard", type=Path, required=True)
    parser.add_argument("--firtool", type=Path, required=True)
    parser.add_argument("--verilator", type=Path, required=True)
    parser.add_argument("--kernel-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    receipt = attest(**vars(args))
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "kernel_tested_binary_sha256",
                    "core_rtl_file_count",
                    "verilated_cpp_header_count",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
