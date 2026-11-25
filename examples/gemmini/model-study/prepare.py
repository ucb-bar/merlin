#!/usr/bin/env python3
"""Pin a model study and lower its real interfaces without changing a frozen backend.

This is an operator diagnostic. Compilation probes establish neither numerical
correctness nor full-model execution. No tensor values are passed to the compiler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def tree_pin(root, expected):
    root = root.resolve(strict=True)
    digest = hashlib.sha256()
    members = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise ValueError("selected compiler library contains a symlink")
    for path in members:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    if digest.hexdigest() != expected:
        raise ValueError("selected compiler library differs from its tree identity")
    return {"path": str(root), "source_sha256": expected}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_compiler(root, inventory, expected):
    document = json.loads(inventory.read_bytes())
    digest = hashlib.sha256()
    names = set()
    for row in document["files"]:
        name = Path(row["source_path"])
        if name.is_absolute() or ".." in name.parts or str(name) in names:
            raise ValueError("invalid or repeated compiler inventory path")
        names.add(str(name))
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"compiler inventory escapes its source root: {name}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"compiler source changed: {name}")
        if bool(path.stat().st_mode & 0o111) != row["executable"]:
            raise ValueError(f"compiler executable mode changed: {name}")
        digest.update(name.as_posix().encode() + b"\0" + data + b"\0")
    members = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise ValueError("compiler source contains an unpinned symlink")
    actual = {p.relative_to(root).as_posix() for p in members if p.is_file()}
    if actual != names:
        raise ValueError("compiler inventory membership differs")
    if digest.hexdigest() != expected or document["tree_sha256"] != expected:
        raise ValueError("compiler inventory digest differs from selected source identity")
    return {"source_sha256": expected, "files": len(names), "inventory": pin(inventory)}


def sealed_entries(bundle, manifest, names):
    """Read a finite selected roster from the complete, hash-checked seal index."""
    seal = manifest["seal"]
    index = bundle / seal["sums"]
    if not index.resolve().is_relative_to(bundle):
        raise ValueError("bundle seal escapes its selected root")
    digest, expected = hashlib.sha256(), {}
    with index.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            value, separator, name = raw.decode().rstrip("\n").partition("  ")
            if separator and name in names:
                if name in expected:
                    raise ValueError("duplicate selected metadata entry in bundle seal")
                expected[name] = value
    if digest.hexdigest() != seal["sums_sha256"] or set(expected) != names:
        raise ValueError("selected bundle metadata has no matching complete seal index")
    return index, expected


def sealed_files(bundle, manifest, names):
    """Verify a finite selected metadata roster against the complete sealed index."""
    index, expected = sealed_entries(bundle, manifest, names)
    result = [pin(bundle / name) for name in sorted(names)]
    if any(row["sha256"] != expected[Path(row["path"]).relative_to(bundle).as_posix()] for row in result):
        raise ValueError("selected bundle metadata differs from its sealed bytes")
    return [pin(index), *result]


def routing_readiness(bundle, manifest, programs):
    """Check required prepared programs and routing without repairing a frozen bundle."""
    names = set()
    for row in programs:
        prefix = f"{row['model']}/programs/{row['program']}"
        names.update(f"{prefix}/{name}" for name in (
            "dev/device_signatures.json", "ref/model.prepared.mlir", "ref/quantization-policy.json"))
        if row["interfaces"]:
            names.update(f"{prefix}/dev/{name}" for name in ("model.prepared.mlir", "quantization-policy.json"))
    _, expected = sealed_entries(bundle, manifest, names)
    rows = []
    for name in sorted(names):
        path = bundle / name
        row = {"path": str(path), "expected_sha256": expected[name]}
        if path.is_symlink():
            row["symlink_target"] = str(path.readlink())
        if not path.is_file():
            row["status"] = "dangling_symlink" if path.is_symlink() else "missing"
        else:
            row["observed"] = pin(path)
            row["status"] = "verified" if row["observed"]["sha256"] == expected[name] else "changed"
        rows.append(row)
    return {"ready": all(row["status"] == "verified" for row in rows), "files": rows}


def runtime_probe(python, environment, output, timeout):
    script = """import importlib, json, sys
import hashlib
from pathlib import Path
result = {'python': sys.version, 'executable': sys.executable, 'modules': {}}
for name in ['z3', 'numpy', 'yaml', 'xdsl']:
    module = importlib.import_module(name)
    result['modules'][name] = {'path': module.__file__, 'source_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), 'version': str(getattr(module, '__version__', 'unknown'))}
import z3
solver = z3.Solver()
value = z3.Int('model_study_probe')
solver.add(value > 0, value < 0)
result['solver_version'] = z3.get_version_string()
result['solver_result'] = str(solver.check())
result['solver_libraries'] = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((Path(z3.__file__).parent / 'lib').glob('libz3*')) if p.is_file()]
print(json.dumps(result))
"""
    command = [str(python), "-B", "-c", script]
    started = time.monotonic()
    with (output / "runtime.stdout").open("wb") as out, (output / "runtime.stderr").open("wb") as err:
        result = subprocess.run(command, env=environment, cwd=output, stdout=out, stderr=err, timeout=timeout)
    record = {"argv": command, "returncode": result.returncode, "seconds": time.monotonic() - started,
              "stdout": pin(output / "runtime.stdout"), "stderr": pin(output / "runtime.stderr")}
    if result.returncode == 0:
        record["observed"] = json.loads((output / "runtime.stdout").read_bytes())
        if record["observed"]["solver_result"] != "unsat":
            raise ValueError("mandatory scheduling solver did not pass its smoke check")
    return record


def prepare(args):
    import yaml

    output = args.output.resolve()
    compiler = args.compiler.resolve(strict=True)
    bundle = args.bundle.resolve(strict=True)
    temporary = args.temporary.resolve()
    library_root = args.compiler_library.resolve(strict=True)
    if any(path.exists() or any(path.is_relative_to(root) for root in (compiler, bundle, library_root))
           for path in (output, temporary)):
        raise ValueError("select fresh output and temporary directories outside the compiler and frozen bundle")
    if output.is_relative_to(temporary) or temporary.is_relative_to(output):
        raise ValueError("output and temporary directories must be disjoint")
    identity = verify_compiler(compiler, args.compiler_inventory, args.compiler_sha256)
    library = tree_pin(args.compiler_library, args.compiler_library_sha256)
    models_pin, manifest_pin = pin(args.models), pin(bundle / "MANIFEST.json")
    config = yaml.safe_load(args.models.read_text())
    manifest = json.loads((bundle / "MANIFEST.json").read_bytes())
    if manifest_pin["sha256"] != args.bundle_manifest_sha256:
        raise ValueError("bundle manifest differs from its selected identity")
    if args.models.read_bytes() != (bundle / "configuration/models.yaml").read_bytes():
        raise ValueError("models configuration differs from the selected frozen reference")
    names = {"configuration/models.yaml"} | {
        f"{name}/programs/{program['name']}/catalog.json"
        for name in args.model for program in config["models"][name]["programs"]}
    consumed = sealed_files(bundle, manifest, names)
    output.mkdir(parents=True)
    producer = output / "prepare-source.py"
    producer.write_bytes(Path(__file__).read_bytes())
    temporary.mkdir(parents=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(temporary))
    if {p.resolve(strict=True) for p in args.pythonpath} != {compiler / "compiler_library", library_root}:
        raise ValueError("select only the frozen compiler overlay and its pinned generic library")
    environment["PYTHONPATH"] = os.pathsep.join(str(p.resolve(strict=True)) for p in args.pythonpath)
    record = {"schema": "merlin.model_study.preparation.v1", "status": "running",
              "compiler": {**identity, "path": str(compiler)}, "models": models_pin,
              "bundle_manifest": manifest_pin, "python": pin(args.python), "compiler_library": library,
              "producer": pin(producer), "invoked_interpreter": str(args.python.absolute()),
              "exact_historical_runtime_equivalence": "not_established_by_import_smoke",
              "environment": {k: environment[k] for k in ("PYTHONDONTWRITEBYTECODE", "TMPDIR", "PYTHONPATH")},
              "numerical_tracks": ["captured_arithmetic", "expert_matched_if_recovered"],
              "numerical_execution": "not_run", "hardware_execution": "not_run", "programs": [],
              "selected_metadata": consumed, "full_bundle_payload_verified": False}
    write(output / "preparation.json", record)
    try:
        record["runtime"] = runtime_probe(args.python, environment, output, args.timeout)
        if record["runtime"]["returncode"]:
            raise ValueError("selected compiler Python cannot import its required runtime")
        for name in args.model:
            model = config["models"][name]
            for program in model["programs"]:
                program_name = program["name"]
                catalog_path = bundle / name / "programs" / program_name / "catalog.json"
                catalog = json.loads(catalog_path.read_bytes())
                stage = Path(catalog["stage"])
                metadata = json.loads((stage / "meta.json").read_bytes())
                receipt = json.loads((stage / "capture_receipt.json").read_bytes())
                if sha(stage / "meta.json") != receipt["artifacts"]["meta.json"]["sha256"]:
                    raise ValueError("capture metadata differs from its own capture receipt")
                row = {"model": name, "program": program_name, "steps": program.get("steps", 1),
                       "catalog": pin(catalog_path), "capture": str(stage),
                       "capture_receipt": pin(stage / "capture_receipt.json"), "metadata": pin(stage / "meta.json"),
                       "loader_provenance": metadata.get("loader_provenance"),
                       "interfaces": len(catalog["interfaces"]), "probes": []}
                record["programs"].append(row)
                consumed.extend([row["metadata"], row["capture_receipt"]])
                # This example must never turn the generic two-layer smoke recipe into a full model.
                if name == "tiny_llama" and not (metadata.get("loader_provenance") or {}).get("full_checkpoint"):
                    raise ValueError("TinyLlama capture does not declare a complete checkpoint")
                folder = output / name / program_name
                folder.mkdir(parents=True)
                for symbol, text in sorted(catalog["interfaces"].items()):
                    if not symbol.isidentifier():
                        raise ValueError("unsupported interface symbol")
                    interface = folder / f"{symbol}.mlir"
                    interface.write_text(text)
                    probe = {"symbol": symbol, "interface": pin(interface), "status": "not_run"}
                    row["probes"].append(probe)
                    if not args.compile_probes or len(row["probes"]) > args.max_probes:
                        continue
                    command = [str(args.python), "-B", str(compiler / "compiler.py"),
                               "--convert-iface-to-gemmini", "--emit-target-artifact", str(interface)]
                    started = time.monotonic()
                    stdout, stderr = folder / f"{symbol}.target.mlir", folder / f"{symbol}.stderr"
                    try:
                        with stdout.open("wb") as out, stderr.open("wb") as err:
                            result = subprocess.run(command, cwd=folder, env=environment, stdout=out,
                                                    stderr=err, timeout=args.timeout)
                        probe.update(status="lowered" if result.returncode == 0 else "failed",
                                     returncode=result.returncode)
                    except subprocess.TimeoutExpired:
                        probe.update(status="timeout", returncode=None)
                    probe.update(argv=command, seconds=time.monotonic() - started,
                                 artifact=pin(stdout), stderr=pin(stderr))
                    write(output / "preparation.json", record)
        verify_compiler(compiler, args.compiler_inventory, args.compiler_sha256)
        tree_pin(args.compiler_library, args.compiler_library_sha256)
        runtime = record["runtime"]["observed"]
        runtime_files = [*runtime["modules"].values(), *runtime["solver_libraries"]]
        if any(sha(row["path"]) != row.get("source_sha256", row.get("sha256")) for row in runtime_files):
            raise ValueError("observed proof runtime files changed during compilation probes")
        if any(pin(row["path"]) != row for row in consumed):
            raise ValueError("selected metadata changed during compilation probes")
        if pin(args.models) != models_pin or pin(bundle / "MANIFEST.json") != manifest_pin:
            raise ValueError("selected model metadata changed during preparation")
        record["routing_artifacts"] = routing_readiness(bundle, manifest, record["programs"])
        record["selected_metadata"].extend(row["observed"] for row in record["routing_artifacts"]["files"]
                                           if row["status"] == "verified")
        failed = [p for r in record["programs"] for p in r["probes"] if p["status"] in {"failed", "timeout"}]
        record["status"] = "probe_failures" if failed else "prepared"
    except Exception as error:
        record.update(status="blocked", error=f"{type(error).__name__}: {error}")
    write(output / "preparation.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compiler", "compiler-inventory", "compiler-library", "bundle", "models", "python", "output", "temporary"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--compiler-sha256", required=True)
    parser.add_argument("--compiler-library-sha256", required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--pythonpath", type=Path, action="append", default=[])
    parser.add_argument("--model", choices=("resnet50", "tiny_llama", "smolvla"), action="append", required=True)
    parser.add_argument("--compile-probes", action="store_true")
    parser.add_argument("--max-probes", type=int, default=1, help="maximum attempted interfaces per program")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 3600 or not 1 <= args.max_probes <= 4096:
        parser.error("timeout must be 1..3600 and max-probes 1..4096")
    result = prepare(args)
    print(json.dumps({"status": result["status"], "report": str(args.output / "preparation.json")}))
    return 0 if result["status"] == "prepared" else 1


if __name__ == "__main__":
    raise SystemExit(main())
