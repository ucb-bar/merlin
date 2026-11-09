#!/usr/bin/env python3
"""Gate: the built wheel is install-clean for the core SDK OUTSIDE a checkout.

Builds the wheel, installs it into a throwaway venv with **no repo on the path** and
``MERLIN_REPO_ROOT`` unset, then asserts the bundled read-only data classes resolve with zero
``FileNotFoundError`` — the exact failure mode that used to make ``pip install merlin`` unusable
outside the repo (data lived in sibling trees that ``parents[N]`` walks could not reach once
installed). See ``docs/reference/repo_structure.md``.

Scope tracks the executed packaging phases: P0 (schemas + prompts), P1 (the light benchmark specs,
asserting the heavy ``recaptures*`` corpora are NOT bundled), P2 (the shared contract, asserting
target-specific contracts and ``rtl_facts`` cert data are NOT bundled), and P3 (the runtime-C substrate sources),
using the same package-owned resource paths as editable installations.

Accepts an already built wheel through ``--wheel`` so CI checks the actual release
artifact without rebuilding it. The default still builds with ``uv``. Optional
local runs skip when ``uv`` is absent; ``--require-tools`` makes that a failure
for the packaging gate.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# These probes run in separate cold interpreters outside the source checkout.
_COLD_PROBE = r"""
import importlib.abc
import sys

class NoOptionalEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "torchao", "xdsl", "numpy", "jsonschema", "aet", "chia", "ray",
            "boto3", "botocore", "matplotlib", "scipy",
        }:
            raise AssertionError("cold core import/root help imported an optional engine: " + fullname)

sys.meta_path.insert(0, NoOptionalEngines())
import merlin
from merlin.cli import main
assert main(["--help"]) == 0
assert not any(name == "merlin.experiments" or name.startswith("merlin.experiments.") for name in sys.modules)
print("standalone-cold-import: OK (root help without optional engines)")
"""

_PROBE = r"""
import merlin, merlin.common.paths as p
import sys
from pathlib import Path
assert p.checkout_root() is None, "installed core still resolves a source checkout"
assert Path(merlin.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), \
    f"core imported outside the isolated venv: {merlin.__file__}"
sd, pd = p.schemas_dir(), p.prompts_dir()
assert sd.is_dir(), f"schemas_dir missing in wheel: {sd}"
assert pd.is_dir(), f"prompts_dir missing in wheel: {pd}"
from merlin.common.schemas import load_schema
load_schema("kernel_record")                       # exercises the schema-resolve + parse path
load_schema("instruction_semantics")              # selected instruction records also work from the wheel
assert (pd / "rvv_mining_v1.md").is_file(), "bundled prompt missing"
# benchmarks: the LIGHT specs bundle (P1); the heavy recaptures* deliberately do NOT.
bd = p.bench_dir()
assert (bd / "dse_guidance" / "accuracy_gate.yaml").is_file(), f"bundled benchmark spec missing: {bd}"
assert (bd / "semantic_memory" / "no_reuse_matmul.yaml").is_file(), "bundled workload spec missing"
assert not (bd / "dse_guidance" / "recaptures").exists(), "heavy recaptures leaked into the wheel"
# Shared contract (P2); target-specific contracts and rtl_facts come from selected OOT inputs.
# Probe the DATA via data_path (the contract module pulls in jsonschema, an optional extra).
cd = p.data_path("contract")
assert (cd / "schemas" / "command_buffer.schema.json").is_file(), f"bundled contract missing: {cd}"
td = p.targets_dir()
assert not any(path.is_file() for path in td.rglob("*")), f"target-specific data leaked into core wheel: {td}"
# runtime C substrate (P3): sources resolve so the compile paths fail on 'need toolchain', not 'no source'.
rd = p.runtime_dir()
assert (rd / "c" / "merlin_model.c").is_file(), f"bundled runtime C source missing: {rd}"
assert (rd / "abi" / "mlir_runtime.c").is_file(), "bundled runtime ABI source missing"
# entry points registered (import-clean at least at the console-script level)
from importlib.metadata import entry_points
console = tuple(entry_points(group="console_scripts"))
assert not any(e.value.startswith(("merlin_dse.", "merlin_experiments.")) for e in console), \
    "retired namespace remains in console entry points"
scripts = {e.name for e in console if e.value.startswith("merlin.")}
assert {"merlin", "merlin-compile", "kernel-index"} <= scripts, f"core entry points missing: {scripts}"
assert {"merlin-dse-guidance", "merlin-compare", "merlin-experiment", "merlin-recovery"} <= scripts, scripts
from importlib.metadata import PackageNotFoundError, distribution
main = distribution("merlin")
assert {"experiments", "analysis", "dse", "mining"} <= set(main.metadata.get_all("Provides-Extra") or ())
for old in ("merlin-experiments", "merlin-analysis", "merlin-dse", "merlin-mining"):
    try:
        distribution(old)
    except PackageNotFoundError:
        continue
    raise AssertionError("obsolete distribution remains installed: " + old)
# Code ships once, regardless of the optional dependencies selected at installation.
# Inspect RECORD rather than importing optional package parents and engines.
payload = {str(item) for item in main.files or ()}
retired = sorted(name for name in payload
                 if name.startswith(("merlin_dse/", "merlin_experiments/"))
                 or name in {"merlin_dse.py", "merlin_experiments.py"})
assert not retired, "retired namespace payload remains installed: " + repr(retired)
for module in ("merlin.agentreport", "merlin.experiments.phase1.controller", "merlin.targetgen.aet_bridge",
               "merlin.targetgen.experiment_tokens", "merlin.targetgen.heavy_oracles",
               "merlin.mining.beam", "merlin.verify.plots", "merlin.perf.recovery",
               "merlin.perf.source_program_pair_provider", "merlin.analysis.dse.search", "merlin.experiments",
               "merlin.analysis.dse", "merlin.analysis.dse.pressure", "merlin.analysis.dse.guidance",
               "merlin.verify.replay", "merlin.verify.replay_layers",
               "merlin.kernels.ceiling_drivers.run_expert_gemm",
               "merlin.kernels.ceiling_drivers.multishape_compare",
               "merlin.targetgen.group_capsules", "merlin.targetgen.store_probe",
               "merlin.targetgen.evaluation_cohort", "merlin.targetgen.numeric_falsifiability",
               "merlin.targetgen.rtl.gen_rocc_replay"):
    path = module.replace(".", "/")
    if path + ".py" in payload or path + "/__init__.py" in payload:
        continue
    # Existing implicit namespaces have no initializer. Resolve their exact
    # installed parent directly, without importing package parents or engines.
    from importlib.machinery import PathFinder
    members = sorted(name for name in payload if name.startswith(path + "/") and name.endswith(".py"))
    assert members, "main RECORD has no module or namespace contents: " + module
    directory = Path(main.locate_file(path)).resolve()
    own_site = Path(merlin.__file__).resolve().parent.parent
    assert directory.is_relative_to(own_site), (module, directory)
    spec = PathFinder.find_spec(module, [str(directory.parent)])
    assert spec is not None and spec.origin is None and spec.loader is None, (module, spec)
    assert tuple(Path(item).resolve() for item in spec.submodule_search_locations or ()) == (directory,), module
    assert all(Path(main.locate_file(name)).is_file() for name in members), module
from merlin.kernels import ceiling_drivers
from pathlib import Path
assert (Path(ceiling_drivers.__file__).parent / "common.h").is_file()
from merlin.targetgen.contract.matmul_interface import emit_interface_mlir
assert 'merlin_iface.target = "probe_device"' in emit_interface_mlir(
    lhs="A", weight="W", out="Y", M=2, K=3, N=4, epilogue=[],
    output_dtype="i32", target="probe_device")
from merlin.targetgen import elf_lanes
assert elf_lanes.LINKED_ELF_EVIDENCE not in elf_lanes.EXECUTED_LANE_EVIDENCE
assert set(elf_lanes.EXECUTED_LANE_EVIDENCE) < set(elf_lanes.negative_lane_evidence())
import sys
assert "merlin.targetgen.capsule_runner" not in sys.modules
print("standalone-install: OK (shared data + entry points, no target data or repo on path)")
"""


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wheel", type=Path, help="check this exact built core wheel without rebuilding")
    parser.add_argument("--require-tools", action="store_true", help="fail instead of skipping when uv is absent")
    args = parser.parse_args(argv)
    selected = args.wheel.resolve() if args.wheel is not None else None
    if selected is not None and (not selected.is_file() or selected.suffix != ".whl"):
        parser.error("--wheel requires an existing wheel file")
    if shutil.which("uv") is None:
        status = "FAIL" if args.require_tools else "SKIP"
        print(f"check_standalone_install: {status} (uv not found; install uv to run the wheel smoke test)")
        return 1 if args.require_tools else 0

    staging = REPO / "out/build/python"
    staging.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(prefix="merlin-standalone-build-", dir=staging) as build_td,
        tempfile.TemporaryDirectory(prefix="merlin-standalone-probe-") as probe_td,
    ):
        wheeldir = Path(build_td) / "wheel"
        probe_root = Path(probe_td).resolve(strict=True)
        if probe_root.is_relative_to(REPO.resolve()):
            print("FAIL: standalone probe temp directory is inside the source checkout")
            return 1
        venv = probe_root / "venv"

        if selected is None:
            r = _run(["uv", "build", "--wheel", "--out-dir", str(wheeldir), str(REPO)])
            if r.returncode != 0:
                print("FAIL: wheel build\n" + r.stderr[-2000:])
                return 1
            wheels = list(wheeldir.glob("*.whl"))
            if len(wheels) != 1:
                print("FAIL: build must produce exactly one wheel")
                return 1
            selected = wheels[0]
        with selected.open("rb") as stream:
            selected_digest = hashlib.file_digest(stream, "sha256").hexdigest()

        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        created = _run(["uv", "venv", str(venv), "--python", pyver])
        if created.returncode != 0 and not args.require_tools:
            created = _run(["uv", "venv", str(venv)])  # fall back to whatever python uv picks
        if created.returncode != 0:
            print("FAIL: external venv creation\n" + created.stderr[-2000:])
            return 1
        r = _run(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), str(selected)])
        if r.returncode != 0:
            print("FAIL: wheel install\n" + r.stderr[-2000:])
            return 1

        # Resolve installed resources, not inherited checkout overrides.
        import os

        env = {k: v for k, v in os.environ.items() if not k.startswith("MERLIN_") and k != "PYTHONPATH"}
        for name, program in (("cold import/root help", _COLD_PROBE), ("payload/resources", _PROBE)):
            r = _run([str(venv / "bin" / "python"), "-I", "-c", program], cwd=str(probe_root), env=env)
            sys.stdout.write(r.stdout)
            if r.returncode != 0:
                print("FAIL: standalone " + name + "\n" + r.stderr[-2000:])
                return 1
        with selected.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != selected_digest:
                print("FAIL: selected wheel changed during installation/probe")
                return 1
        print(f"checked-wheel-sha256: {selected_digest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
