#!/usr/bin/env python3
"""Gate: the built wheel is install-clean for the core SDK OUTSIDE a checkout.

Builds the wheel, installs it into a throwaway venv with **no repo on the path** and
``MERLIN_REPO_ROOT`` unset, then asserts the bundled read-only data classes resolve with zero
``FileNotFoundError`` — the exact failure mode that used to make ``pip install merlin`` unusable
outside the repo (data lived in sibling trees that ``parents[N]`` walks could not reach once
installed). See ``docs/design/standalone_packaging.md``.

Scope tracks the executed packaging phases: P0 (schemas + prompts), P1 (the light benchmark specs,
asserting the heavy ``recaptures*`` corpora are NOT bundled), P2 (the contract + reference target
contracts, asserting ``rtl_facts`` cert data is NOT bundled), and P3 (the runtime-C substrate sources),
via the ``merlin/_data`` bundle produced by setup.py's build_py hook.

Runs the build with ``uv`` (the repo's toolchain). Skips cleanly (exit 0) when ``uv`` is absent so a
minimal CI image without it doesn't hard-fail; wire the full run into the packaging CI job.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Probe run inside the fresh venv (cwd outside repo, MERLIN_REPO_ROOT unset). Prints OK / raises.
_PROBE = r"""
import merlin, merlin.common.paths as p
sd, pd = p.schemas_dir(), p.prompts_dir()
assert sd.is_dir(), f"schemas_dir missing in wheel: {sd}"
assert pd.is_dir(), f"prompts_dir missing in wheel: {pd}"
from merlin.common.schemas import load_schema
load_schema("kernel_record")                       # exercises the schema-resolve + parse path
assert (pd / "rvv_mining_v1.md").is_file(), "bundled prompt missing"
# benchmarks: the LIGHT specs bundle (P1); the heavy recaptures* deliberately do NOT.
bd = p.bench_dir()
assert (bd / "dse_guidance" / "accuracy_gate.yaml").is_file(), f"bundled benchmark spec missing: {bd}"
assert (bd / "semantic_memory" / "no_reuse_matmul.yaml").is_file(), "bundled workload spec missing"
assert not (bd / "dse_guidance" / "recaptures").exists(), "heavy recaptures leaked into the wheel"
# contract + reference target contracts (P2); rtl_facts cert data deliberately NOT bundled.
# Probe the DATA via data_path (the contract module pulls in jsonschema, an optional extra).
cd = p.data_path("contract")
assert (cd / "schemas" / "command_buffer.schema.json").is_file(), f"bundled contract missing: {cd}"
td = p.targets_dir()
assert (td / "gemmini" / "contracts" / "target_contract.yaml").is_file(), "bundled target contract missing"
assert not (td / "gemmini" / "contracts" / "rtl_facts").exists(), "rtl_facts cert data leaked into the wheel"
# runtime C substrate (P3): sources resolve so the compile paths fail on 'need toolchain', not 'no source'.
rd = p.runtime_dir()
assert (rd / "c" / "merlin_model.c").is_file(), f"bundled runtime C source missing: {rd}"
assert (rd / "abi" / "mlir_runtime.c").is_file(), "bundled runtime ABI source missing"
# entry points registered (import-clean at least at the console-script level)
from importlib.metadata import entry_points
scripts = {e.name for e in entry_points(group="console_scripts") if e.value.startswith("merlin.")}
assert "merlin-dse-guidance" in scripts and "kernel-index" in scripts, f"entry points missing: {scripts}"
print("standalone-install: OK (import + schemas + prompts + entry points, no repo on path)")
"""


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    if shutil.which("uv") is None:
        print("check_standalone_install: SKIP (uv not found; install uv to run the wheel smoke test)")
        return 0

    with tempfile.TemporaryDirectory(prefix="merlin-standalone-") as td:
        tmp = Path(td)
        wheeldir, venv = tmp / "wheel", tmp / "venv"

        r = _run(["uv", "build", "--wheel", "--out-dir", str(wheeldir), str(REPO)])
        if r.returncode != 0:
            print("FAIL: wheel build\n" + r.stderr[-2000:])
            return 1
        wheels = list(wheeldir.glob("*.whl"))
        if not wheels:
            print("FAIL: no wheel produced")
            return 1

        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        if _run(["uv", "venv", str(venv), "--python", pyver]).returncode != 0:
            _run(["uv", "venv", str(venv)])  # fall back to whatever python uv picks
        r = _run(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), str(wheels[0]), "pyyaml"])
        if r.returncode != 0:
            print("FAIL: wheel install\n" + r.stderr[-2000:])
            return 1

        # cwd = tmp (outside the repo); scrub MERLIN_REPO_ROOT / PYTHONPATH so nothing points home.
        import os
        env = {k: v for k, v in os.environ.items() if k not in ("MERLIN_REPO_ROOT", "PYTHONPATH")}
        r = _run([str(venv / "bin" / "python"), "-c", _PROBE], cwd=str(tmp), env=env)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            print("FAIL: standalone probe\n" + r.stderr[-2000:])
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
