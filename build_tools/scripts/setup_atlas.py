#!/usr/bin/env python3
"""Bring up the ATLAS NPU target's external dependencies for THIS clone — elegantly, without vendoring.

atlas is onboarded the same way gemmini is: everything derivable is discovered from atlas's RTL by
mlc, and the two machine-specific external checkouts (the atlas-npu RTL repo + its npu_model functional
simulator) are resolved by path via the `.env` registry (`merlin.common.paths.ext_path`), exactly like
chipyard/merlin-iree. Users of this repo do NOT clone atlas-npu as a `third_party/` submodule — they
point `.env` at wherever it already lives (typically the mlc sibling checkout, which pins it).

This script:
  1. RESOLVES the atlas-npu checkout + its embedded npu-model (from --atlas-npu / $MERLIN_EXT_ATLAS_NPU
     / the mlc sibling default), and PINS them (reports the git sha).
  2. VERIFIES the pieces the atlas oracle ladder needs: mlc arc model (`arc_available("atlas")`, the L3
     RTL-cert tier) and npu_model importability (the L2 functional tier).
  3. Optionally `--write-env` appends the two `MERLIN_EXT_*` keys to this clone's `.env`.

It never clones or builds by default (the checkouts already exist on a dev box); `--sync-npu-model`
runs `uv sync` in the npu-model dir if you need its venv. Idempotent; read-only unless --write-env.

Usage:
  python build_tools/scripts/setup_atlas.py                 # check + report (default)
  python build_tools/scripts/setup_atlas.py --write-env     # + append MERLIN_EXT_* to .env
  python build_tools/scripts/setup_atlas.py --sync-npu-model
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "merlin" / "python"))

# The mlc sibling checkout pins atlas-npu as a submodule; that is the canonical dev location.
_MLC_DEFAULT = Path("/scratch2/agustin/mvp-lhwir/modeling")
_ATLAS_NPU_DEFAULT = _MLC_DEFAULT / "third_party" / "atlas-npu"
# chipyard checkout with atlas wired in + a prebuilt whole-program Verilator sim (the L4 RTL tier).
_CHIPYARD_ATLAS_DEFAULT = Path("/scratch/agustin/projects/chipyard-atlas")
_VERILATOR_SIM_REL = "sims/verilator/simulator-chipyard.harness-AtlasRocketConfig"
# Pinned shas we onboarded against (informational — a newer master is fine, we just record drift).
_PIN_ATLAS_NPU = "569b7c3"
_PIN_NPU_MODEL = "11598ec"


def _sha(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _resolve_atlas_npu(cli: str | None) -> Path | None:
    from merlin.common.paths import env
    for cand in (cli, env("MERLIN_EXT_ATLAS_NPU"), str(_ATLAS_NPU_DEFAULT)):
        if cand and Path(cand).is_dir():
            return Path(cand)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas-npu", help="atlas-npu checkout (else $MERLIN_EXT_ATLAS_NPU / mlc sibling default)")
    ap.add_argument("--npu-model", help="npu-model checkout (else $MERLIN_EXT_NPU_MODEL / <atlas-npu>/npu-model)")
    ap.add_argument("--chipyard-atlas", help="chipyard-atlas checkout w/ the prebuilt Verilator sim (L4 RTL)")
    ap.add_argument("--write-env", action="store_true", help="append MERLIN_EXT_* to this clone's .env")
    ap.add_argument("--sync-npu-model", action="store_true", help="run `uv sync` in the npu-model dir")
    ap.add_argument("--target-package-dir", help="where to materialize the atlas OOT definition package "
                    "(default: the generated home out/build/generated/atlas — auto-discovered by "
                    "target_registry; point MERLIN_TARGET_PATH at a pinned/versioned copy to override). "
                    "Pass --no-target-package to skip.")
    ap.add_argument("--no-target-package", action="store_true",
                    help="skip materializing the OOT target-definition package")
    a = ap.parse_args()

    ok = True

    atlas_npu = _resolve_atlas_npu(a.atlas_npu)
    if not atlas_npu:
        sys.stderr.write("atlas-npu checkout not found — pass --atlas-npu or set MERLIN_EXT_ATLAS_NPU.\n")
        return 2
    npu_model = Path(a.npu_model) if a.npu_model else atlas_npu / "npu-model"
    print(f"atlas-npu   : {atlas_npu}  (sha {_sha(atlas_npu) or '?'}; onboarded @{_PIN_ATLAS_NPU})")
    print(f"npu-model   : {npu_model}  (sha {_sha(npu_model) or '?'}; onboarded @{_PIN_NPU_MODEL})")
    if not npu_model.is_dir():
        sys.stderr.write(f"npu-model dir missing under {atlas_npu} — pass --npu-model.\n"); ok = False

    if a.sync_npu_model and npu_model.is_dir():
        print(f"[sync] uv sync in {npu_model} ...")
        r = subprocess.run(["uv", "sync"], cwd=str(npu_model))
        ok = ok and (r.returncode == 0)

    # L3 RTL-cert tier: the mlc arc model must resolve (this is what makes not_run_is_not_pass real).
    try:
        from merlin.targetgen.rtl import mlc_bridge
        arc = mlc_bridge.arc_available("atlas")
        print(f"mlc arc(atlas) [L3 RTL cert] : {'OK' if arc else 'UNAVAILABLE'}")
        ok = ok and arc
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"mlc arc check failed: {type(e).__name__}: {e}\n"); ok = False

    # L2 functional tier: npu_model must be importable (the adapter shells its Simulation).
    pkg = npu_model / "npu_model"
    print(f"npu_model pkg [L2 functional] : {'present' if pkg.is_dir() else 'MISSING'} ({pkg})")
    ok = ok and pkg.is_dir()

    # L4 cycle-accurate RTL tier: the prebuilt chipyard whole-program Verilator sim (optional — the eval
    # can grade on arcilator L3 without it; verilator is the 2nd RTL tier + cross-check).
    from merlin.common.paths import env as _env
    chip = Path(a.chipyard_atlas or _env("MERLIN_EXT_CHIPYARD_ATLAS") or _CHIPYARD_ATLAS_DEFAULT)
    sim = chip / _VERILATOR_SIM_REL
    print(f"chipyard Verilator [L4 RTL]   : {'built' if sim.is_file() else 'not built'} ({sim})")

    # Materialize the OOT target-definition PACKAGE (contract + dialect_plan, derived from the CIRCT
    # facts) into the search path, so `target_registry.resolve("atlas")` finds it with ZERO env — the
    # seamless default. This is the SAME package format a user can clone/pin elsewhere and select via
    # MERLIN_TARGET_PATH. See docs/guides/target_resolution.md.
    if ok and not a.no_target_package:
        try:
            from merlin.targetgen import capability_manifests as _cm
            from merlin.targetgen import target_registry as _tr
            dest = Path(a.target_package_dir) if a.target_package_dir else _tr.generated_target_home() / "atlas"
            _cm.write_oot_target("atlas", dest)
            print(f"target package [OOT def]      : {dest} (endpoint/mesh/encoding derived from CIRCT facts)")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"target-package materialization failed: {type(e).__name__}: {e}\n"); ok = False

    if a.write_env:
        envf = ROOT / ".env"
        lines = envf.read_text().splitlines() if envf.is_file() else []
        have = {ln.split("=", 1)[0].strip() for ln in lines if "=" in ln and not ln.strip().startswith("#")}
        add = []
        if "MERLIN_EXT_ATLAS_NPU" not in have:
            add.append(f"MERLIN_EXT_ATLAS_NPU={atlas_npu}")
        if "MERLIN_EXT_NPU_MODEL" not in have:
            add.append(f"MERLIN_EXT_NPU_MODEL={npu_model}")
        if "MERLIN_EXT_CHIPYARD_ATLAS" not in have and chip.is_dir():
            add.append(f"MERLIN_EXT_CHIPYARD_ATLAS={chip}")
        if add:
            with envf.open("a") as f:
                f.write("\n# atlas-npu + npu_model (setup_atlas.py)\n" + "\n".join(add) + "\n")
            print(f"[write-env] appended {len(add)} key(s) to {envf}")
        else:
            print("[write-env] MERLIN_EXT_ATLAS_NPU / MERLIN_EXT_NPU_MODEL already set — no change")

    print("\natlas setup:", "READY" if ok else "INCOMPLETE (see messages above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
