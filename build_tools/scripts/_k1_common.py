"""Helpers the ``k1_*.py`` board drivers in this directory share -- one definition each.

Every function here was defined, identically, in two or more drivers (the comment above each names
them). Each driver imports it back under the SAME name, so ``<driver>._cc`` and friends still resolve
for anything that imports a driver as a module (``k1_fp16_gemm`` reads ``k1_cross_framework_ops._cc``,
``k1_escape_cost`` reads ``k1_large_shape_packing._cc``).

Only exact duplicates belong here. A helper whose copies differ -- a different remote scratch name,
scp timeout, an extra PMU path, a different census -- stays in its driver, because that difference is
part of what the driver measured, and merging it would change a number somebody cited.

Stdlib + merlin only. A driver run as a file reaches this module through its own directory, which is
``sys.path[0]``; a driver loaded by path (as the tests load them) inserts that directory first.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.mining import k1


# From k1_cross_framework, k1_cross_framework_ops, k1_intrinsic_microkernel, k1_large_shape_packing.
def _cc() -> Path:
    cc = k1.toolchain_cc()
    if cc is None:
        raise RuntimeError("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")
    return cc


# From k1_cross_framework and k1_intrinsic_microkernel. The k1_cross_framework_ops copy (/tmp/k1ops_*,
# optional PMU wrap) and the k1_large_shape_packing copy (/tmp/k1pack_*, 180 s scp, 600 s run) differ
# and stay in their drivers.
def _deploy_run(binary: Path, tag: str, *, timeout: int = 300) -> tuple[str | None, str]:
    """scp the binary to the board, run it, return (stdout-or-None, detail)."""
    remote = f"/tmp/k1ceil_{tag}"
    try:
        subprocess.run(
            [
                "scp",
                "-i",
                k1.K1_SSH_KEY,
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                str(binary),
                f"{k1.K1_HOST}:{remote}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return None, f"scp failed: {e.stderr[-200:] if e.stderr else e}"
    try:
        k1._ssh(f"chmod +x {remote}", timeout=30)
        p = k1._ssh(remote, timeout=timeout)
    finally:
        try:
            k1._ssh(f"rm -f {remote}", timeout=30)
        except Exception:  # noqa: BLE001
            pass
    if p.returncode != 0:
        return None, f"run rc={p.returncode}; stderr: {p.stderr.strip()[-200:]}; stdout: {p.stdout.strip()[-200:]}"
    return p.stdout, "ok"


# From k1_amax_reduction_ab and k1_pkg_cflags_ab (whose docstring was a shorter form of this one).
def run_paddings(bundle: Path, bwork: Path, pkg, elf: Path, n_paddings: int, iters: int, timeout: int) -> list[dict]:
    """Run the ALREADY-BUILT ELF on the board once per environment padding; a row per padding.

    Through `k1.run_binary_on_k1`, which deploys and runs a given binary under an explicit
    environment and takes the board lock around deploy+run only. Building once and running many
    times is the point: rebuilding per padding would put a different object under each measurement
    and make the comparison unattributable.

    The padding is a single environment variable whose length doubles each step. It changes nothing
    the program reads -- only the size of the environment block the kernel copies above the initial
    stack pointer, and therefore the alignment and absolute address of every stack object. A digest
    that moves across these has an address dependence, which for a rewrite claimed EXACT is a defect
    no cosine gate would catch.
    """
    rows = []
    for i in range(n_paddings):
        pad = "X" * (1 << (6 + i))  # 64 B, 128 B, ... doubling per padding
        env = {"MERLIN_AB_PAD": pad, "MERLIN_ITERS": str(iters)}
        try:
            res = k1.run_binary_on_k1(bundle, bwork, pkg, elf, env=env, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            rows.append({"padding_bytes": len(pad), "error": f"{type(e).__name__}: {e}"})
            continue
        # The harness prints its own digest over the output bytes; recompute host-side over the
        # parsed values as an independent check, and REPORT BOTH. A single digest that the same code
        # both produces and checks cannot detect a harness-side bug.
        outputs = res.get("outputs")
        host = hashlib.sha256(b"".join(float(v).hex().encode() for v in outputs)).hexdigest() if outputs else None
        walls = res.get("iter_wall_ns") or []
        rows.append(
            {
                "padding_bytes": len(pad),
                "board_out_hash": res.get("out_hash"),
                "host_digest_over_parsed_outputs": host,
                "n_outputs": len(outputs) if outputs else 0,
                "wall_ns_min": min(walls) if walls else None,
            }
        )
    return rows


# From k1_int8_et_campaign and k1_lever_ablation.
def _dirty(paths) -> list:
    out = []
    for rel in paths:
        got = subprocess.run(
            ["git", "status", "--porcelain", "--", rel], cwd=str(repo_root()), capture_output=True, text=True
        )
        if got.stdout.strip():
            out.append(rel)
    return sorted(out)


# From k1_int8_et_campaign and k1_lever_ablation (whose docstring was a shorter form of this one).
def _write_manifest(outdir: Path, product) -> None:
    """Keep manifest.yaml current in BOTH the fresh and the resumed case.

    A product dir under out/artifacts/<topic>/v*/ without a manifest fails the layout gate for
    everyone on the tree, so it is rewritten after every cell rather than once at the end. On a
    resume the existing manifest's identity fields (run_id / timestamp / git_sha) are PRESERVED --
    they name the campaign, and re-stamping them would silently re-date somebody's cited result.
    """
    from merlin.common.yaml import dump_yaml, load_yaml

    files = sorted(p.name for p in outdir.iterdir() if p.is_file() and p.name != "manifest.yaml")
    cells = sorted(f"cells/{p.name}" for p in (outdir / "cells").iterdir()) if (outdir / "cells").is_dir() else []
    mf = outdir / "manifest.yaml"
    if mf.is_file():
        existing = load_yaml(mf) or {}
        if isinstance(existing, dict):
            existing["artifacts"] = files + cells
            mf.write_text(dump_yaml(existing), encoding="utf-8")
            return
    if product is not None:
        product._artifacts = files + cells
        product.write_manifest()
