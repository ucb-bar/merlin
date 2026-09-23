"""Conventional entry the merlin RTL-cert adapter loads (``<gsim_dir>/gsim_run.py``).

Exposes the SAME ``run_program`` the Verilator oracle's ``verilator_run.py`` does, so merlin can pick
either engine without naming a binary or a target. Both engines elaborate the same design and consume
the same spec JSON; GSIM is the fast one (measured 377k cycles/s against Verilator's 7.4k on this core,
cycle-exact on 17/17 runs across cycles, output bytes and DMA read/write counts).

Only two things differ from the Verilator wrapper, both handled here:
  * the trace path is a positional argument rather than a ``per_cycle_csv`` key inside the spec;
  * the binary name.
"""
import json
import subprocess
import tempfile
from pathlib import Path

_GBIN = Path(__file__).resolve().parent / "atlas_gsim_sim"


def run_program(words, preload=None, reads=None, max_cycles=20000, *, gbin=None, timeout=600,
                per_cycle_csv=None):
    """Run ``words`` (u32 IMEM program) on the GSIM AtlasCore engine.

    Signature and return shape are identical to the Verilator oracle's ``run_program`` -- that identity
    is the point: the cert tier records a fidelity, and which engine answered is an availability choice.

    Returns:
        dict: ``{"halted": bool, "cycles": int, "halt_reason": int,
                 "outputs": list[bytes], "reads": int, "writes": int}``.
    """
    gbin = Path(gbin) if gbin else _GBIN
    if not gbin.is_file():
        raise FileNotFoundError(f"GSIM engine binary absent: {gbin}")

    spec = {
        "words": [int(w) & 0xFFFFFFFF for w in words],
        "preload": [[int(a), bytes(d).hex()] for a, d in (preload or [])],
        "reads": [[int(a), int(n)] for a, n in (reads or [])],
        "max_cycles": int(max_cycles),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(spec, tf)
        spec_path = tf.name

    argv = [str(gbin), spec_path] + ([str(per_cycle_csv)] if per_cycle_csv else [])
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    finally:
        try:
            Path(spec_path).unlink()
        except OSError:
            pass

    if p.returncode != 0:
        raise RuntimeError(f"{gbin.name} rc={p.returncode}: {p.stderr[-500:]}")
    line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.strip().startswith("{")), None)
    if line is None:
        raise RuntimeError(f"no JSON on {gbin.name} stdout: {p.stdout[-500:]}")
    out = json.loads(line)
    out["outputs"] = [bytes.fromhex(h) for h in out.get("outputs", [])]
    return out
