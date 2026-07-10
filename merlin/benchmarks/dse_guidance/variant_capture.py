"""P20-S3: one-off CAVEAT-RESOLVING variant captures (does NOT touch the committed corpus).

Produces capture variants that resolve two P19 caveats, into recaptures_decode/<tag>/model.mlir (gitignored,
regenerable). Bypasses workloads/capture.py (whose capture.toml [env] would override our knobs) and calls
m2m.convert directly in the model venv with an overridden env:

  - tiny_llama_decode : M2M_SEQ=1  -> true single-token decode (M=1), vs the committed M=4/8 PREFILL
                        ("GEMV is capture-M-induced" caveat).
  - rdt_depth6        : M2M_RDT_DEPTH=6 -> the blocks.1.cross_attn.kv giant-op share drops vs depth=2
                        ("rdt giant op doesn't generalize" caveat).

Usage: python variant_capture.py        (runs both)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

M2M = Path("/path/to/model2MLIR")
WL = M2M / "workloads"
OUT = Path(__file__).resolve().parent / "recaptures_decode"

# (tag, workload, env_overrides)
VARIANTS = [
    ("tiny_llama_decode", "tiny_llama", {"M2M_SEQ": "1"}),
    ("rdt_depth6", "rdt", {"M2M_RDT_DEPTH": "6"}),
]

_INNER = r'''
import sys
sys.path.insert(0, sys.argv[1])
from loader import get_model_and_inputs
import m2m
mdl, inputs = get_model_and_inputs()
r = m2m.convert(mdl, inputs, backend="fx_importer", quantization=None, level="linalg-on-tensors")
open(sys.argv[2], "w").write(r.mlir_text)
print("MLIR_OK", len(r.mlir_text))
'''


def _toml(d: Path) -> dict:
    import tomllib
    f = d / "capture.toml"
    return tomllib.loads(f.read_text()) if f.exists() else {}


def run(tag: str, wl: str, overrides: dict) -> str:
    d = WL / wl
    cfg = _toml(d)
    venv = Path(cfg.get("venv", ".venv"))
    py = (venv if venv.is_absolute() else (d / venv).resolve()) / "bin" / "python"
    if not py.exists():
        return f"{tag}: venv missing {py}"
    env = dict(os.environ)
    env.update({k: str(v) for k, v in cfg.get("env", {}).items()})  # capture.toml defaults...
    env.update({k: str(v) for k, v in overrides.items()})           # ...then OUR overrides win
    odir = OUT / tag
    odir.mkdir(parents=True, exist_ok=True)
    out = odir / "model.mlir"
    proc = subprocess.run([str(py), "-c", _INNER, str(d), str(out)], env=env,
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode == 0 and out.is_file():
        return f"{tag}: OK ({out.stat().st_size} bytes; overrides={overrides})"
    (odir / "FAILED.txt").write_text(proc.stdout[-3000:] + "\n---\n" + proc.stderr[-3000:])
    return f"{tag}: FAILED (see {odir}/FAILED.txt)"


if __name__ == "__main__":
    for tag, wl, ov in VARIANTS:
        print(run(tag, wl, ov), flush=True)
