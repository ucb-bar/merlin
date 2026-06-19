"""Re-run torch.export per workload and dump the FX graph + ATen op histogram.

The exported module (`torch.export.ExportedProgram`) is ephemeral — m2m never persists it and there is
no dump flag — so the P19 forensic audit's "exported module" column is reconstructed here, from the SAME
loader + model venv that `model2MLIR/workloads/capture.py` uses (so the dump matches the real capture's
front end). Read-only w.r.t. the model repos.

The ATen op histogram is the key audit signal: it shows ops (attention `aten.bmm`, `aten.softmax`,
`scaled_dot_product_attention`, conv, quant) AS THEY EXIST IN THE EXPORTED GRAPH — before torch-mlir
lowers them to linalg.generic — so we can see exactly what export preserves vs what lowering erases.

Usage:  python dump_exported_fx.py [<workload> ...]    (default: all workloads with a loader.py)
Output: case_study/manual_validation/exported_fx/<wl>.txt   (or <wl>.FAILED.txt on error)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

M2M = Path("/scratch/agustin/projects/model2MLIR")
WL = M2M / "workloads"
OUT = Path(__file__).resolve().parent / "case_study" / "manual_validation" / "exported_fx"

# Runs INSIDE the model's venv: rebuild the exact loader inputs, re-export, dump histogram + graph.
_INNER = r'''
import sys
from collections import Counter
sys.path.insert(0, sys.argv[1])              # workloads/<wl> dir, for `from loader import ...`
from loader import get_model_and_inputs
from m2m.capture.torch_export import capture_model
mdl, inputs = get_model_and_inputs()
ep = capture_model(mdl, inputs)
gm = getattr(ep, "graph_module", None) or ep.exported_program.graph_module
graph = getattr(ep, "graph", None) or ep.exported_program.graph
hist = Counter(str(n.target) for n in graph.nodes if n.op == "call_function")
print("=== OP HISTOGRAM (exported ATen graph, pre-lowering) ===")
for k, v in sorted(hist.items(), key=lambda x: -x[1]):
    print(f"{v:6d}  {k}")
print("=== GRAPH ===")
print(gm.print_readable(print_output=False))
'''


def _toml(d: Path) -> dict:
    import tomllib
    f = d / "capture.toml"
    return tomllib.loads(f.read_text()) if f.exists() else {}


def dump(wl: str) -> str:
    d = WL / wl
    if not (d / "loader.py").is_file():
        return f"{wl}: no loader.py"
    cfg = _toml(d)
    venv = Path(cfg.get("venv", ".venv"))
    py = (venv if venv.is_absolute() else (d / venv).resolve()) / "bin" / "python"
    if not py.exists():
        return f"{wl}: venv missing at {py}"
    env = dict(os.environ)
    env.update({k: str(v) for k, v in cfg.get("env", {}).items()})
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run([str(py), "-c", _INNER, str(d)], env=env,
                              capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        (OUT / f"{wl}.FAILED.txt").write_text("timeout (1800s)")
        return f"{wl}: TIMEOUT"
    if proc.returncode == 0 and "=== GRAPH ===" in proc.stdout:
        (OUT / f"{wl}.txt").write_text(proc.stdout)
        nhist = proc.stdout.split("=== GRAPH ===")[0].count("\n") - 1
        return f"{wl}: OK ({len(proc.stdout)} bytes, {nhist} distinct aten ops)"
    (OUT / f"{wl}.FAILED.txt").write_text(
        "STDOUT:\n" + proc.stdout[-4000:] + "\n\nSTDERR:\n" + proc.stderr[-4000:])
    return f"{wl}: FAILED (see {wl}.FAILED.txt)"


if __name__ == "__main__":
    wls = sys.argv[1:] or sorted(x.name for x in WL.iterdir() if (x / "loader.py").is_file())
    for w in wls:
        print(dump(w), flush=True)
