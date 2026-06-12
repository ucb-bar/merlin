#!/usr/bin/env python3
"""Run a single consistent bundle through the dispatch gate, with timing + peak RSS."""
import sys
import tempfile
import time
import resource
from pathlib import Path

from merlin.runtime.dispatch_runtime import run_model

name = sys.argv[1]
REPO = Path(__file__).resolve().parents[2]
t = time.time()
try:
    r = run_model(REPO / "output" / name, Path(tempfile.mkdtemp()),
                  cache_dir=REPO / "output" / f".kc_{name}")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    cos = r.get("cos")
    cs = f"{cos:.7f}" if cos == cos else "nan"
    print(f"RUNRESULT {name} cos={cs} rel={r.get('rel'):.2e} ok={r.get('ok')} "
          f"kernels={r['n_kernels']} {time.time()-t:.0f}s peakRSS={rss:.1f}GB", flush=True)
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    print(f"RUNRESULT {name} ERROR {type(exc).__name__}: {str(exc)[:200]}", flush=True)
