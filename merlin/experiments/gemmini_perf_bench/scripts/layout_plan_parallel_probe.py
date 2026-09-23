import argparse
import importlib.util
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import islice, product
from math import prod
from pathlib import Path

_G = {}


def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _init(graph, names, index):
    _G["graph"], _G["names"], _G["index"] = graph, names, index


def _chunk(choices_list):
    """Evaluate a block of assignments; return the best (key, plan-is-not-None) seen."""
    from merlin.perf.physical_layout import _evaluate

    graph, names, index = _G["graph"], _G["names"], _G["index"]
    best = None
    for choices in choices_list:
        assignments = {n: choices[index[n]] for n in names}
        plan, _ref = _evaluate(graph, assignments)
        if plan is None:
            continue
        key = (*plan.cost.objective, tuple(assignments[n] for n in sorted(assignments)))
        if best is None or key < best:
            best = key
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe Gemmini layout-plan enumeration over a captured model")
    parser.add_argument("--capture", type=Path, required=True, help="captured model directory")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    P = _load("p", root / "layout_graph_probe.py")
    G = _load("gmp", root / "group_model_program.py")
    from merlin.perf.physical_layout import _component_domains

    model = G.extract(args.capture, "gemmini")
    bk = P.derive_boundary_kinds("gemmini")
    sub = dict(model)
    sub["steps"] = model["steps"][:44]
    g = P.build_graph_from_groups(sub, max_assignments=10**12, boundary_kinds=bk)
    domains, index = _component_domains(g)
    names = [v.name for v in g.values]
    total = prod(len(d) for d in domains)
    workers = args.workers
    it = product(*domains)
    blocks = []
    while True:
        b = list(islice(it, 4096))
        if not b:
            break
        blocks.append(b)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(g, names, index)) as ex:
        results = [r for r in ex.map(_chunk, blocks) if r is not None]
    dt = time.perf_counter() - t0
    print(f"assignments={total} workers={workers} blocks={len(blocks)} parallel={dt:.2f}s")
    print("serial measured earlier = 80.09s  ->  speedup = %.1fx" % (80.09 / dt))
    print(
        "extrapolated FULL model (33,554,432 assignments) at this rate: %.1f hours" % (33_554_432 * (dt / total) / 3600)
    )
