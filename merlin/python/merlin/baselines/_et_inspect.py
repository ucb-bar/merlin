"""Post-process an ExecuTorch etdump into PER-REGION timing (runs UNDER the ExecuTorch venv).

The board run of the etdump-enabled ``executor_runner`` writes an ``etdump`` of per-op events; the
devtools ``Inspector`` correlates each event back to its originating nn.Module (``module_hierarchy``,
the SAME fqn key space Merlin's ``prov.fqn`` uses). This aggregates the per-op event times by that
fqn and emits ``et_regions.json`` — a list of ``{fqn, wall_ns, n_events, delegated}`` the merlin-side
runner (:mod:`.executorch`, in merlin's venv) turns into per-region ``RegionProfile``s (role via
``role_from_fqn``) for the region×framework compare. Dependency-light on merlin (argv in, JSON out),
mirroring ``_et_export``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# THIS SCRIPT SHADOWS THE PACKAGE IT NEEDS. It lives in ``merlin/baselines/``, which also contains
# ``executorch.py`` (merlin's OWN ExecuTorch arm), and Python puts a script's own directory first on
# sys.path. So ``from executorch.devtools import Inspector`` below resolved to that sibling module,
# which then failed importing merlin (absent from the ET venv) -- i.e. this helper could never run,
# and the whole per-op ExecuTorch timing path was unreachable. Drop our own directory before any
# executorch import; being argv-in/JSON-out, we need nothing from it.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path
               if p and os.path.abspath(p) != _HERE] or [p for p in sys.path if p]


def _clean_module_path(raw: str) -> str | None:
    """Extract the bare nn.Module path from an ExecuTorch event/module name.

    ExecuTorch names an op like ``aten_addmm_default_1_.L__self__layers.0.mlp`` (the ``L__self__``
    marker precedes the FX module path). Return the trailing module path (``layers.0.mlp``) — the SAME
    dotted key space Merlin's ``prov.fqn`` / ``role_from_fqn`` use — or None if there is no module tail."""
    if not raw:
        return None
    marker = "L__self__"
    tail = raw.split(marker, 1)[1] if marker in raw else raw
    tail = tail.lstrip("._")
    return tail or None


def _deepest_fqn(event) -> str | None:
    """The nn.Module path an op event came from.

    The FX module path rides in ``module_hierarchy`` as strings carrying the ``L__self__`` marker
    (e.g. ``aten_addmm_default_1_.L__self__layers.0.mlp``); other module_hierarchy strings are class /
    method names (``torch.nn.modules.linear.Linear``, ``Method::execute``) — noise, filtered out by
    requiring the marker. Delegated (XNNPACK) ops carry no such string (the delegate is opaque) ->
    None -> the caller buckets them as ``other`` (the honest delegation asymmetry)."""
    marked: list[str] = []

    def _collect(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and "L__self__" in k:
                    marked.append(k)
                _collect(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _collect(v)
        elif isinstance(obj, str) and "L__self__" in obj:
            marked.append(obj)

    _collect(getattr(event, "module_hierarchy", None))
    name = getattr(event, "name", "") or ""
    if "L__self__" in name:
        marked.append(name)
    cleaned = [c for c in (_clean_module_path(m) for m in marked) if c]
    return max(cleaned, key=lambda p: str(p).count(".")) if cleaned else None


def _event_wall_ns(event) -> float | None:
    """Best-available per-event time in ns from perf_data (median preferred, else mean/raw)."""
    pd = getattr(event, "perf_data", None)
    if pd is None:
        return None
    for attr in ("p50", "median", "avg", "mean"):
        v = getattr(pd, attr, None)
        if isinstance(v, (int, float)):
            return float(v) * 1e6            # Inspector default target scale is ms -> ns
    raw = getattr(pd, "raw", None)
    if raw:
        vals = [float(x) for x in raw if isinstance(x, (int, float))]
        if vals:
            return (sum(vals) / len(vals)) * 1e6
    return None


def inspect_etdump(etdump_path: str, etrecord_path: str | None) -> list[dict]:
    from executorch.devtools import Inspector

    kw = {"etdump_path": etdump_path, "enable_module_hierarchy": True}
    if etrecord_path:
        kw["etrecord"] = etrecord_path
    insp = Inspector(**kw)

    agg: dict[str, dict] = {}
    for block in insp.event_blocks:
        for event in block.events:
            wall = _event_wall_ns(event)
            if wall is None:
                continue
            fqn = _deepest_fqn(event) or "other"
            delegated = bool(getattr(event, "is_delegated_op", False))
            row = agg.setdefault(fqn, {"fqn": fqn, "wall_ns": 0.0, "n_events": 0, "delegated": delegated})
            row["wall_ns"] += wall
            row["n_events"] += 1
            row["delegated"] = row["delegated"] or delegated
    return sorted(({**r, "wall_ns": int(r["wall_ns"])} for r in agg.values()),
                  key=lambda r: -r["wall_ns"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--etdump", required=True)
    ap.add_argument("--etrecord", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    regions = inspect_etdump(args.etdump, args.etrecord or None)
    with open(args.out, "w") as fh:
        json.dump(regions, fh, indent=2)
    print("ET_REGIONS_JSON " + json.dumps({"out": args.out, "n_regions": len(regions),
                                           "total_wall_ns": sum(r["wall_ns"] for r in regions)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
