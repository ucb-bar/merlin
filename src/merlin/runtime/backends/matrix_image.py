"""Build a whole-model image whose contractions are routed to a matrix extension, with its gate twin.

One implementation for every caller (``merlin-compile --matrix-unit ...`` and the delivery packager), so
the two cannot drift on what "a matrix image" means:

* the SHIPPED image, whose selected contractions call the support provider's kernel;
* the stand-in TWIN, the same lowering with the unit's datapath replaced by a scalar tile body, which a
  plain functional simulator can run and grade;
* optionally the DIAGNOSTIC build of the shipped image.

The pair is only meaningful if the twin is the same lowering, and if the shipped image really uses the
unit while the twin does not. Both are checked here, fail-closed, before anyone grades anything.

The matrix unit, its configuration and the executable support are parameters (``MatrixRouting``): no
unit, target or encoding is named in this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import zephyr_model as zm

#: Where :func:`routing_report` is written beside a shipped build.
ROUTING_REPORT = "routing_report.json"


@dataclass
class MatrixPair:
    ship: dict[str, Any]
    twin: dict[str, Any]
    debug: dict[str, Any] | None
    lowering_digest: str
    unit_instruction_counts: dict[str, int]
    debug_instruction_counts: dict[str, int]
    report: dict[str, Any]
    ship_dir: Path


class NothingRouted(RuntimeError):
    """The selector routed no contraction of this model to the unit, so there is no matrix image to ship.

    Not a build failure: every contraction was declined for a stated reason (typically smaller than one
    tile -- a decode-length token count, say), and a "matrix" image would be the RVV image under another
    name. Carries the routing report (also written beside the build) so a caller can say WHY rather than
    only that the image carries none of the unit's instructions.
    """

    def __init__(self, unit: str, report: dict[str, Any], report_path: Path) -> None:
        self.report, self.report_path = report, report_path
        reasons = "; ".join(f"{n}x {why}" for why, n in (report.get("not_routed_by_reason") or {}).items())
        example = next((str(r.get("why")) for r in report.get("not_routed") or []), "")
        super().__init__(
            f"nothing routed to {unit}: {report.get('not_routed_contractions', 0)} contraction(s) declined "
            f"({reasons or 'no contraction found'}" + (f", e.g. {example})" if example else ")")
        )


def _audit(elf: Path, encodings: dict) -> dict[str, int]:
    from merlin.kernels.decode import opu as unit_audit

    counts = unit_audit.audit_object(elf, encodings).counts or {}
    # Count by VALUE: the audit returns an entry per instruction whether or not it occurs.
    return {k: int(v) for k, v in sorted(counts.items()) if v}


def _lowering_digest(build_dir: Path) -> str:
    h = hashlib.sha256()
    h.update((build_dir / "model.o").read_bytes())
    h.update((build_dir / "cgen" / "weights.bin").read_bytes())
    return h.hexdigest()[:16]


def _selected_kernel(routing: Any):
    """The routing's matrix kernel (``merlin.kernels.matrix_kernels.for_routing``): its name, file, sha256."""
    from merlin.kernels.matrix_kernels import for_routing

    return for_routing(routing)


def _kernel_record(routing: Any) -> dict[str, Any]:
    return _selected_kernel(routing).record()


def routing_report(build_dir: Path, routing: zm.MatrixRouting) -> dict[str, Any]:
    """What THIS build routed and what it left on the vector unit, read from the rewrite's own sidecar.

    ``opu_mac_share`` is the fraction of the model's contraction MACs that run on the unit: the number that
    bounds any speedup (Amdahl), and the first thing to read before expecting one.
    """
    zm.load_matrix_signatures(Path(build_dir), routing)
    sidecar = Path(build_dir) / routing.provider().SIDECAR_NAME
    if not sidecar.is_file():
        return {"routed_contractions": 0, "note": "no routing sidecar: nothing was routed"}
    got = json.loads(sidecar.read_text(encoding="utf-8"))
    routed = got.get("routed") or []

    def macs(r: dict) -> int:
        return int(r.get("b") or 1) * int(r["m"]) * int(r["n"]) * int(r["k"])

    routed_macs = sum(macs(r) for r in routed)
    left = got.get("not_routed") or []
    left_macs = int((got.get("macs") or {}).get("not_routed") or 0)
    from merlin.llvmlower.int8_contractions import GROUPED_REASONS

    reasons: dict[str, int] = {}
    for item in left:
        why = str(item.get("why") or "unknown")
        # Grouped by the reason's class, which keeps the tile edge ("smaller than one tile (64)"): the
        # class alone, "declined by the selector", told a reader nothing about what to change.
        key = why.partition(":")[0] if why.startswith(GROUPED_REASONS) else why
        reasons[key] = reasons.get(key, 0) + 1
    widest = max(routed, key=macs, default=None)
    total = routed_macs + left_macs
    from merlin.llvmlower.matrix_route_report import by_site, numeric_policy, priced_routes

    return {
        "support_target": routing.support_target,
        "unit": routing.unit,
        "config": routing.config,
        # Which kernel every routed call runs, and the sha256 of its file (a hand-written one), so a report
        # read later says which bytes computed the routed contractions.
        "kernel": _kernel_record(routing),
        "tile_edge": (got.get("routing_rule") or {}).get("tile_edge"),
        "policy": (got.get("routing_rule") or {}).get("policy"),
        # Where the MACs sit (conv / attention / linear, read from provenance) and how much of each the
        # unit takes; and every route the rule above would have refused, with the decision that took it.
        "macs_by_site": by_site(routed, left),
        "priced_sub_tile_routes": priced_routes(routed),
        "numeric_policy": numeric_policy(build_dir),
        "routed_contractions": int(got.get("count") or len(routed)),
        "distinct_signatures": len(got.get("signatures") or {}),
        "not_routed_contractions": len(left),
        "not_routed_by_reason": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "macs_routed": routed_macs,
        "macs_not_routed": left_macs,
        "opu_mac_share": (routed_macs / total) if total else None,
        "skipped": [f"{d.get('what')}: {d.get('why')}" for d in (got.get("skipped") or [])],
        "widest_routed": (
            None
            if widest is None
            else {"fqn": widest.get("fqn"), "shape": [widest.get("b"), widest["m"], widest["n"], widest["k"]]}
        ),
        "not_routed": left,
    }


def build_matrix_pair(
    bundle: Path,
    work: Path,
    *,
    board: str,
    harts: int,
    cpus: int,
    routing: zm.MatrixRouting,
    features: frozenset[str],
    rvv_schedule: str,
    cflags: list[str],
    vlen: int | None = None,
    sdk_dir: Path | None = None,
    debug: bool = False,
    iters: int = 1,
    warmup: int = 0,
) -> MatrixPair:
    """Build the shipped image, its stand-in twin and (optionally) the diagnostic build, and check them.

    Raises :class:`NothingRouted` when the selector routed no contraction (its report says why);
    ``RuntimeError`` when the twin is not the same lowering, when the shipped (or diagnostic) image
    carries none of the unit's instructions though contractions were routed, or when the twin carries any;
    ``BoardRegistryError`` when ``harts`` would reach a hart the board does not list among its ``matrix_harts``.
    """
    # A routed image splits its tile loop over every worker, so each must sit on a hart that HAS the unit
    # (a heterogeneous SoC can give a hart a vector unit and no matrix unit). Refused before any build;
    # the twin too, since it exists only to gate an image that could ship.
    from ..boards import board as _board

    _board(board).check_matrix_image(harts)
    # The unit's encodings and geometry are derived from pinned checkouts. Off the pin in a byte the
    # derivation reads, the image would be about other hardware: refused (UnitPinError) unless allowed.
    from merlin.targetgen import unit_pins

    unit_pins.require(routing.support_target, routing.unit)
    # The kernel is resolved (its file read and hashed) before anything is built, so a typo in a kernel name
    # or a missing file costs a second rather than a whole-model lowering.
    kernel = _selected_kernel(routing)
    provider = routing.provider()
    encodings = provider.derive_encodings(provider.load_contract(routing.unit)).encodings

    def build(*, stand_in: bool, dbg: bool, where: str) -> dict:
        return zm.build_app(
            bundle,
            work / where,
            board=board,
            backend="rvv",
            rvv_hart=0,
            cpus=cpus,
            int8_compute=True,
            features=features,
            n_harts=harts,
            sdk_dir=sdk_dir,
            debug=dbg,
            matrix=routing,
            matrix_scalar_tile=stand_in,
            rvv_schedule=rvv_schedule,
            cflags_override=cflags,
            vlen=vlen,
            iters=iters,
            warmup=warmup,
        )

    ship_where, twin_where = f"board_h{harts}", f"twin_h{harts}"
    ship = build(stand_in=False, dbg=False, where=ship_where)
    ship_dir = work / ship_where
    counts = _audit(ship["elf"], encodings)
    report = routing_report(ship_dir, routing)
    (ship_dir / ROUTING_REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not counts and not report.get("routed_contractions"):
        raise NothingRouted(routing.unit, report, ship_dir / ROUTING_REPORT)  # before building a twin for it
    if not counts:
        raise RuntimeError(
            f"the shipped image contains NONE of {routing.unit}'s instructions, though the routing report "
            f"says {report.get('routed_contractions')} contraction(s) were routed"
            + (
                ""
                if kernel.generated
                else f": the selected kernel {kernel.name!r} ({kernel.entry.file}) issues none of the unit's "
                "instructions, so it would compute on the host core and prove nothing about the unit"
            )
        )
    twin = build(stand_in=True, dbg=False, where=twin_where)
    digest, twin_digest = _lowering_digest(ship_dir), _lowering_digest(work / twin_where)
    if digest != twin_digest:
        raise RuntimeError(
            f"the stand-in twin lowers to {twin_digest} against the shipped image's {digest}; they are not "
            "the same lowering, so grading the twin would say nothing about what ships"
        )
    twin_counts = _audit(twin["elf"], encodings)
    if twin_counts:
        raise RuntimeError(f"the stand-in twin contains the unit's instructions ({twin_counts}); it is not a stand-in")
    dbg = build(stand_in=False, dbg=True, where=f"board_h{harts}_dbg") if debug else None
    dbg_counts = {} if dbg is None else _audit(dbg["elf"], encodings)
    if dbg is not None and not dbg_counts:
        raise RuntimeError(
            f"the diagnostic build contains NONE of {routing.unit}'s instructions, so the build someone falls "
            "back to on a bad run would not exercise the unit"
        )
    return MatrixPair(ship, twin, dbg, digest, counts, dbg_counts, report, ship_dir)
