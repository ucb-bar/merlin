"""``merlin-surface`` — emit a target's optimization surface, or read it.

Repeatable on purpose. The surface is DERIVED from the live registries, so it goes stale the moment
a seam, a route or a CCA axis changes; a report someone wrote once would describe a compiler that no
longer exists.
"""
from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True, help="the target whose surface is assembled")
    ap.add_argument("--json", action="store_true", help="emit the full surface as JSON")
    ap.add_argument("--write", action="store_true",
                    help="write a versioned artifact under out/artifacts/optimization-surface/")
    ap.add_argument("--gaps-only", action="store_true", help="list only the declared gaps")
    a = ap.parse_args(argv)

    from merlin.kernels.surface import build

    s = build(a.target)
    if a.write:
        from merlin.common.artifacts import new_product
        p = new_product("optimization-surface", version=1, target=a.target,
                        notes="derived from kernels.regions + action_catalog + cca_contract")
        (p.path / "manifest.json").write_text(json.dumps(s.to_dict(), indent=1))
        print(f"wrote {p.path}")

    if a.json:
        print(json.dumps(s.to_dict(), indent=2))
        return 0

    shown = s.gaps() if a.gaps_only else s.entries
    print(f"target: {a.target}   seams={len(s.entries)}  forkable={len(s.forkable())}  "
          f"gaps={len(s.gaps())}")
    for e in shown:
        flag = "" if e.forkable_now else "  [GAP: no registrable hook yet]"
        print(f"  {e.seam_id:52s} {e.phase:15s} {e.mechanism:9s} {e.scope:8s}"
              f" rebuild={e.rebuild_scope}{flag}")
        if e.cca_axes:
            print(f"      axes: {', '.join(e.cca_axes)}")
    if s.ungoverned:
        # Reported, never dropped: the place exists and no CCA axis reaches it, so nothing can route
        # a divergence to it even though it is editable.
        print(f"  {len(s.ungoverned)} seam(s) govern no CCA axis: {', '.join(s.ungoverned[:6])}"
              + (" ..." if len(s.ungoverned) > 6 else ""))
    for n in s.notes:
        print(f"  ! {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
