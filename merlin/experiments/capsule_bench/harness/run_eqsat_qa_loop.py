"""Isolated EQUIVALENCE-SEAM agentic track — the merlin_assisted loop plus the e-graph seam.

Built on the same in-process pattern as :mod:`run_rtlchecks_qa_loop`: the baseline QA loop and the
merlin_assisted arm are reused UNMODIFIED, and exactly one thing changes — the bundle served for the
merlin arm becomes ``merlin_assisted_eqsat_*``, whose only difference from ``merlin_assisted_*`` is that
the equivalence seam (``contraction_egraph`` + ``persistent_equivalence``) is granted. Nothing else moves,
so a difference in outcome attributes to the seam rather than to a second changed variable.

What the seam IS, stated plainly because it is easy to overclaim: the agent's own implementation is
registered as an alternative inside an e-class alongside the linalg form, and the extractor picks one. So
what runs and what the capsules grade is whatever was EXTRACTED. That makes arm5 evidence about agent
productivity given the seam — never evidence that equational saturation is a better compiler strategy.
The two hypotheses that would say the latter (H-EQ1, H-EQ2) remain ``not_established`` and the tests that
pin them must stay green.

Usage mirrors the other loops::

    run_eqsat_qa_loop.py --run-id eqsat_0001 --model claude-opus-4-8 [--max-rounds 6] ...
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_baseline_qa_loop as L  # noqa: E402
from merlin_experiments.phase1.options import parse_options  # noqa: E402

_EQSAT_BUNDLE = "merlin_assisted_eqsat_public_v0"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = parse_options(argv, default_arm="merlin_assisted")
    if not any(arg == "--arm" or arg.startswith("--arm=") for arg in argv):
        argv += ["--arm", selected.arm]
    if selected.arm == "merlin_assisted" and not selected.bundle:
        argv += ["--bundle", _EQSAT_BUNDLE]
    return L.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
