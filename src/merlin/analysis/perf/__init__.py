"""The performance layer: what a target's legal choices cost.

Every analysis here is generic and gated on DERIVED TRAITS, never on a target name and never on an
archetype label -- an archetype is only a prior about which questions to ask. See AGENT.md.
"""

from __future__ import annotations

from pkgutil import extend_path

# ---- perf --------------------------------------------------------------------------------------

# Optional execution and benchmark-scoring modules keep their historical import names.

__path__ = extend_path(__path__, __name__)
