"""The operand-direction cache must be named for the oracle that produced it.

`derive_direction` probes through `run_program_debug`, which is the target's FUNCTIONAL core; the `tier`
argument only picks whose settle contract the preamble uses. Naming the cache after the tier produced
`operand_direction_vsim.json` -- a file whose name claims the RTL tier while its content is
model-derived.

That mattered concretely. The functional model implements the shipped architectural spec, so where a
spec and its RTL disagree the cache records the spec's answer. One such entry -- a zero-fill immediate
recorded as writing only its named register, where the elaborated design clears the whole register file
-- was read as an RTL fact.
"""
from __future__ import annotations

import sys

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments/performance_contract"))
import dependence as D  # noqa: E402


def test_the_cache_is_named_for_the_functional_oracle_not_the_tier():
    name = D.direction_path("someTarget", "vsim").name
    assert "functional" in name, f"name must say what produced it: {name}"
    assert not name.startswith("operand_direction_vsim"), "must not read as an RTL-tier artifact"


def test_the_settle_tier_is_still_distinguishable():
    """Two tiers give two caches -- the settle contract genuinely differs, so they must not collide."""
    a = D.direction_path("t", "vsim").name
    b = D.direction_path("t", "spike").name
    assert a != b and "vsim" in a and "spike" in b


def test_the_probe_really_does_run_on_the_functional_core():
    """Guards the premise: if the probe ever routes to an RTL engine, this name becomes the wrong one."""
    import inspect
    src = inspect.getsource(D.derive_direction)
    assert "run_program_debug" in src, "probe no longer uses the functional debugger; revisit the name"
