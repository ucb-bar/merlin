"""A Phase-0 software-spec selection resolved against FIXTURE facts, plus an optional real bundle.

The fixture facts are the minimal mesh/datapath/readout views the admission tests need. They stand in
for a facts bundle, so a test built on them pins a screening rule, not a target's fact derivation.
When ``MERLIN_RTL_FACTS`` names a real facts bundle, :func:`real_selection` resolves the same spec
against it through the production ``select_spec`` path instead.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

RAW = {
    "facts": {
        "arrays": [{"name": "mesh", "rows": 16, "cols": 16, "corroborated": True, "mac_idiom": {"muls": 1, "adds": 1}}],
        "datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
        "interfaces": [{"name": "mesh_dma"}],
        "storage_datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
    }
}
FACETS = [
    {
        "unit": "systolic_mesh",
        "accumulator_dtype": "i32",
        "readouts": [
            {"selector": "i8", "applies": ["acc_scale", "relu", "maxpool"], "evidence": "fixture"},
            {"selector": "i32", "applies": [], "evidence": "fixture"},
        ],
        "operand_sum": {"operands": 2},
        "unknown": {},
        "scale": {"granularities": ["tensor"], "granularities_complete": True, "carriers": []},
    }
]
QUANT = [
    {
        "status": "derived",
        "unit": "systolic_mesh",
        "format": "int8",
        "recipe": {"families": ["contraction", "operand_sum", "window_mean"]},
    }
]


def fixture_resolved(target_dir: Path, target: str) -> dict:
    """The target's authored software spec, its hardware forms filled from the fixture facts."""
    from merlin.common.yaml import load_yaml
    from merlin.targetgen import spec_fact_drift as D
    from merlin.targetgen.software_spec import load_software_spec

    contract = load_yaml(target_dir / "contracts/target_contract.yaml")
    facts = D.fact_capabilities(
        target=target,
        contract=contract,
        raw_facts=RAW,
        readout_facets=FACETS,
        quantization_candidates=QUANT,
        taxonomy={},
        semantic_facts={"facts": {"accumulate_on_load": {"status": "derived"}}},
    )
    spec, _ = D.resolve_spec(load_software_spec(target_dir / "software-spec.yaml", target=target), facts)
    return spec


def real_selection(target_dir: Path, target: str, host_capabilities: Path | None = None) -> dict | None:
    """``select_spec`` against the bundle ``MERLIN_RTL_FACTS`` names, or None when none is selected.

    ``host_capabilities`` names the host spec when it does not live beside the software spec in
    ``target_dir`` (it defaults to ``target_dir / "host-capabilities.yaml"``).
    """
    selected = os.environ.get("MERLIN_RTL_FACTS")
    if not selected or not Path(selected).is_file():
        return None
    from merlin.targetgen.core_aten_spec_int8 import select_spec
    from merlin.targetgen.rtl.facts import observed_facts

    raw = json.loads(Path(selected).read_text())
    with observed_facts(target, raw):
        host = host_capabilities if host_capabilities is not None else target_dir / "host-capabilities.yaml"
        return select_spec(target, raw, target_dir / "software-spec.yaml", host)
