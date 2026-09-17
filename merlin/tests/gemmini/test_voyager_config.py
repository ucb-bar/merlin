"""Voyager's machine model for the systolic target, derived from the target's facts (never typed in).

Target-specific by design: the expected numbers are the pinned systolic target's derived address space
(16x16 array; 256 KiB scratchpad in 4 banks of 16-byte rows; 1024 accumulator rows), and the point of
the test is that the Voyager config is read from there -- so a revision that changes the facts changes
the config with it, and a fact that goes missing is a refusal rather than a default.
"""

from __future__ import annotations

import pytest

from merlin.baselines.voyager import VoyagerConfigError, accelerator_config_for
from merlin.targetgen.address_space import derive_address_space

TARGET = "gemmini"  # target-ok: this test is about the pinned systolic target's derived facts


def test_the_config_is_the_derived_address_space() -> None:
    derived = accelerator_config_for(TARGET)
    space = derive_address_space(TARGET)
    spad, acc = space.store("scratchpad"), space.store("accumulator")
    assert derived.fields == {
        "pe_array_size": [space.array_rows, space.array_cols],
        "scratchpad_size": spad.nbytes,
        "num_banks": spad.banks,
        "bank_width": spad.row_bytes,
        "input_buffer_size": spad.total_rows,
        "weight_buffer_size": spad.total_rows,
        "accum_buffer_size": acc.total_rows,
    }
    # The sensitivity reading counts only the PE-resident block as weight L1; it must be opt-in.
    variant = accelerator_config_for(TARGET, weight_residency="pe_block").fields
    assert variant["weight_buffer_size"] == space.array_rows
    assert {k: v for k, v in variant.items() if k != "weight_buffer_size"} == {
        k: v for k, v in derived.fields.items() if k != "weight_buffer_size"
    }
    with pytest.raises(VoyagerConfigError):
        accelerator_config_for(TARGET, weight_residency="guess")
    # And those facts are the ones every other consumer of this target reads.
    assert derived.fields["pe_array_size"] == [16, 16]
    assert (derived.fields["scratchpad_size"], derived.fields["num_banks"], derived.fields["bank_width"]) == (
        262144,
        4,
        16,
    )
    assert derived.fields["accum_buffer_size"] == 1024
    assert set(derived.sources) == set(derived.fields)
    assert "dram_bandwidth" in derived.not_modelled


def test_the_bridge_geometry_is_the_same_facts_as_the_config() -> None:
    from merlin.baselines.voyager import geometry_for

    config = accelerator_config_for(TARGET).fields
    geometry = geometry_for(TARGET)
    assert geometry.dim == config["pe_array_size"][0]
    assert geometry.spad_rows * geometry.spad_row_bytes == config["scratchpad_size"]
    assert geometry.acc_rows == config["accum_buffer_size"]


def test_a_missing_store_is_refused_not_defaulted() -> None:
    facts = {"facts": {"arrays": [{"name": "mesh", "rows": 16, "cols": 16}], "memories": [], "datapaths": []}}
    with pytest.raises(VoyagerConfigError):
        accelerator_config_for(TARGET, facts=facts)
