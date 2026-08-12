"""The ``harness_abi`` contract block: the four facts a runner needs and cannot derive.

The generic compile path used to write one target's entry symbol, fence symbol, include and metric
name as literals, which meant a target spelling any of them differently could not use that path at
all. They are genuinely underivable — no fact bundle says a kernel is called ``foo_kernel`` rather
than ``bar_entry`` — so they belong in a contract the target authors.

What is tested here is mostly the REFUSALS. A missing or malformed block must raise, because the
failure a default produces is uniquely bad: the harness compiles, then fails to link, and the error
names the linker rather than the absent declaration.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.contract import harness_abi as HA


def _block(**over):
    base = {"entry_symbol": "npu_kernel", "fence_symbol": "npu_fence",
            "includes": ["include/npu_testutils.h"], "cycle_window_metric": "cycle_window_npu"}
    base.update(over)
    return {"harness_abi": base}


# ------------------------------------------------------------------ refusals
def test_a_contract_with_no_harness_abi_is_refused():
    with pytest.raises(HA.HarnessAbiError, match="declares no `harness_abi`"):
        HA.from_contract({}, target="synth_npu")


def test_an_entry_symbol_is_required():
    with pytest.raises(HA.HarnessAbiError, match="entry_symbol is required"):
        HA.from_contract({"harness_abi": {"includes": ["x.h"]}}, target="synth_npu")


def test_a_non_string_optional_field_is_refused_rather_than_coerced():
    with pytest.raises(HA.HarnessAbiError, match="fence_symbol"):
        HA.from_contract(_block(fence_symbol=17), target="synth_npu")


# ------------------------------------------------------------------ rendering
def test_the_rendered_harness_names_only_the_declaring_target():
    abi = HA.from_contract(_block(), target="synth_npu")
    rendered = "\n".join([abi.declarations(), abi.call("(void*)a, (void*)b"), abi.cycle_window_line()])
    assert "npu_kernel((void*)a, (void*)b)" in rendered
    assert "npu_fence();" in rendered
    assert '#include "include/npu_testutils.h"' in rendered
    assert "METRIC cycle_window_npu 1" in rendered


def test_a_target_with_no_fence_emits_no_fence_call():
    abi = HA.from_contract(_block(fence_symbol=None), target="synth_npu")
    assert abi.call("x") == "  npu_kernel(x);"


def test_a_target_with_no_cycle_window_metric_emits_no_metric_line():
    """Not the same as emitting it as 0. The runner's parser treats an absent key and a falsy value
    identically, so a 0-valued line is noise carrying another target's vocabulary."""
    abi = HA.from_contract(_block(cycle_window_metric=None), target="synth_npu")
    assert abi.cycle_window_line() == ""


def test_extern_decls_default_to_declaring_the_entry_symbol():
    abi = HA.from_contract(_block(), target="synth_npu")
    assert "extern void npu_kernel();" in abi.declarations()
    explicit = HA.from_contract(_block(extern_decls=["extern int npu_kernel(void*);"]),
                                target="synth_npu")
    assert "extern int npu_kernel(void*);" in explicit.declarations()
    assert "extern void npu_kernel();" not in explicit.declarations()


# ------------------------------------------------------------------ the shipped contract
def test_the_reference_target_declares_a_resolvable_harness_abi():
    """The regression for the migration: these four values were literals in the generic path, so the
    contract must now carry them or that path silently loses its harness."""
    abi = HA.for_target("gemmini")
    assert abi.entry_symbol and abi.fence_symbol and abi.includes and abi.cycle_window_metric
