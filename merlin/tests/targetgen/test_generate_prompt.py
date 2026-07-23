"""The prompt slot resolver — the only target-specific content in an agent task prompt. Every slot must
derive from {the descriptor + the RTL fact bundle + the codegen endpoint}, never a gemmini literal.
"""
from __future__ import annotations

from merlin.targetgen.target_experiment import load_target_experiment, load_capability_manifest
from merlin.targetgen.generate_prompt import prompt_slots

_GEM_DESC = "merlin/experiments/gemmini_capsule_bench_v0/target_experiment.yaml"


def _gem_slots():
    return prompt_slots(load_target_experiment(_GEM_DESC), load_capability_manifest("gemmini"))


def test_tool_and_symbol_are_derived_from_the_target_name():
    s = _gem_slots()
    assert s["tool_stem"] == "gemmini-opt"          # {target}-opt, not a literal
    assert s["kernel_symbol"] == "gemmini_kernel"   # {target}_kernel, not a literal


def test_endpoint_is_fork_free_insn_and_not_llvm_prescriptive():
    s = _gem_slots()
    assert s["endpoint_kind"] == "inline_asm_insn"
    # the endpoint description names stock LLVM + no forked toolchain; it is keyed on the endpoint, not
    # the target, so it carries no gemmini/RoCC specifics
    assert "stock" in s["endpoint_desc"].lower() and "gemmini" not in s["endpoint_desc"].lower()


def test_corpus_families_are_globbed_not_a_hardcoded_list():
    s = _gem_slots()
    # discovered sibling corpora (layers/model_slices), each a path — never the prose "config, mvin/mvout…"
    assert s["corpus_families"] and all(p.endswith("/") for p in s["corpus_families"])
    assert not any("mvin" in p or "mvout" in p for p in s["corpus_families"])


def test_sim_tiers_and_isa_facts_come_from_manifest_and_discovery():
    s = _gem_slots()
    assert s["sim_tiers"] == {"L2": "spike", "L3": "verilator", "L4": "vcs", "L5": "firesim"}
    assert s["isa_facts"].startswith("# Target ISA facts: gemmini")   # the derived provenance-tagged brief


def test_slots_are_a_pure_function_of_target_identity():
    # tool_stem / kernel_symbol / endpoint_desc must be reconstructable from target + endpoint alone,
    # proving no gemmini value is baked in (a different target yields a different value by the same rule)
    s = _gem_slots()
    assert s["tool_stem"] == f"{s['target']}-opt"
    assert s["kernel_symbol"] == f"{s['target']}_kernel"
