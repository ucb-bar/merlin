"""The two shared ABI contracts carry NO target literal — they template the target-specific tokens with
the ``{target}`` placeholder, resolved at load by the contract readers. This pins that (a) the on-disk
yaml has zero ``gemmini`` literal, and (b) resolving for ``gemmini`` reproduces the former hand-authored
values byte-for-byte, mirroring ``test_generate_prompt`` (``kernel_symbol == f"{target}_kernel"``).
"""

from __future__ import annotations

import yaml

from merlin.targetgen.contract.schemas import (
    contract_dir,
    render_contract_text,
    render_legacy_kernel_abi,
    render_oracle_runner_contract,
)


def render_backend_contract(target: str) -> dict:
    return yaml.safe_load(render_contract_text("mlir_oot_backend_contract.yaml", target))


_GENERIC = ["mlir_oot_backend_contract.yaml", "oracle_runner_contract.yaml", "legacy/kernel_abi_v1.yaml"]


def test_generic_contracts_have_no_target_literal():
    for name in _GENERIC:
        text = (contract_dir() / name).read_text(encoding="utf-8")
        assert "gemmini" not in text, f"{name} still hardcodes a target literal (must template {{target}})"


def test_backend_contract_resolves_kernel_abi_to_the_target():
    c = render_backend_contract("gemmini")
    assert c["logical_kernel_abi"]["symbol"] == "gemmini_kernel"  # {target}_kernel, not a literal
    assert c["logical_kernel_abi"]["signature"].startswith("void gemmini_kernel(")
    # a different target resolves by the same rule (nothing baked in for gemmini)
    assert render_backend_contract("radiance")["logical_kernel_abi"]["symbol"] == "radiance_kernel"
    # the candidate-facing contract carries only the logical ABI
    assert "kernel_abi" not in c


def test_legacy_kernel_abi_resolves_to_the_target():
    abi = render_legacy_kernel_abi("gemmini")
    assert abi["symbol"] == "gemmini_kernel"
    assert abi["signature"].startswith("void gemmini_kernel(")
    assert render_legacy_kernel_abi("radiance")["symbol"] == "radiance_kernel"


def test_backend_entrypoint_argv_resolve_to_the_target():
    argv = render_backend_contract("gemmini")["entrypoints"]["lower_interface_to_target"]["example_argv"]
    assert "--convert-iface-to-gemmini" in argv


def test_oracle_runner_resolves_ladder_and_cycle_window_to_the_target():
    c = render_oracle_runner_contract("gemmini")
    names = {lvl["level"]: lvl["name"] for lvl in c["oracle_ladder"]}
    assert names[1] == "spike_gemmini_functional"  # spike_{target}_functional
    assert names[2] == "gemmini_verilator_rtl"  # {target}_verilator_rtl
    # levels carrying no {target} placeholder are unchanged
    assert names[0] == "merlin_reference_and_simulate" and names[3] == "firesim"
    notes = c["execution_artifact"]["harness_output_format"]["notes"]
    assert any("cycle_window = gemmini_region" in n for n in notes)


def test_contract_text_render_fills_placeholders_in_comments():
    # the parse_output reference lives in a YAML comment; the text renderer resolves it too
    text = render_contract_text("oracle_runner_contract.yaml", "gemmini")
    assert 'get_backend("gemmini").parse_output' in text
    assert "{target}" not in text
