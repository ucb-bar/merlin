"""Hardware screens permit compiler scheduling choices without native tooling."""

import pytest

from merlin.targetgen import isa_taxonomy
from merlin.targetgen import rtl_check_compiler as compiler
from merlin.targetgen import rtl_check_runner as runner


@pytest.fixture
def taxonomy(monkeypatch):
    selected = {
        "complete_isa": True,
        "by_mnemonic": {
            "transfer": {"class": "TRANSFER", "fixed_mask": 255, "fixed_value": 16},
            "compute": {"class": "COMPUTE", "fixed_mask": 255, "fixed_value": 32},
        },
    }
    monkeypatch.setattr(isa_taxonomy, "taxonomy_for_target", lambda *_: selected)
    monkeypatch.setattr(
        isa_taxonomy, "role_classes", lambda *_: pytest.fail("a shared screen must not choose a schedule")
    )
    return selected


def capsule(rows=3, cols=5):
    return {
        "operation": {"op": "matmul"},
        "inputs": [{"role": "input", "shape": [rows, 7]}, {"role": "weight", "shape": [7, cols]}],
        "expected": {"instruction_classes": ["TRANSFER", "COMPUTE"]},
    }


def test_large_shapes_and_mesh_changes_do_not_prescribe_command_count(taxonomy):
    small = compiler.compile_kernel_checks(capsule(), target="synthetic_device")
    large = compiler.compile_kernel_checks(
        capsule(1009, 1021), facts_rec={"arrays": [{"name": "mesh", "rows": 13, "cols": 17}]}, target="synthetic_device"
    )
    assert small == large
    assert "CLASS_PRESENT TRANSFER" in small and "CLASS_PRESENT COMPUTE" in small
    assert "EMPTY_KERNEL no" in small and "ILLEGAL_OPCODE_COUNT 0" in small
    assert all(text not in small for text in ("CLASS_COUNT", "CLASS_ZEROOPS", "KORDER", "tiling"))


@pytest.mark.parametrize("words", [(16, 32), (32, 16), (16, 32, 32)])
def test_instruction_order_count_and_zero_payload_remain_measurements(taxonomy, words):
    rendered = runner.render_kernel_decode("\n".join(f".word {word}" for word in words), {}, taxonomy)
    assert "EMPTY_KERNEL no" in rendered and "ILLEGAL_OPCODE_COUNT 0" in rendered
    assert "CLASS_PRESENT TRANSFER" in rendered and "CLASS_PRESENT COMPUTE" in rendered
    assert "CLASS_ZEROOPS TRANSFER 1" in rendered
    # Presence and ISA legality are the shared assertions; schedule data stays observational.
    checks = compiler.compile_kernel_checks(capsule(), target="synthetic_device")
    assert "CLASS_COUNT" not in checks and "CLASS_ZEROOPS" not in checks and "KORDER" not in checks


def test_empty_and_illegal_kernel_diagnostics_are_retained(taxonomy):
    empty = runner.render_kernel_decode("ret", {}, taxonomy)
    illegal = runner.render_kernel_decode(".word 255", {}, taxonomy)
    assert "EMPTY_KERNEL yes" in empty
    assert "ILLEGAL_OPCODE_COUNT 1" in illegal
    assert "CLASS_PRESENT COMPUTE" not in illegal
    taxonomy["complete_isa"] = False
    assert "ILLEGAL_OPCODE_COUNT" not in compiler.compile_kernel_checks(capsule(), target="synthetic_device")


def test_provenance_does_not_launch_obsolete_role_probe(monkeypatch):
    from merlin.targetgen.rtl import mlc_bridge

    monkeypatch.setattr(
        mlc_bridge, "semantic_roles", lambda *_: pytest.fail("screen provenance must not launch a probe")
    )
    facts = {"interfaces": [{"name": "funct_decode_table", "complete_isa": False, "custom_opcode": 91}]}
    provenance = compiler._provenance(facts)
    assert set(provenance) == {"isa_legality", "abi_encoding"}
    assert provenance["isa_legality"]["derived"] is False


def test_runner_uses_only_hardware_assertion_prefix(taxonomy, monkeypatch, tmp_path):
    monkeypatch.setattr(compiler, "_endpoint_kind_for", lambda *_: "external_backend")
    monkeypatch.setattr(runner, "_load_capsule", lambda *_: capsule())
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "kernel.S").write_text(".word 16\n.word 32\n")
    calls = []
    monkeypatch.setattr(runner, "run_filecheck", lambda *args: calls.append(args) or (True, "owned control"))
    result = runner.screen_run(tmp_path, {}, {}, "synthetic-filecheck", target="synthetic_device")
    assert result["verdict"] == "ok" and calls[0][3] == "KERNEL"
