"""Retained verification reopens metadata without prior backend/compiler initialization."""

from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import private_static_products as products
from merlin_experiments.phase1.feedback import private_static_revalidation as replay

from merlin.llvmlower.device_build import executable_kernel_entry
from merlin.runtime.backends import base
from merlin.runtime.backends import chipyard_rocc_bound as bound
from merlin.targetgen import target_registry
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64
from merlin.targetgen.rtl import facts


@pytest.fixture
def selected(tmp_path, monkeypatch):
    target = "retained_scope_unit"
    tool = tmp_path / "never-executed"
    tool.write_text("#!/bin/sh\nexit 97\n")
    tool.chmod(0o700)
    layout = tmp_path / "layout.ld"
    layout.write_text("ENTRY(_start)\nSECTIONS { .text : { *(.text*) } }\n")

    def unit(name, roster):
        return {"id": name, "sha256": products.pin(bound._resource(name, roster))["sha256"]}

    contract = {
        "name": target,
        "runner": {
            "backend": "chipyard_rocc",
            "chipyard_rocc": {
                "version": 1, "config": "selected_config",
                "toolchain": {
                    "compiler": products.pin(tool),
                    "runtime_units": [unit(n, bound._RUNTIME) for n in ("startup", "htif_console")],
                    "headers": [unit(n, bound._HEADERS) for n in ("htif", "out_b64")],
                    "link_script": products.pin(layout), "load_address": 4096,
                    "cflags": ["-march=rv64gc", "-mabi=lp64d", "-ffreestanding"], "ldflags": [],
                    "kernel_stack_frame": {"entry_symbol": "selected_entry", "max_static_bytes": 8192},
                },
                "engines": {"spike": {
                    "binary": products.pin(tool), "argv": ["--isa=rv64gc", "{elf}"],
                    "environment": {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}, "cwd": str(tmp_path),
                    "max_timeout_s": 5, "max_console_bytes": 4096, "failure_markers": ["FAILURE"],
                }},
            },
            "spike_extension": {
                "extension_name": "selected_extension", "extlib": str(tool),
                "sha256": products.pin(tool)["sha256"],
            },
        },
        "harness_abi": {
            "version": 2, "kind": "logical_pointer", "entry_symbol": "selected_entry",
            "fence_symbol": None, "tensor_alignment": 8, "byte_order": "little",
            "main_convention": "void", "readback_transport": FULL_VALUES_B64,
        },
        "rocc_operand_roles": {
            "version": 1, "word_bits": 64,
            "instructions": [{"funct": 1, "class": "TRANSPORT", "operands": {}}],
        },
        "rtl_checks": ["decode_clean", "legal_funct"],
    }
    original_facts = {
        "inputs": {"target": target, "fir_sha256": "f" * 64},
        "source_consistency": {"status": "unverified", "config": "selected_config"},
        "facts": {"source": {"config": "selected_config"}, "interfaces": [{
            "name": "funct_decode_table", "legal_funct": [1], "custom_opcode": 11, "funct3": 7,
            "scope": "complete_rocc_funct7", "complete_isa": True,
        }]},
    }
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(tmp_path / "no-target-providers"))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    monkeypatch.setattr(facts, "load_facts", lambda *a, **k: original_facts)
    monkeypatch.setattr(base, "_ensure_discovered", lambda: pytest.fail("provider discovery is not authorized"))
    capability = tmp_path / "capability.yaml"
    capability.write_text(yaml.safe_dump(contract))
    spec = tmp_path / "private.yaml"
    document = {
        "target": target, "models": [{"id": "network", "capability_contract": str(capability),
                                    "capability_contract_sha256": products.pin(capability)["sha256"]}],
    }
    spec.write_text(yaml.safe_dump(document))
    record = {
        "schema": replay.SCHEMA, "scope": replay.SCOPE, "target": target, "selected_model": "network",
        "private_spec_sha256": products.pin(spec)["sha256"],
        # Actual revalidation repeats old input pins; repetition is not ambiguity.
        "inputs": [products.pin(spec), products.pin(spec)],
    }
    return SimpleNamespace(record=record, target=target, capability=capability, spec=spec, document=document)


def test_revalidation_verifier_selects_actual_metadata_backend_without_prior_compile(selected, monkeypatch):
    seen = []

    def custody(record):
        backend = base.get_backend(record["target"])
        assert type(backend) is bound.BoundRoCCBackend
        seen.append(executable_kernel_entry(record["target"]))
        assert target_registry.resolve(record["target"]).capability_contract_path == selected.capability

    monkeypatch.setattr(replay, "_verify_record", custody)
    assert selected.target not in base._REGISTRY
    replay.verify_record(selected.record)
    replay.verify_record(selected.record)
    assert seen == ["selected_entry", "selected_entry"]
    assert selected.target not in base._REGISTRY
    assert selected.target not in target_registry._OBSERVED_CONTRACTS.get()


@pytest.mark.parametrize("damage", ["missing_spec_pin", "spec_bytes", "missing_contract_pin", "contract_bytes",
                                  "contract_target", "model", "spec_target"])
def test_missing_or_mismatched_original_selection_refuses_before_custody(selected, monkeypatch, damage):
    monkeypatch.setattr(replay, "_verify_record", lambda _: pytest.fail("unbound custody must not run"))
    if damage == "missing_spec_pin":
        selected.record["inputs"] = []
    elif damage == "spec_bytes":
        selected.spec.write_text("{}")
    elif damage == "contract_bytes":
        selected.capability.write_text("{}")
    else:
        if damage == "missing_contract_pin":
            selected.document["models"][0].pop("capability_contract_sha256")
        elif damage == "contract_target":
            data = yaml.safe_load(selected.capability.read_bytes())
            data["name"] = "another_target"
            selected.capability.write_text(yaml.safe_dump(data))
            selected.document["models"][0]["capability_contract_sha256"] = products.pin(selected.capability)["sha256"]
        elif damage == "model":
            selected.document["models"] = []
        elif damage == "spec_target":
            selected.document["target"] = "another_target"
        selected.spec.write_text(yaml.safe_dump(selected.document))
        selected.record.update(inputs=[products.pin(selected.spec)],
                               private_spec_sha256=products.pin(selected.spec)["sha256"])
    with pytest.raises(ValueError):
        replay.verify_record(selected.record)
    assert selected.target not in target_registry._OBSERVED_CONTRACTS.get()


def test_contract_changed_during_verification_refuses_and_restores_scope(selected, monkeypatch):
    def custody(record):
        assert executable_kernel_entry(record["target"]) == "selected_entry"
        selected.capability.write_text("{}")

    monkeypatch.setattr(replay, "_verify_record", custody)
    with pytest.raises(ValueError, match="changed"):
        replay.verify_record(selected.record)
    assert selected.target not in target_registry._OBSERVED_CONTRACTS.get()


def test_selected_contract_restores_existing_caller_selection(selected, monkeypatch):
    original = {"name": selected.target, "caller_selection": "must survive"}
    monkeypatch.setattr(replay, "_verify_record",
                        lambda record: executable_kernel_entry(record["target"]))
    with target_registry.observed_contract(selected.target, original, source_path=selected.spec):
        assert replay.verify_record(selected.record) == "selected_entry"
        info = target_registry.resolve(selected.target)
        assert info.load_contract() == original and info.capability_contract_path == selected.spec
    assert selected.target not in target_registry._OBSERVED_CONTRACTS.get()
