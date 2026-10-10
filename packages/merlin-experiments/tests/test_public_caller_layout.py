"""The author sees a bounded physical layout receipt, never caller values or grader state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback.caller_layout import _checked_projection, inspect_caller_layout

from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.runtime.backends import base as backends
from merlin.targetgen import plugins


@pytest.fixture
def selected_provider(tmp_path, monkeypatch):
    """An isolated selected renderer; real OOT C parity lives in its own suite."""
    root = tmp_path / "support"
    root.mkdir()
    source = root / "renderer.py"
    source.write_text("LAYOUT_POLICY = 4\n")
    contract = root / "contract.yaml"
    contract.write_text("plugin: renderer\n")
    (root / "provider.yaml").write_text("role: support\n")
    info = SimpleNamespace(base=root, contract_path=contract)

    def describe(cb, *, target, facts):
        assert target == cb["target"] == facts["inputs"]["target"]
        return {
            "schema": "caller_storage_layout_v1",
            "policy": {"mode": "legacy_aligned_row_major_v1", "row_alignment_elements": 4},
            "tensors": [
                {
                    "tensor": name,
                    "dtype": spec["dtype"],
                    "logical_shape": spec["shape"],
                    "physical_extents": [4, 4],
                    "logical_strides_elements": [4, 1],
                    "storage_elements": 16,
                    "offset_elements": 0,
                }
                for name, spec in cb["tensors"].items()
            ],
        }

    module = SimpleNamespace(
        __file__=str(source), caller_layout_source_paths=lambda: (source,), describe_caller_layout=describe
    )
    monkeypatch.setattr(plugins, "resolve_support", lambda target: info)
    monkeypatch.setattr(backends, "get_backend", lambda target: module)
    return info


def _fixture(tmp_path: Path) -> tuple[Path, Path, bytes]:
    submission = tmp_path / "submission"
    submission.mkdir()
    command = {
        "target": "gemmini",
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "A", "access": "read"}, {"tensor": "Y", "access": "write"}],
            "outputs": ["Y"],
        },
        "tensors": {
            "A": {"shape": [2, 3], "dtype": "i8", "role": "input"},
            "Y": {"shape": [2, 3], "dtype": "i8", "role": "output"},
        },
        "commands": [],
        "params": {},
    }
    raw = json.dumps(command, sort_keys=True).encode()
    (submission / "command_buffer.json").write_bytes(raw)
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"inputs": {"target": "gemmini"}, "facts": {"arrays": [{"name": "mesh", "cols": 16}]}}))
    return submission, facts, raw


def test_selected_harness_receipt_is_value_free_and_bound_to_exact_bytes(tmp_path, selected_provider):
    submission, facts, raw = _fixture(tmp_path)
    receipt = inspect_caller_layout(
        submission=submission, command_buffer_member="command_buffer.json", target="gemmini", facts_path=facts
    )
    assert receipt["status"] == "layout_only"
    assert receipt["command_buffer_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["rtl_facts_sha256"] == hashlib.sha256(facts.read_bytes()).hexdigest()
    assert len(receipt["provider_sha256"]) == len(receipt["policy_sha256"]) == 64
    assert receipt["tensors"][0]["logical_strides_elements"] == [4, 1]
    assert receipt["tensors"][1]["physical_extents"] == [4, 4]
    assert not any(key in receipt for key in ("all_pass", "expected", "golden", "inputs", "values"))
    assert all(not any(key in row for key in ("weight", "input", "value", "expected")) for row in receipt["tensors"])


def test_layout_probe_refuses_external_paths_and_unselected_provider(tmp_path, monkeypatch, selected_provider):
    submission, facts, _ = _fixture(tmp_path)
    (submission / "linked.json").symlink_to(facts)
    for member in ("../facts.json", str(facts), "linked.json"):
        with pytest.raises(ValueError):
            inspect_caller_layout(
                submission=submission, command_buffer_member=member, target="gemmini", facts_path=facts
            )

    def unavailable(target):
        raise ValueError("no selected provider")

    monkeypatch.setattr(plugins, "resolve_support", unavailable)
    with pytest.raises(Exception):
        inspect_caller_layout(
            submission=submission, command_buffer_member="command_buffer.json", target="gemmini", facts_path=facts
        )


def test_selected_renderer_without_optional_layout_query_refuses_only_the_probe(
    tmp_path, monkeypatch, selected_provider
):
    submission, facts, _ = _fixture(tmp_path)
    from merlin.runtime.backends import base as backends

    renderer = SimpleNamespace(__file__=str(selected_provider.base / "renderer.py"), render_harness=lambda *a, **k: "C")
    monkeypatch.setattr(backends, "get_backend", lambda target: renderer)
    assert renderer.render_harness({}) == "C"
    with pytest.raises(ValueError, match="does not expose caller-layout inspection"):
        inspect_caller_layout(
            submission=submission, command_buffer_member="command_buffer.json", target="gemmini", facts_path=facts
        )


def test_projection_rejects_extra_fields_and_wrong_physical_bounds(tmp_path, selected_provider):
    submission, facts, _ = _fixture(tmp_path)
    receipt = inspect_caller_layout(
        submission=submission, command_buffer_member="command_buffer.json", target="gemmini", facts_path=facts
    )
    command = json.loads((submission / "command_buffer.json").read_text())
    projection = {"schema": "caller_storage_layout_v1", "policy": receipt["policy"], "tensors": receipt["tensors"]}
    projection["tensors"] = [dict(row) for row in projection["tensors"]]
    projection["tensors"][0]["secret"] = "not allowed"
    with pytest.raises(ValueError, match="extra field"):
        _checked_projection(projection, command)
    del projection["tensors"][0]["secret"]
    projection["tensors"][0]["storage_elements"] = 1
    with pytest.raises(ValueError, match="physical bounds"):
        _checked_projection(projection, command)


def test_explicit_projection_must_equal_the_declared_encoding(tmp_path, selected_provider):
    submission, _, _ = _fixture(tmp_path)
    command = json.loads((submission / "command_buffer.json").read_text())
    command["params"]["storage_encodings"] = {
        name: GroupedAxesStorage((2, 3), "i8", ((0,), (1,)), (2, 3), (4, 1), 8).to_dict() for name in ("A", "Y")
    }
    projection = {
        "schema": "caller_storage_layout_v1",
        "policy": {"mode": "declared_grouped_axes_storage_v1"},
        "tensors": [
            {
                "tensor": name,
                "dtype": "i8",
                "logical_shape": [2, 3],
                "physical_extents": [2, 3],
                "logical_strides_elements": [3, 1],
                "storage_elements": 6,
                "offset_elements": 0,
            }
            for name in ("A", "Y")
        ],
    }
    with pytest.raises(ValueError, match="declared encoding"):
        _checked_projection(projection, command)


def test_a_core_backend_is_admitted_only_when_the_providers_contract_names_it(tmp_path, monkeypatch, selected_provider):
    """A data-only provider is served by an installed GENERIC backend outside its directory. That is
    admitted only for the core module its own contract names; any other core module is still refused."""
    from merlin.runtime.backends import chipyard_rocc
    from merlin.targetgen.contract import harness_render

    submission, facts, _ = _fixture(tmp_path)
    generic = Path(chipyard_rocc.__file__)
    impostor = SimpleNamespace(
        __file__=str(harness_render.__file__),
        caller_layout_source_paths=lambda: (Path(harness_render.__file__),),
        describe_caller_layout=lambda cb, **_: {},
    )
    selected_provider.plugin = lambda: {"backend": "merlin.runtime.backends.chipyard_rocc"}
    monkeypatch.setattr(backends, "get_backend", lambda target: impostor)
    with pytest.raises(ValueError, match="differs from explicit support provider"):
        inspect_caller_layout(
            submission=submission, command_buffer_member="command_buffer.json", target="gemmini", facts_path=facts
        )
    named = SimpleNamespace(
        __file__=str(generic),
        caller_layout_source_paths=lambda: (generic, Path("/etc/hostname")),
        describe_caller_layout=lambda cb, **_: {},
    )
    monkeypatch.setattr(backends, "get_backend", lambda target: named)
    with pytest.raises(ValueError, match="outside its provider and the installed core"):
        inspect_caller_layout(
            submission=submission, command_buffer_member="command_buffer.json", target="gemmini", facts_path=facts
        )
