"""The package classifier: one vocabulary, and an absence that names its own cause.

THE SHARPEST FORM OF THE PROBLEM IT SOLVES. Two loaders want ``<dir>/manifest.yaml`` with DISJOINT
schemas, and both shapes ship in sibling directories of the same tree:
``out/artifacts/targets/<t>/hand_v0/manifest.yaml`` is registry-format (``target`` / ``run_id`` /
``outputs.dialect_module``, no ``artifact_type``), while
``out/artifacts/targets/<t>/<t>_xdsl_rtl_v0/manifest.yaml`` is experiment-ABI format
(``artifact_type`` / ``entrypoints`` / ``commands``, no ``run_id``). Handed one of them, no code could
say which contract it claimed without trial-parsing against two schemas, and each loader found out by
failing somewhere inside itself.

So the tests below are mostly about DISCRIMINATION — the same directory tree, two answers — and about
the shape of the answer: a capability is provided with evidence, or absent with the exact missing file
or key. Never a bare boolean, because that is the shape in which "no compiler" and "could not tell"
became the same value.
"""

from __future__ import annotations

import pytest
import yaml

from merlin.targetgen import package as pkg


def _contract(root, name="synth", plugin=None, extra=None):
    body = {
        "name": name,
        "version": "0.1",
        "capabilities": {"ops": ["matmul"]},
        "memory_model": {},
        "compiler_obligations": [],
        "hardware_promises": [],
        "runtime_promises": [],
        "legality": [],
        **(extra or {}),
    }
    if plugin is not None:
        body["plugin"] = plugin
    (root / "contracts").mkdir(parents=True, exist_ok=True)
    (root / "contracts" / "target_contract.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")


def _abi_manifest(root, *, tool="tools/compile.py", commands=None, make_tool=True):
    body = {
        "artifact_type": "mlir_oot_target_backend",
        "target": "synth",
        "language": "python",
        "authoring": {"mode": "deterministic_generated_from_spec"},
        "integrity_exempt": False,
        "entrypoints": {"tool": tool},
        "commands": {
            c: {"argv": ["python3", tool]} for c in (commands if commands is not None else pkg._REQUIRED_COMMANDS)
        },
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    if make_tool:
        (root / tool).parent.mkdir(parents=True, exist_ok=True)
        (root / tool).write_text("", encoding="utf-8")


# ------------------------------------------------------------------ the answer shape
def test_every_capability_is_settled_even_for_an_empty_directory(tmp_path):
    """Not-in-the-dict is the shape that let a missing answer read as a negative one."""
    caps = pkg.capabilities_of(tmp_path)
    assert set(caps) == set(pkg.CAPABILITIES)
    for name, cap in caps.items():
        assert cap.provided is False
        assert cap.missing, f"{name}: an absence with no named cause is not an answer"


def test_an_absence_names_the_exact_file_or_key(tmp_path):
    _contract(tmp_path)  # a contract, but no plugin block
    backend = pkg.capability(tmp_path, "backend")
    assert backend.provided is False
    assert backend.missing == ("contracts/target_contract.yaml:plugin.backend",)

    _contract(tmp_path, plugin={"backend": "nowhere"})
    dangling = pkg.capability(tmp_path, "backend")
    assert dangling.provided is False
    assert "plugin.backend -> nowhere" in dangling.missing[0]


def test_an_unknown_capability_name_raises_rather_than_answering(tmp_path):
    with pytest.raises(KeyError, match="unknown capability"):
        pkg.capability(tmp_path, "simulator")


def test_require_raises_with_the_cause_in_the_message(tmp_path):
    with pytest.raises(pkg.PackageCapabilityMissing) as exc:
        pkg.require(tmp_path, "compiler")
    assert "manifest.yaml" in str(exc.value)


# ------------------------------------------------------------------ discrimination
def test_the_two_manifest_formats_are_told_apart(tmp_path):
    """The collision itself: one filename, two schemas, two sibling directories, two answers."""
    registry_pkg = tmp_path / "hand_v0"
    registry_pkg.mkdir()
    (registry_pkg / "manifest.yaml").write_text(
        yaml.safe_dump({"target": "synth", "run_id": "hand_v0", "outputs": {"dialect_module": "dialect.py"}}),
        encoding="utf-8",
    )
    (registry_pkg / "dialect.py").write_text("SPEC_OPS = {}\nDIALECT_NAME = 'synth'\n", encoding="utf-8")
    (registry_pkg / "lowering.yaml").write_text("interface_to_target: {}\ntarget_to_opcode: {}\n", encoding="utf-8")

    abi_pkg = tmp_path / "synth_xdsl_rtl_v0"
    abi_pkg.mkdir()
    _abi_manifest(abi_pkg)

    assert pkg.capability(registry_pkg, "dialect").provided is True
    assert pkg.capability(registry_pkg, "compiler").provided is False
    assert pkg.capability(abi_pkg, "compiler").provided is True
    assert pkg.capability(abi_pkg, "dialect").provided is False


def test_a_dialect_module_without_spec_ops_is_not_a_dialect(tmp_path):
    """Structural, by AST: the file exists and is not the thing. Importing to find out would run it."""
    (tmp_path / "dialect.py").write_text("DIALECT_NAME = 'synth'\n", encoding="utf-8")
    (tmp_path / "lowering.yaml").write_text("interface_to_target: {}\n", encoding="utf-8")
    cap = pkg.capability(tmp_path, "dialect")
    assert cap.provided is False
    assert "dialect.py:SPEC_OPS" in cap.missing


def test_a_missing_abi_command_is_named(tmp_path):
    """Which command is missing, by name — not "the manifest is invalid".

    The frozen ABI schema already requires all four, so this reason comes from the schema check and
    quotes the command it missed. The per-command loop in ``_infer_compiler`` is the fallback for when
    the schema machinery itself is unavailable; either way the answer names a command, never a shrug.
    """
    _abi_manifest(tmp_path, commands=["parse", "lower_interface_to_target"])
    cap = pkg.capability(tmp_path, "compiler")
    assert cap.provided is False
    assert any("emit_command_buffer" in m for m in cap.missing), cap.missing
    assert all(m.startswith("manifest.yaml") for m in cap.missing), "an absence names the FILE it is in"


def test_a_tool_a_build_step_will_produce_is_not_a_missing_tool(tmp_path):
    """A C++ package's tool does not exist until it is built; that is not an absent capability."""
    _abi_manifest(tmp_path, make_tool=False)
    body = yaml.safe_load((tmp_path / "manifest.yaml").read_text(encoding="utf-8"))
    body["build"] = {"command": ["make"], "tool_output": "build/synth-opt"}
    body["language"] = "cpp"
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    assert pkg.capability(tmp_path, "compiler").provided is True


# ------------------------------------------------------------------ explicit declaration
def test_an_explicit_declaration_overrides_inference_and_says_so(tmp_path):
    _abi_manifest(tmp_path)
    assert pkg.capability(tmp_path, "compiler").source == "inferred"
    body = yaml.safe_load((tmp_path / "manifest.yaml").read_text(encoding="utf-8"))
    body["package_capabilities"] = {"compiler": {"provided": False, "missing": ["not written yet"]}}
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    cap = pkg.capability(tmp_path, "compiler")
    assert cap.provided is False and cap.source == "declared"
    assert cap.missing == ("not written yet",)


def test_the_declaration_key_does_not_collide_with_the_hardware_capabilities_block(tmp_path):
    """``capabilities:`` already means the hardware's op/layout surface. Reusing it would make one
    key mean two unrelated things depending on which reader got there first."""
    _contract(tmp_path, plugin={"backend": "backend"})
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "__init__.py").write_text("", encoding="utf-8")
    contract = yaml.safe_load((tmp_path / "contracts" / "target_contract.yaml").read_text(encoding="utf-8"))
    assert contract["capabilities"] == {"ops": ["matmul"]}
    assert pkg.DECLARATION_KEY != "capabilities"
    # The hardware block is not read as a capability declaration.
    assert pkg.capability(tmp_path, "backend").source == "inferred"


def test_a_declaration_naming_an_unknown_capability_is_refused(tmp_path):
    _abi_manifest(tmp_path)
    body = yaml.safe_load((tmp_path / "manifest.yaml").read_text(encoding="utf-8"))
    body["package_capabilities"] = {"teleporter": True}
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown capabilities"):
        pkg.capabilities_of(tmp_path)


# ------------------------------------------------------------------ mutation guard
def test_the_classifier_would_notice_a_capability_going_away(tmp_path):
    """MUTATION: remove the evidence and the answer must flip, with the cause named.

    A classifier that answered ``provided`` for everything would pass every test above that only
    asserts a positive. This one asserts the transition in both directions on the same directory.
    """
    _contract(tmp_path, plugin={"backend": "backend"})
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "__init__.py").write_text("", encoding="utf-8")
    before = pkg.capability(tmp_path, "backend")
    assert before.provided is True and "backend" in before.evidence

    (tmp_path / "backend" / "__init__.py").unlink()
    (tmp_path / "backend").rmdir()
    after = pkg.capability(tmp_path, "backend")
    assert after.provided is False
    assert any("plugin.backend -> backend" in m for m in after.missing)
