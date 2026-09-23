"""Public ISA inputs are example-owned and selected through the descriptor."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.generate_bundles import generate_bundles
from merlin.targetgen.sandbox.bwrap import resolve_grant
from merlin.targetgen.target_experiment import load_target_experiment

pytestmark = pytest.mark.target("atlas", "radiance", "mx_gemmini")


@pytest.mark.parametrize(
    ("target", "bringup", "files"),
    [
        ("mx_gemmini", "hwbringup_mx_v0", ["isa_definition.py", "isa_patterns.py", "mmio_abi.py"]),
        ("radiance", "hwbringup_radiance_v0", ["isa_definition.py", "isa_patterns.py"]),
        ("atlas", "hwbringup_atlas_v0", ["isa_definition.py"]),
    ],
)
def test_bundle_grants_select_example_contract(target, bringup, files):
    root = repo_root()
    descriptor = load_target_experiment(root / f"examples/{target}/target/descriptor.yaml")
    contract = f"examples/{target}/phase1/contracts/{bringup}"
    assert descriptor.hwbringup_set == contract
    assert contract + "/isa_include/isa_definition.py" in descriptor.isa_headers
    assert all(path.startswith(contract + "/isa_include/") for path in descriptor.isa_headers)
    assert all((root / path).is_file() for path in descriptor.isa_headers)
    for manifest in generate_bundles(descriptor).values():
        grants = {entry["path"] for entry in manifest["allowed"]}
        assert contract in grants
        assert set(descriptor.isa_headers) <= grants
        assert not any(f"targets/{target}/contracts/{bringup}" in path for path in grants)
        assert resolve_grant(contract, root) == root / contract
    assert sorted(path.name for path in (root / contract / "isa_include").glob("*.py")) == files


def test_atlas_preflight_paths_and_hooks_resolve_without_running_tools(monkeypatch):
    from merlin.targetgen.inline_assembly_probe import _memory_annotations, _resolve_path, load_hook

    def forbidden(*args, **kwargs):
        raise AssertionError("path validation must not execute tools or a program oracle")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setitem(sys.modules, "merlin.targetgen.program_oracle", SimpleNamespace(run_raw_program=forbidden))
    root = repo_root()
    descriptor = load_target_experiment(root / "examples/atlas/target/descriptor.yaml")
    assert len(descriptor.preflight_capability_probes) == 2
    for probe in descriptor.preflight_capability_probes:
        fixture = probe.fixture
        assert fixture["source"].startswith(descriptor.hwbringup_set + "/preflight/")
        source = _resolve_path(fixture["source"])
        assert source == root / fixture["source"]
        assert _memory_annotations(source.read_text())[1]
        for name in ("assembler", "runner"):
            reference = fixture[name]
            assert reference.startswith(descriptor.hwbringup_set + "/preflight/adapter.py:")
            assert callable(load_hook(reference))


def test_atlas_crosscheck_reads_selected_example_evidence(monkeypatch):
    from merlin.targetgen import corpora, isa_rtl_crosscheck

    root = repo_root()
    descriptor = root / "examples/atlas/target/descriptor.yaml"
    monkeypatch.setattr(corpora, "descriptor_path", lambda target: descriptor)
    selected = root / "examples/atlas/phase1/contracts/hwbringup_atlas_v0"
    assert isa_rtl_crosscheck.contracts_dir("atlas") == selected
    assert isa_rtl_crosscheck.green_card_paths("atlas") == [selected / "isa_include/atlas_isa_green_card.md"]


def test_radiance_isa_sibling_import():
    include = repo_root() / "examples/radiance/phase1/contracts/hwbringup_radiance_v0/isa_include"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); from isa_definition import ISA; "
            "assert len(set(ISA.operations.values())) == 7; "
            "assert ISA.operations['vx_bar'] is ISA.operations['vx_barrier']; "
            "assert all(isinstance(op().to_bytecode(), int) for op in ISA.operations.values())",
            str(include),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_moved_isa_sibling_import_and_abi_agree():
    include = repo_root() / "examples/mx_gemmini/phase1/contracts/hwbringup_mx_v0/isa_include"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from isa_definition import MX_INVOKE; from mmio_abi import compose_rocc_word; "
            "word = MX_INVOKE(); "
            "exec('for command in range(128):\\n word.imm = command\\n"
            " assert word.to_bytecode() == compose_rocc_word(command)')",
            str(include),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
