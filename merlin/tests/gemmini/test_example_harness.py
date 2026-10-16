"""Fresh experiment sources contain no copied reference runtime or kernel helpers.

The gemmini support provider is DATA served by generic core code (merlin.runtime.backends.chipyard_rocc):
it ships no backend, headers or kernel library of its own, and no kernel library is on the include path.
"""

import json

import pytest
import yaml
from merlin_experiments.phase1 import component_witness

from merlin.common.paths import repo_root
from merlin.targetgen.target_experiment import load_target_experiment

pytestmark = pytest.mark.target("gemmini")


#: What a DATA-ONLY support provider may hold: declarations, never code or copied vendor bytes.
_DATA_SUFFIXES = {".yaml", ".yml", ".json", ".md"}


def test_reference_provider_and_headers_are_absent():
    root = repo_root()
    for relative in (
        "examples/gemmini/phase1/contracts/harness_curated",
        "merlin/experiments/capsule_bench/targets/gemmini/contracts/harness_curated",
        "merlin/experiments/capsule_bench/targets/gemmini_universal/contracts/harness_curated",
        "merlin/experiments/capsule_bench/targets/gemmini_universal/contracts/isa_include",
    ):
        path = root / relative
        assert not path.exists() and not path.is_symlink(), relative


def test_the_support_provider_is_data_served_by_generic_core_code():
    """examples/gemmini/support exists again, but only as DATA: no backend, kernel, header or CRT
    bytes, and its plugin block selects generic core modules rather than shipping its own."""
    from merlin.targetgen import plugins

    support = repo_root() / "examples/gemmini/support"
    members = [path for path in support.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    assert members, "the data provider is missing"
    # The ISA probe package is experimenter-side tooling kept inside the masked provider tree; it is
    # never selected by the plugin block below.
    probe = support / "probe_compiler"
    assert [path for path in members if path.suffix not in _DATA_SUFFIXES and probe not in path.parents] == []
    contract = yaml.safe_load((support / "contracts/target_contract.yaml").read_text(encoding="utf-8"))
    plugin = contract["plugin"]
    assert plugins.core_module_path(plugin["backend"]) is not None, plugin["backend"]
    spec = yaml.safe_load((support / plugin["isa_headers"]).read_text(encoding="utf-8"))
    # The vendor kernel library never reaches the include path; only the runtime and a generated header.
    for root in spec["include_roots"]:
        assert root not in (".", "include", "riscv-tests"), root
    assert not any(name.startswith("include/") for name in spec["files"])
    assert "include/gemmini.h" in spec["excluded_from_include_path"]
    assert spec["generated_header"]["name"].endswith(".h")


def test_reference_runtime_is_not_registered_or_selected_as_evidence():
    root = repo_root()
    document = json.loads((root / "build_tools/upstreams/target_support.json").read_bytes())
    rows = [row for row in document["companions"] if row["target"] == "gemmini"]
    # Only the data provider may be listed: no companion snapshot, vendored copy or reference tree.
    assert len(rows) == 1 and rows[0]["ownership"] == "generic_data_support"
    assert "vendored" not in rows[0] and "companion_commit" not in rows[0]
    descriptor = load_target_experiment(root / "examples/gemmini/target/descriptor.yaml")
    assert descriptor.isa_headers == ()
    assert descriptor.hwbringup_set is None
    assert descriptor.curated_harness is None
    assert not hasattr(component_witness, "selected_component_witness_verifier")


def test_a_sandboxed_candidate_cannot_read_the_probe_package():
    """The probe solves two ISA capsules, so it lives in the masked provider tree: a sandbox that
    exposes the entire checkout still hides every probe byte, and no regenerated bundle grants it."""
    pytest.importorskip("merlin_experiments")
    import importlib

    from merlin.targetgen.generate_bundles import generate_bundles

    surfaces = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    bwrap = importlib.import_module("merlin.targetgen.sandbox.bwrap")
    root = repo_root()
    probe = root / "examples/gemmini/support/probe_compiler"
    members = [path for path in probe.rglob("*") if path.is_file()]
    assert members, "the probe package is missing, so this proves nothing"
    descriptor = load_target_experiment(root / "examples/gemmini/target/descriptor.yaml")
    derived = surfaces.answer_surfaces(descriptor)
    exposed = ["--ro-bind", str(root), str(root)]
    assert all(bwrap.is_exposed(exposed, member) for member in members)
    masked = bwrap.apply_answer_masks(exposed, derived)
    assert [member for member in members if bwrap.is_exposed(masked, member)] == []
    for manifest in generate_bundles(descriptor).values():
        assert not any("probe_compiler" in entry["path"] for entry in manifest["allowed"])
