"""Convolution readout discovery separates defaults, opt-in and qualification."""
import importlib
import json

from merlin.common.paths import repo_root
from merlin.compile.scheduling.api_catalog import scheduling_capabilities
from merlin_experiments.phase1 import operation_optimizations as operations
from merlin_experiments.phase1 import optimization_suggestions as suggestions
from merlin_experiments.phase1.feedback import case_declarations, mechanism_hints


def selection():
    catalog = json.loads((repo_root()/"docs/design/joint_optimization_catalog.json").read_text())
    capabilities = {suggestions.ALIASES[t]: "supported" for row in catalog["mechanisms"]
                    for t in row["required_traits"] if not suggestions.ALIASES[t].startswith("workload.")}
    return suggestions.select(catalog, capabilities=capabilities,
        public_modules=tuple(row["compiler"]["module"] for row in catalog["mechanisms"]))


def readout_row(selected):
    return next(row for row in selected["all_mechanisms"] if row["id"] == "convolution_readout_epochs")


def test_capability_symbols_resolve_and_old_inventory_does_not_gain_capability():
    from merlin.compile.scheduling.export import public_modules
    from merlin.compile.scheduling.library_sources import library_modules

    old = scheduling_capabilities(22)
    assert not any(c["name"] == "source_qualified_convolution_readout_epochs" for c in old["capabilities"])
    for name in ("source_qualified_convolution_readout_epochs", "capacity_phased_convolution_rhs_residency"):
        cap = next(c for c in scheduling_capabilities(23)["capabilities"] if c["name"] == name)
        for symbol in cap["symbols"]:
            module, _, symbol_name = symbol.rpartition(".")
            assert callable(getattr(importlib.import_module(module), symbol_name))
        assert cap["requires"]
    assert "merlin.compile.scheduling.affine_gather_runs" in library_modules(23)
    assert "merlin.compile.scheduling.affine_gather_runs" not in library_modules(22)
    assert "merlin.compile.scheduling.batched_dense" in public_modules(23)
    assert "merlin.compile.scheduling.batched_dense" not in library_modules(22)


def test_operation_menu_links_readout_only_from_convolution_family():
    catalog = json.loads((repo_root()/operations.CATALOG).read_text())
    assert operations.validate(catalog, repo=repo_root())
    owners = [f["id"] for f in catalog["families"]
              if any(m.get("id") == "convolution_readout_epochs" for m in f["mechanisms"])]
    assert owners == ["convolution"]


def test_known_dense_excludes_both_conv_epoch_actions(monkeypatch, tmp_path):
    dense = dict(operation=dict(op="matmul", attributes=dict(lhs="a", weight="w")),
        inputs=[dict(name="a", role="input", shape=[4, 8]), dict(name="w", role="weight", shape=[8, 12])])
    monkeypatch.setattr(case_declarations, "load", lambda *_: dense)
    selected = mechanism_hints.for_case(selection(), "arbitrary-case", tmp_path)
    assert readout_row(selected)["status"] == "unsupported"
    found, unknown = mechanism_hints.fence_hints(selected, why="fence-bound", limit=100)
    assert not {r["id"] for r in (*found, *unknown)} & {
        "convolution_readout_epochs", "convolution_compute_epochs"}


def test_a_pure_windowed_reduction_is_not_a_convolution_producer(monkeypatch, tmp_path):
    pool = dict(semantic=dict(semantic_family="reduction"),
        operation=dict(op="maxpool", attributes=dict(kernel=[2, 2], stride=[2, 2])),
        inputs=[dict(name="a", role="input", shape=[1, 4, 4, 4])])
    monkeypatch.setattr(case_declarations, "load", lambda *_: pool)
    selected = mechanism_hints.for_case(selection(), "arbitrary-case", tmp_path)
    assert readout_row(selected)["status"] == "unsupported"


def test_confirmed_convolution_surfaces_missing_readout_seam_in_compact_hints(monkeypatch, tmp_path):
    conv = dict(operation=dict(op="conv2d", attributes=dict(ifm="x", weight="w", layout="NHWC",
        kernel=[3, 3, 8, 12], stride=[1, 1], padding=[0, 0, 0, 0])),
        inputs=[dict(name="x", role="input", shape=[1, 5, 7, 8]),
                dict(name="w", role="weight", shape=[3, 3, 8, 12])])
    monkeypatch.setattr(case_declarations, "load", lambda *_: conv)
    selected = mechanism_hints.for_case(selection(), "arbitrary-case", tmp_path)
    found, _ = mechanism_hints.fence_hints(selected, why="fence-bound")
    hint = next(row for row in found if row["id"] == "convolution_readout_epochs")
    assert hint["opportunity_status"] == "applicability_unknown"
    assert "source" in hint["next_action"] and hint["source_call_sites"] == []


def test_default_call_is_not_observed_readout_application(tmp_path):
    root = tmp_path/"submission"
    root.mkdir()
    path = root/"lowering.py"
    prefix = "from merlin.compile.scheduling.convolution_exploration import emit_convolution_schedule as emit\n"
    path.write_text(prefix+"emit(schedule, builder, lower, complete=drain)\n"
                    +"emit(schedule, builder, lower, complete=drain, readout_law=None)\n")
    selected = mechanism_hints.with_source_evidence(selection(), tmp_path)
    observed = mechanism_hints.opportunity(readout_row(selected), selected)
    assert observed["consumer_call_sites"] and not observed["source_call_sites"]
    assert observed["opportunity_status"] == "applicability_unknown"
    assert any("consumer receipt" in p for p in observed["prerequisites"])
    path.write_text(prefix+"emit(schedule, builder, lower, complete=drain, readout_law=selected_law)\n")
    selected = mechanism_hints.with_source_evidence(selection(), tmp_path)
    # Source observation remains conditional: workload and source premises are
    # not inferred from an import or this keyword.
    observed = mechanism_hints.opportunity(readout_row(selected), selected)
    assert observed["source_call_sites"] == ["lowering.py:2"]
    assert observed["observation_basis"] == "source call only; runtime application unconfirmed"
