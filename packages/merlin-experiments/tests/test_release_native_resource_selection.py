"""Host resource changes preserve original contract bytes and semantic declarations."""

import copy

import pytest
import yaml
from merlin_experiments.corpus.preparation import _stage_capability_contract
from merlin_experiments.spec import SpecError

from merlin.common.digest import sha256_file


def _source(tmp_path):
    from merlin.runtime.backends import chipyard_rocc_bound as backend

    headers = [{"id": identifier, "sha256": sha256_file(backend._resource(identifier, backend._HEADERS))}
               for identifier in ("htif", "out_b64")]
    document = {
        "name": "fixture", "numeric_policy": {"kind": "exact"},
        "harness_abi": {"readback_transport": "out_b64_v1"},
        "runner": {"backend": "chipyard_rocc", "chipyard_rocc": {
            "version": 1, "config": "fixture", "toolchain": {"selected": "original", "headers": headers},
            "engines": {
                "gsim": {"max_timeout_s": 600, "argv": ["{elf}", "+max-cycles=20000000"]},
                "spike": {"max_timeout_s": 600, "argv": ["{elf}"]},
            },
        }},
    }
    path = tmp_path / "original.yaml"
    path.write_text(yaml.safe_dump(document))
    return path, document


@pytest.mark.parametrize("selected_timeout", [1800, 86400])
def test_explicit_resource_selection_preserves_source_and_all_nonresource_fields(tmp_path, selected_timeout):
    source, original = _source(tmp_path)
    before = source.read_bytes()
    destination = tmp_path / "selected.yaml"
    record = _stage_capability_contract(source, sha256_file(source), destination, selected_timeout)
    selected = yaml.safe_load(destination.read_text())
    restored = copy.deepcopy(selected)
    block = restored["runner"]["chipyard_rocc"]
    assert block.pop("execution_resources") == {"max_native_timeout_s": selected_timeout}
    assert block["engines"]["gsim"]["max_timeout_s"] == selected_timeout
    block["engines"]["gsim"]["max_timeout_s"] = 600
    assert restored == original
    assert source.read_bytes() == before
    assert record["sha256"] == sha256_file(source)
    assert record["execution_resource_selection"]["selected_contract_sha256"] == sha256_file(destination)
    assert record["execution_resource_selection"]["engines"] == ["gsim"]


def test_default_selection_keeps_exact_original_bytes(tmp_path):
    source, _ = _source(tmp_path)
    destination = tmp_path / "selected.yaml"
    record = _stage_capability_contract(source, sha256_file(source), destination, None)
    assert destination.read_bytes() == source.read_bytes()
    assert "execution_resource_selection" not in record


def test_model_spike_budget_is_a_pinned_resource_variant_of_unchanged_phase0_contract(tmp_path):
    source, original = _source(tmp_path)
    before = source.read_bytes()
    destination = tmp_path / "selected.yaml"
    record = _stage_capability_contract(
        source, sha256_file(source), destination, 1800, model_spike_timeout_s=172800
    )
    selected = yaml.safe_load(destination.read_text())
    block = selected["runner"]["chipyard_rocc"]
    assert block["execution_resources"] == {
        "max_native_timeout_s": 1800, "max_model_spike_timeout_s": 172800,
    }
    assert block["engines"]["gsim"]["max_timeout_s"] == 1800
    assert block["engines"]["spike"]["max_timeout_s"] == 172800
    assert source.read_bytes() == before
    assert record["model_spike_resource_selection"]["selected_contract_sha256"] == sha256_file(destination)
    assert original["runner"]["chipyard_rocc"]["engines"]["spike"]["max_timeout_s"] == 600


@pytest.mark.parametrize("mutate", [False, True])
def test_reviewed_phase1_handoff_reconstructs_selected_model_spike_budget(tmp_path, monkeypatch, mutate):
    from types import SimpleNamespace

    from merlin_experiments.phase0 import coverage_commitment
    from merlin_experiments.phase1 import corpus_inputs

    source, original = _source(tmp_path)
    experiment = tmp_path / "release/payload/experiment"
    destination = experiment / "contracts/selected.yaml"
    _stage_capability_contract(
        source, sha256_file(source), destination, 1800, model_spike_timeout_s=172800
    )
    if mutate:
        selected = yaml.safe_load(destination.read_text())
        selected["runner"]["chipyard_rocc"]["engines"]["spike"]["max_timeout_s"] = 172799
        destination.write_text(yaml.safe_dump(selected))
    bundle = experiment / "input_bundles/el4/input_bundle_manifest.yaml"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("bundle_id: el4\n")
    te = SimpleNamespace(path=experiment / "target_experiment.yaml", capsule_corpus=experiment / "corpus/isa",
                         declared_contract_path=lambda: destination, graded_roots=lambda: [])
    monkeypatch.setattr(coverage_commitment, "read_inputs", lambda _: {"capability_contract": original})
    declaration = {"bundle_id": "el4", "allowed": [{"mode": "ro", "path": str(destination)}]}
    if mutate:
        with pytest.raises(ValueError, match="differs from selected Phase 0"):
            corpus_inputs.require_reviewed_bundle(te, bundle, declaration)
    else:
        corpus_inputs.require_reviewed_bundle(te, bundle, declaration)


@pytest.mark.parametrize("budget", [True, 0, -1, 86401, 1800.0])
def test_invalid_resource_selection_publishes_no_contract(tmp_path, budget):
    source, _ = _source(tmp_path)
    destination = tmp_path / "selected.yaml"
    with pytest.raises(SpecError):
        _stage_capability_contract(source, sha256_file(source), destination, budget)
    assert not destination.exists()


def test_changed_original_contract_refuses_before_publication(tmp_path):
    source, _ = _source(tmp_path)
    digest = sha256_file(source)
    source.write_text("changed original\n")
    destination = tmp_path / "selected.yaml"
    with pytest.raises(SpecError, match="differs from verified evidence"):
        _stage_capability_contract(source, digest, destination, 1800)
    assert not destination.exists()


@pytest.mark.parametrize("mutation", [None, "numeric", "cycle", "spike", "unbounded", "extra_resource"])
def test_reviewed_phase1_handoff_accepts_only_the_exact_resource_variant(tmp_path, monkeypatch, mutation):
    from types import SimpleNamespace

    from merlin_experiments.phase0 import coverage_commitment
    from merlin_experiments.phase1 import corpus_inputs

    source, original = _source(tmp_path)
    experiment = tmp_path / "release/payload/experiment"
    destination = experiment / "contracts/selected.yaml"
    _stage_capability_contract(source, sha256_file(source), destination, 1800)
    selected = yaml.safe_load(destination.read_text())
    block = selected["runner"]["chipyard_rocc"]
    if mutation == "numeric":
        selected["numeric_policy"]["kind"] = "approximate"
    elif mutation == "cycle":
        block["engines"]["gsim"]["argv"][-1] = "+max-cycles=40000000"
    elif mutation == "spike":
        block["engines"]["spike"]["max_timeout_s"] = 1800
    elif mutation == "unbounded":
        block["execution_resources"]["max_native_timeout_s"] = 86401
    elif mutation == "extra_resource":
        block["execution_resources"]["extra"] = 1
    destination.write_text(yaml.safe_dump(selected))
    bundle = experiment / "input_bundles/el4/input_bundle_manifest.yaml"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("bundle_id: el4\n")
    te = SimpleNamespace(path=experiment / "target_experiment.yaml", capsule_corpus=experiment / "corpus/isa",
                         declared_contract_path=lambda: destination, graded_roots=lambda: [])
    monkeypatch.setattr(coverage_commitment, "read_inputs", lambda _: {"capability_contract": original})
    declaration = {"bundle_id": "el4", "allowed": [{"mode": "ro", "path": str(destination)}]}
    if mutation is None:
        corpus_inputs.require_reviewed_bundle(te, bundle, declaration)
    else:
        with pytest.raises(ValueError, match="differs from selected Phase 0"):
            corpus_inputs.require_reviewed_bundle(te, bundle, declaration)


@pytest.mark.parametrize("timeout", [None, 1800])
@pytest.mark.parametrize("transport", ["coherent_dump_v1", "coherent_packet_v1"])
def test_full_value_transport_selection_retains_original_and_records_exact_selected_bytes(tmp_path, timeout, transport):
    source, original = _source(tmp_path)
    before = source.read_bytes()
    destination = tmp_path / "selected.yaml"
    record = _stage_capability_contract(source, sha256_file(source), destination, timeout, transport)
    selected = yaml.safe_load(destination.read_bytes())
    restored = copy.deepcopy(selected)
    block = restored["runner"]["chipyard_rocc"]
    assert block["engines"]["gsim"].pop("readback_transport") == transport
    if transport == "coherent_packet_v1":
        from merlin.runtime.backends import chipyard_rocc_bound as backend

        appended = block["toolchain"]["headers"][2:]
        assert appended == [
            {"id": identifier, "sha256": sha256_file(backend._resource(identifier, backend._HEADERS))}
            for identifier in ("out_bin", "out_bin_memory")
        ]
        del block["toolchain"]["headers"][2:]
    if timeout is not None:
        assert block.pop("execution_resources") == {"max_native_timeout_s": timeout}
        block["engines"]["gsim"]["max_timeout_s"] = 600
    assert restored == original
    assert source.read_bytes() == before and record["sha256"] == sha256_file(source)
    assert record["execution_transport_selection"] == {
        "engine": "gsim", "readback_transport": transport,
        "selected_contract_sha256": sha256_file(destination),
        "scope": "complete output values only; original ABI, instruction and numerical declarations retained",
    }
    if timeout is not None:
        assert record["execution_resource_selection"]["selected_contract_sha256"] == sha256_file(destination)


@pytest.mark.parametrize("transport", ["out_digest_v1", "out_b64_v1", "other_transport", True, ""])
def test_other_transport_selection_refuses_without_publication(tmp_path, transport):
    source, _ = _source(tmp_path)
    destination = tmp_path / "selected.yaml"
    with pytest.raises(SpecError, match="complete coherent values"):
        _stage_capability_contract(source, sha256_file(source), destination, 1800, transport)
    assert not destination.exists()


@pytest.mark.parametrize("mutation", [
    None, "numeric", "abi", "spike_transport", "gsim_argv", "digest", "extra_engine", "header_hash", "header_extra",
])
@pytest.mark.parametrize("timeout", [None, 1800])
@pytest.mark.parametrize("transport", ["coherent_dump_v1", "coherent_packet_v1"])
def test_reviewed_handoff_allows_only_exact_full_values_transport_variant(
    tmp_path, monkeypatch, mutation, timeout, transport
):
    from types import SimpleNamespace

    from merlin_experiments.phase0 import coverage_commitment
    from merlin_experiments.phase1 import corpus_inputs

    source, original = _source(tmp_path)
    experiment = tmp_path / "release/payload/experiment"
    destination = experiment / "contracts/selected.yaml"
    _stage_capability_contract(source, sha256_file(source), destination, timeout, transport)
    selected = yaml.safe_load(destination.read_bytes())
    block = selected["runner"]["chipyard_rocc"]
    if mutation == "numeric":
        selected["numeric_policy"]["kind"] = "approximate"
    elif mutation == "abi":
        selected["harness_abi"]["readback_transport"] = "coherent_dump_v1"
    elif mutation == "spike_transport":
        block["engines"]["spike"]["readback_transport"] = "coherent_dump_v1"
    elif mutation == "gsim_argv":
        block["engines"]["gsim"]["argv"].append("+unreviewed-mode")
    elif mutation == "digest":
        block["engines"]["gsim"]["readback_transport"] = "out_digest_v1"
    elif mutation == "extra_engine":
        block["engines"]["gsim"]["unreviewed"] = True
    elif mutation == "header_hash":
        block["toolchain"]["headers"][-1]["sha256"] = "0" * 64
    elif mutation == "header_extra":
        block["toolchain"]["headers"].append({"id": "unreviewed", "sha256": "0" * 64})
    destination.write_text(yaml.safe_dump(selected))
    bundle = experiment / "input_bundles/el4/input_bundle_manifest.yaml"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("bundle_id: el4\n")
    te = SimpleNamespace(path=experiment / "target_experiment.yaml", capsule_corpus=experiment / "corpus/isa",
                         declared_contract_path=lambda: destination, graded_roots=lambda: [])
    monkeypatch.setattr(coverage_commitment, "read_inputs", lambda _: {"capability_contract": original})
    declaration = {"bundle_id": "el4", "allowed": [{"mode": "ro", "path": str(destination)}]}
    if mutation is None:
        corpus_inputs.require_reviewed_bundle(te, bundle, declaration)
    else:
        with pytest.raises(ValueError, match="differs from selected Phase 0"):
            corpus_inputs.require_reviewed_bundle(te, bundle, declaration)


def test_packet_selection_verifies_existing_codec_pins_and_refuses_corruption(tmp_path):
    from merlin.runtime.backends import chipyard_rocc_bound as backend
    from merlin.runtime.backends import rocc_selection

    _, original = _source(tmp_path)
    once = rocc_selection.select_gsim_readback(original, "coherent_packet_v1")
    assert rocc_selection.select_gsim_readback(once, "coherent_packet_v1") == once
    corrupted = copy.deepcopy(once)
    corrupted["runner"]["chipyard_rocc"]["toolchain"]["headers"][-1]["sha256"] = "0" * 64
    with pytest.raises(backend.RoCCExecutionError, match="declared execution member changed"):
        rocc_selection.select_gsim_readback(corrupted, "coherent_packet_v1")
