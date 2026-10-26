"""Exact support-source selection preserves the ordinary host package authority."""

import copy

import pytest
import yaml
from merlin_experiments.corpus.preparation import _select_host_capability, verify_host_capability
from merlin_experiments.spec import SpecError

from merlin.common.digest import sha256_file
from merlin.common.tree_hash import hash_tree


def inputs(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            dict(
                target="host",
                run_id="fixture",
                family="scalar_linalg",
                status="unverified",
                authoring={
                    "mode": "deterministic_generated_from_spec",
                    "generated_by_agent": False,
                    "author": "fixture",
                },
                outputs={"knobs": "knobs.yaml"},
            )
        )
    )
    (package / "knobs.yaml").write_text(
        yaml.safe_dump(
            dict(
                backend="scalar",
                dtype_strategy="int8_w8a8",
                cflags=["-march=rv64gc", "-mabi=lp64d"],
                expected_instructions=[],
            )
        )
    )
    spec = tmp_path / "support.yaml"
    spec.write_text(
        yaml.safe_dump(
            dict(
                schema="merlin.host_capabilities.v1",
                status="reviewed",
                compiler=dict(package_sha256=hash_tree(package)["sha256"], dtype_strategy="int8_w8a8"),
                operations=[],
                evidence={"source": "synthetic unit metadata only"},
            )
        )
    )
    host = dict(
        description="synthetic metadata selection",
        repo_canonical="https://example.invalid/source",
        package="package",
        requires_paths=["manifest.yaml", "knobs.yaml"],
        read_only=["package/"],
        deny_modification=[],
        provenance="in_tree_minted",
        dtype_strategy="int8_w8a8",
        capability_spec="support.yaml",
    )
    document = dict(host_lane=host, hardware_spec={"facts": "unchanged"}, capsule_corpus="unchanged")
    return document, spec, package, tmp_path / "descriptor.yaml"


@pytest.mark.parametrize("matrix", [False, True])
def test_support_pin_changes_only_selected_profile_metadata(tmp_path, matrix):
    doc, spec, package, descriptor = inputs(tmp_path)
    if matrix:
        doc["host_lane"] = {
            "default": "selected",
            "profiles": {"selected": copy.deepcopy(doc["host_lane"]), "other": copy.deepcopy(doc["host_lane"])},
        }
    before = copy.deepcopy(doc)
    package_pin = hash_tree(package)["sha256"]
    record = _select_host_capability(doc, (spec, sha256_file(spec)), descriptor=descriptor, root=tmp_path)
    verify_host_capability(doc, record, descriptor=descriptor, root=tmp_path)
    assert record["package_sha256"] == package_pin
    expected = copy.deepcopy(before)
    host = expected["host_lane"]["profiles"]["selected"] if matrix else expected["host_lane"]
    host["capability_spec_sha256"] = sha256_file(spec)
    assert doc == expected and hash_tree(package)["sha256"] == package_pin


@pytest.mark.parametrize("mutation", ["source", "package", "pointer", "record", "unreviewed"])
def test_changed_support_or_package_is_refused(tmp_path, mutation):
    doc, spec, package, descriptor = inputs(tmp_path)
    record = _select_host_capability(doc, (spec, sha256_file(spec)), descriptor=descriptor, root=tmp_path)
    if mutation == "source":
        spec.write_text(spec.read_text() + "\n# changed\n")
    elif mutation == "package":
        (package / "knobs.yaml").write_text((package / "knobs.yaml").read_text() + "\n# changed\n")
    elif mutation == "pointer":
        doc["host_lane"]["capability_spec_sha256"] = "f" * 64
    elif mutation == "record":
        record["package_sha256"] = "f" * 64
    else:
        data = yaml.safe_load(spec.read_text())
        data["status"] = "unreviewed"
        spec.write_text(yaml.safe_dump(data))
        record["sha256"] = sha256_file(spec)
        doc["host_lane"]["capability_spec_sha256"] = record["sha256"]
    with pytest.raises((ValueError, SpecError)):
        verify_host_capability(doc, record, descriptor=descriptor, root=tmp_path)


def test_unrelated_support_file_is_not_a_new_source_grant(tmp_path):
    doc, spec, _, descriptor = inputs(tmp_path)
    other = tmp_path / "unselected.yaml"
    other.write_bytes(spec.read_bytes())
    with pytest.raises(SpecError, match="ownership"):
        _select_host_capability(doc, (other, sha256_file(other)), descriptor=descriptor, root=tmp_path)
