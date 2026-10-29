"""Fresh file records alone cannot establish the original-to-snapshot association."""

import copy
import json

import pytest

from merlin.verify.lean.check import _file_ref
from merlin.verify.lean.snapshot_binding import FileReference, SnapshotBinding, contract_bindings


def pair(tmp_path):
    original = tmp_path / "original.json"
    snapshot = tmp_path / "snapshot.json"
    original.write_text(json.dumps({"caps": [127]}))
    snapshot.write_bytes(original.read_bytes())
    return {"original": _file_ref(original), "snapshot": _file_ref(snapshot)}


def test_equal_bytes_at_different_paths_are_bound(tmp_path):
    raw = pair(tmp_path)
    bound = SnapshotBinding.decode(raw, "contract")
    bound.assert_current("contract")
    assert bound.encode() == raw
    assert bound.original.path != bound.snapshot.path


@pytest.mark.parametrize("kind", ["input_contracts", "exponent_contracts"])
@pytest.mark.parametrize("change", ["valid_caps", "valid_json_whitespace"])
def test_individually_fresh_original_cannot_replace_its_snapshot(tmp_path, kind, change):
    raw = pair(tmp_path)
    other = tmp_path / "different-original.json"
    other.write_text(json.dumps({"caps": [254]}) if change == "valid_caps" else '{ "caps": [127] }\n')
    changed = {**raw, "original": _file_ref(other)}
    report = {"input_contracts": raw, "exponent_contracts": raw, kind: changed}
    # Both records are real and current; the missing fact is that they have equal bytes.
    FileReference.decode(changed["original"], "original").assert_current("original")
    FileReference.decode(changed["snapshot"], "snapshot").assert_current("snapshot")
    with pytest.raises(ValueError, match="different byte identities"):
        contract_bindings(report)


@pytest.mark.parametrize(
    "field,value",
    [("bytes", True), ("bytes", -1), ("path", False), ("path", "relative.json"), ("sha256", 0), ("sha256", "A" * 64)],
)
def test_file_identity_types_are_validated_before_comparison(tmp_path, field, value):
    raw = pair(tmp_path)["original"]
    raw[field] = value
    with pytest.raises(ValueError):
        FileReference.decode(raw, "contract")
    with pytest.raises(ValueError):
        FileReference(**raw)


@pytest.mark.parametrize("side", ["original", "snapshot"])
def test_binding_rechecks_live_bytes_after_it_was_validated(tmp_path, side):
    raw = pair(tmp_path)
    binding = SnapshotBinding.decode(raw, "contract")
    changed = tmp_path / ("original.json" if side == "original" else "snapshot.json")
    changed.write_text(json.dumps({"caps": [254]}))
    with pytest.raises(ValueError, match="file identity changed"):
        binding.assert_current("contract")


def test_equal_hash_with_forged_size_is_not_content_identity(tmp_path):
    raw = pair(tmp_path)
    changed = copy.deepcopy(raw)
    changed["original"]["bytes"] += 1
    with pytest.raises(ValueError, match="different byte identities"):
        SnapshotBinding.decode(changed, "contract")
