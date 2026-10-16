"""The committed evaluation inputs are small, hashed, free of holdouts, and selected by stated rules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.model_check import inputs as INP

#: Each model's stored fields: (arrays in ``visible.npz``, metadata in ``visible.json``).
FIELDS = {
    "resnet50": ({"image"}, {"member", "synset", "imagenet_label", "source_sha256"}),
    "smolvla": ({"frame", "state"}, {"episode", "frame", "noise_seed", "task", "raw_state"}),
    "tiny_llama": (set(), {"split", "article", "tokens"}),
}


def _root():
    return repo_root() / "examples" / "gemmini" / "numerics"


def test_the_public_example_fits_its_budget():
    assert INP.stored_bytes(_root()) <= INP.PUBLIC_BUDGET


def test_no_sealed_input_is_committed():
    assert not [p for p in (_root() / "inputs").rglob("*") if "sealed" in p.name]


def test_every_committed_input_matches_its_manifest_hash():
    manifest = json.loads((_root() / "MANIFEST.json").read_text())
    assert sorted(manifest["models"]) == sorted(FIELDS)
    for model, entry in manifest["models"].items():
        assert INP.stored_hashes(_root(), model, "visible") == entry["visible"], model
        assert sorted(p.name for p in (_root() / "inputs" / model).iterdir()) == sorted(entry["visible"]), model
        # Sealed forms are committed to by hash only, one per visible file.
        assert sorted(entry["sealed"]) == sorted(name.replace("visible", "sealed") for name in entry["visible"])


@pytest.mark.parametrize("model", sorted(FIELDS))
def test_stored_returns_the_eight_visible_inputs_with_their_fields(model):
    items = INP.stored(_root(), model, "visible")
    assert sorted(items) == [f"v{index:02d}" for index in range(8)]
    arrays, meta = FIELDS[model]
    for ident, item in items.items():
        assert set(item["arrays"]) == arrays, ident
        assert set(item["meta"]) == meta, ident


def _items() -> dict:
    return {
        "v01": {
            "meta": {"frame": 3, "task": "t", "state": [0.1, 0.2]},
            "arrays": {
                "frame": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
                "state": np.linspace(0, 1, 4, dtype=np.float32),
            },
        },
        "v00": {
            "meta": {"frame": 1, "task": "u", "state": []},
            "arrays": {"frame": np.zeros((2, 2, 3), np.uint8), "state": np.ones(4, np.float32)},
        },
    }


def test_stored_forms_round_trip_exactly_and_byte_stably(tmp_path):
    items = _items()
    written = INP.write_stored(tmp_path / "a", "m", "visible", items)
    assert written == list(INP.stored_paths(tmp_path / "a", "m", "visible"))
    assert written[0] == tmp_path / "a" / "inputs" / "m" / "visible.npz"
    back = INP.stored(tmp_path / "a", "m", "visible")
    assert sorted(back) == sorted(items)
    for ident, item in items.items():
        assert back[ident]["meta"] == item["meta"]
        assert set(back[ident]["arrays"]) == set(item["arrays"])
        for field, value in item["arrays"].items():
            got = back[ident]["arrays"][field]
            assert got.dtype == value.dtype and got.shape == value.shape and np.array_equal(got, value)
    # Writing what was read back (in another order) reproduces the same bytes.
    again = INP.write_stored(tmp_path / "b", "m", "visible", dict(reversed(list(back.items()))))
    assert [p.read_bytes() for p in again] == [p.read_bytes() for p in written]


def test_a_split_without_arrays_is_metadata_only(tmp_path):
    arrays_path, meta_path = INP.stored_paths(tmp_path, "m", "sealed")
    assert meta_path == tmp_path / "stored" / "m" / "sealed.json"
    INP.write_stored(tmp_path, "m", "sealed", {"s00": {"meta": {"tokens": [1]}, "arrays": {"x": np.zeros(1)}}})
    assert INP.write_stored(tmp_path, "m", "sealed", {"s00": {"meta": {"tokens": [1, 2]}, "arrays": {}}}) == [meta_path]
    assert not arrays_path.exists()
    assert INP.stored(tmp_path, "m", "sealed") == {"s00": {"meta": {"tokens": [1, 2]}, "arrays": {}}}


def test_an_array_of_no_listed_input_or_a_dotted_id_is_refused(tmp_path):
    INP.write_stored(tmp_path, "m", "visible", _items())
    INP.stored_paths(tmp_path, "m", "visible")[1].write_text('{"v00": {}}\n')
    with pytest.raises(INP.InputError):
        INP.stored(tmp_path, "m", "visible")
    with pytest.raises(INP.InputError):
        INP.write_stored(tmp_path, "m", "visible", {"v.0": {"meta": {}, "arrays": {}}})


def test_prepare_writes_one_pair_per_split_and_keeps_per_input_forms_private(tmp_path, monkeypatch):
    public, private = tmp_path / "public", tmp_path / "private"
    public.mkdir()
    sources = public / "sources.yaml"
    sources.write_text(yaml.safe_dump({"schema": INP.SOURCES_SCHEMA, "models": {"m": {}}}))
    rows = [{"split": "visible", "id": "v00"}, {"split": "sealed", "id": "s00"}]
    seen = {}

    def preparer(spec, data_root, private_root, python, scratch, items):
        assert items.parent == private_root  # a worker's per-input forms never leave the private root
        seen["items"] = items
        forms = {}
        for row in rows:
            item = INP._item_dir(items, row["split"], row["id"])
            INP._save_npz(item / "x.npz", x=np.full(3, ord(row["id"][0]), np.uint8))
            INP._save_npz(INP._derived_dir(private_root, "m", row["split"], row["id"]) / "inputs.npz", y=np.zeros(2))
            forms[row["split"]] = {row["id"]: {"meta": {"k": row["id"]}, "arrays": INP._load_npz(item / "x.npz")}}
        return {"inputs": rows, "stored": forms, "calibration": []}

    monkeypatch.setitem(INP.PREPARERS, "m", preparer)
    entry = INP.prepare(
        "m", sources, data_root=tmp_path, public=public, private=private, python=Path("python"), scratch=tmp_path
    )
    assert not seen["items"].exists()
    assert entry["visible"] == INP.stored_hashes(public, "m", "visible")
    assert sorted(entry["visible"]) == ["visible.json", "visible.npz"]
    assert entry["sealed"] == INP.stored_hashes(private, "m", "sealed")
    assert sorted(entry["sealed"]) == ["sealed.json", "sealed.npz"]
    assert sorted(entry["derived"]) == ["sealed/s00/inputs.npz", "visible/v00/inputs.npz"]
    assert entry["selection"] == rows
    assert not [p for p in public.rglob("*") if "sealed" in p.name]
    assert INP.stored(private, "m", "sealed")["s00"]["arrays"]["x"].tolist() == [ord("s")] * 3
    manifest = json.loads((public / "MANIFEST.json").read_text())
    assert manifest["models"]["m"] == entry


def test_articles_split_on_top_level_headings_only():
    rows = [" = A = \n", " text a \n", " = = sub = = \n", " more a \n", " = B = \n", " text b \n"]
    assert INP._articles(rows) == ["text a \n = = sub = = \n more a", "text b"]
