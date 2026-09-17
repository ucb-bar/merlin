"""Reproducible, attributed input preparation for the frozen K1 study."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from merlin.common.paths import bench_dir
from merlin.common.yaml import load_yaml


def _module():
    path = bench_dir() / "rvv_paper" / "prepare_inputs.py"
    spec = importlib.util.spec_from_file_location("rvv_paper_prepare_inputs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_input_source_manifest_has_only_active_holdouts_and_full_pins():
    source = load_yaml(bench_dir() / "rvv_paper" / "input_sources_v1.yaml")
    expected = ["gemma2_2b", "tinyllama_1_1b", "smolvla", "resnet50_v1_5", "lstmnetvit"]
    assert source["holdout_models"] == expected
    assert set(source["checkpoints"]) == set(expected)
    assert set(source["inputs"]) == set(expected)
    language = source["language_corpus"]["recommended_source"]
    assert len(language["revision"]) == 40
    assert len(language["file_sha256"]) == 64
    resnet_input = source["inputs"]["resnet50_v1_5"]["recommended_raw_source"]
    assert resnet_input["selection"] == (
        "first_256_jpegs_in_lexicographic_member_order")
    assert resnet_input["archive_bytes"] == 815585330
    assert len(resnet_input["archive_sha256"]) == 64
    assert source["inputs"]["smolvla"]["recommended_raw_source"]["selection"] == (
        "episode_index_0_frame_index_0")
    vitfly_checkpoint = source["checkpoints"]["lstmnetvit"]
    assert vitfly_checkpoint["checkpoint_archive_bytes"] == 51343360
    assert len(vitfly_checkpoint["checkpoint_archive_sha1"]) == 40
    assert len(vitfly_checkpoint["checkpoint_archive_sha256"]) == 64
    assert len(vitfly_checkpoint["checkpoint_member_sha256"]) == 64
    vitfly_input = source["inputs"]["lstmnetvit"]["recommended_raw_source"]
    assert vitfly_input["archive_bytes"] == 2656185217
    assert len(vitfly_input["archive_sha1"]) == 40
    for checkpoint in source["checkpoints"].values():
        revisions = ([checkpoint["revision"]] if "revision" in checkpoint else
                     [item["revision"] for item in checkpoint.get("components", [])])
        revisions += ([checkpoint["source_revision"]]
                      if "source_revision" in checkpoint else [])
        assert all(len(value) == 40 for value in revisions)


def test_dry_run_audit_is_read_only_and_accepts_pinned_local_snapshot(tmp_path, monkeypatch, capsys):
    module = _module()
    source = load_yaml(module.SPEC)
    cfg = source["checkpoints"]["tinyllama_1_1b"]
    cache = tmp_path / "hf"
    root = cache / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
    snapshot = root / "snapshots" / cfg["revision"]
    snapshot.mkdir(parents=True)
    (root / "refs").mkdir()
    # A mutable branch ref may move after the paper pin.  Readiness is based on the explicit pinned
    # snapshot and its byte tree, not on ``main`` still naming the same commit.
    (root / "refs" / "main").write_text("0" * 40, encoding="utf-8")
    for name in cfg["required_files"]:
        (snapshot / name).write_bytes(f"fixture:{name}".encode())
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("An attributed corpus with enough fixture text.", encoding="utf-8")
    out = tmp_path / "generated"
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(out))

    status = module.main([
        "--models", "tinyllama_1_1b", "--hf-cache", str(cache),
        "--language-corpus", str(corpus), "--language-source", "fixture/corpus@sha256",
        "prepare", "--dry-run",
    ])

    report = json.loads(capsys.readouterr().out)
    assert status == 0
    assert report["ready"] is True
    assert report["models"]["tinyllama_1_1b"]["checkpoint"]["revision"] == cfg["revision"]
    assert not out.exists()


def test_deterministic_npz_has_stable_bytes_and_is_numpy_compatible(tmp_path):
    module = _module()
    arrays = {"z": np.arange(8, dtype=np.int64),
              "a": np.arange(6, dtype=np.float32).reshape(2, 3)}
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    module._write_deterministic_npz(first, arrays)
    module._write_deterministic_npz(second, dict(reversed(list(arrays.items()))))
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as data:
        assert sorted(data.files) == ["a", "z"]
        np.testing.assert_array_equal(data["a"], arrays["a"])


def test_smolvla_observation_writer_is_byte_stable_and_records_seeded_noise(tmp_path):
    module = _module()
    rng = np.random.default_rng(0)
    arrays = {
        "image": np.zeros((1, 3, 512, 512), dtype=np.float32),
        "image_mask": np.ones((1,), dtype=np.bool_),
        "language_tokens": np.arange(48, dtype=np.int64)[None],
        "language_mask": np.ones((1, 48), dtype=np.bool_),
        "state": np.zeros((1, 32), dtype=np.float32),
        "noise": rng.standard_normal((1, 50, 32), dtype=np.float32),
    }
    provenance = {"dataset": {"revision": "a" * 40},
                  "noise": {"algorithm": "torch.randn CPU float32", "seed": 0}}
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    one = module.write_smolvla_observation(first, arrays, provenance)
    two = module.write_smolvla_observation(second, dict(reversed(list(arrays.items()))), provenance)

    assert first.read_bytes() == second.read_bytes()
    assert one["output_sha256"] == two["output_sha256"]
    assert one["arrays"]["noise"]["sha256"] == two["arrays"]["noise"]["sha256"]
    persisted = json.loads(Path(one["provenance_path"]).read_text(encoding="utf-8"))
    assert persisted["provenance"]["noise"]["seed"] == module.SMOLVLA_NOISE_SEED


def test_pinned_language_extraction_preserves_rows_and_records_both_hashes(tmp_path):
    module = _module()
    source = tmp_path / "source.parquet"
    source.write_bytes(b"pinned parquet fixture")
    config = {
        "dataset": "fixture/wikitext", "revision": "a" * 40,
        "config": "raw", "split": "test", "file": "test.parquet",
        "file_sha256": _sha(source),
        "extraction": "concatenate text rows in stored order with one newline separator",
    }
    output = tmp_path / "corpus.txt"
    def read_rows(_path):
        return ["", "alpha", "beta", ""], "21.0.0"

    first = module.extract_language_parquet(
        source, config, output=output, row_reader=read_rows)
    first_bytes = output.read_bytes()
    second = module.extract_language_parquet(
        source, config, output=output, row_reader=read_rows)

    assert first_bytes == b"\nalpha\nbeta\n"
    assert output.read_bytes() == first_bytes
    assert first["source_sha256"] == _sha(source)
    assert first["output_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert first["rows"] == 4 and first["pyarrow_version"] == "21.0.0"
    assert second["output_sha256"] == first["output_sha256"]
    provenance = json.loads(Path(first["provenance_path"]).read_text(encoding="utf-8"))
    assert provenance["source_label"] == "fixture/wikitext@" + "a" * 40 + ":raw/test"


def test_language_extraction_rejects_unpinned_parquet_bytes(tmp_path):
    module = _module()
    source = tmp_path / "source.parquet"
    source.write_bytes(b"wrong bytes")
    config = {"file_sha256": "0" * 64}
    try:
        module.extract_language_parquet(source, config, output=tmp_path / "corpus.txt",
                                        row_reader=lambda _path: [])
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("unverified language source unexpectedly passed")


def test_bundle_validation_detects_byte_and_size_corruption(tmp_path):
    module = _module()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"pinned checkpoint")
    artifact = tmp_path / "inputs" / "tinyllama_1_1b" / "token_ids.npy"
    artifact.parent.mkdir(parents=True)
    np.save(artifact, np.arange(160, dtype=np.int64), allow_pickle=False)
    record = {
        "version": 1, "target": "k1", "active_holdouts": list(module.ACTIVE_MODELS),
        "models": {"tinyllama_1_1b": {"artifacts": [{
            "path": artifact.relative_to(tmp_path).as_posix(), "bytes": artifact.stat().st_size,
            "sha256": _sha(artifact),
        }], "environment": {}, "provenance": {"checkpoint": {
            "kind": "torch_hub_file", "path": str(checkpoint), "sha256": _sha(checkpoint),
        }}}},
    }
    (tmp_path / "paper_inputs.json").write_text(
        json.dumps(record, sort_keys=True), encoding="utf-8")
    assert module.validate_bundle(tmp_path) == []
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")
    errors = module.validate_bundle(tmp_path)
    assert any("SHA-256 mismatch" in value for value in errors)
    assert any("byte count mismatch" in value for value in errors)


def test_bundle_validation_rejects_artifact_path_traversal(tmp_path):
    module = _module()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"pinned checkpoint")
    outside = tmp_path.parent / "outside.npy"
    outside.write_bytes(b"must not be admitted")
    record = {
        "version": 1, "target": "k1", "active_holdouts": list(module.ACTIVE_MODELS),
        "models": {"tinyllama_1_1b": {"artifacts": [{
            "path": "../outside.npy", "bytes": outside.stat().st_size,
            "sha256": _sha(outside),
        }], "environment": {}, "provenance": {"checkpoint": {
            "kind": "torch_hub_file", "path": str(checkpoint), "sha256": _sha(checkpoint),
        }}}},
    }
    (tmp_path / "paper_inputs.json").write_text(
        json.dumps(record, sort_keys=True), encoding="utf-8")

    assert any("unsafe artifact path" in value for value in module.validate_bundle(tmp_path))


def test_materialization_emits_relocatable_paper_ready_environment(tmp_path, monkeypatch):
    module = _module()
    source = load_yaml(module.SPEC)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("attributed source bytes", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    tree_sha256, files = module._tree_record(snapshot)
    args = SimpleNamespace(
        spec=module.SPEC, hf_cache=tmp_path / "hf", language_corpus=corpus,
        language_source="fixture/wikitext@revision", smolvla_input_npz=None,
        smolvla_input_source=None, resnet_images_list=None, resnet_input_source=None,
        vitfly_dir=tmp_path / "vitfly", vitfly_checkpoint=None,
        vitfly_session_npz=None, vitfly_session_source=None,
    )
    report = {"ready": True, "models": {"tinyllama_1_1b": {
        "ready": True, "blockers": [],
        "checkpoint": {"kind": "huggingface_snapshot", "cache_path": str(snapshot),
                       "revision": source["checkpoints"]["tinyllama_1_1b"]["revision"],
                       "tree_sha256": tree_sha256, "files": files,
                       "ready": True, "blockers": []},
        "input": {"kind": "token_ids", "source_label": args.language_source,
                  "ready": True, "blockers": []},
    }}}
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "generated"))
    monkeypatch.setattr(module, "_tokenize",
                        lambda *_args: np.arange(160, dtype=np.int64))

    bundle = module.materialize(source, ["tinyllama_1_1b"], report, args)

    assert module.validate_bundle(bundle) == []
    environment = (bundle / "env" / "tinyllama_1_1b.sh").read_text(encoding="utf-8")
    assert "M2M_LLAMA_PAPER_READY=1" in environment
    assert "export M2M_LLAMA_LAYERS=''" in environment
    assert 'M2M_LLAMA_TOKEN_IDS="${BUNDLE_DIR}/inputs/tinyllama_1_1b/token_ids.npy"' in environment
    assert (bundle.parent / "latest").resolve() == bundle.resolve()


def test_input_validators_reject_synthetic_shape_substitutions(tmp_path):
    module = _module()
    bad_smol = tmp_path / "smol.npz"
    np.savez(bad_smol, image=np.zeros((1, 3, 256, 256), np.float32))
    try:
        module.validate_smolvla_npz(bad_smol)
    except ValueError as error:
        assert "omits" in str(error)
    else:
        raise AssertionError("incomplete SmolVLA input unexpectedly passed")

    bad_vitfly = tmp_path / "vitfly.npz"
    np.savez(bad_vitfly, frames=np.zeros((2, 1, 1, 60, 90), np.float32),
             desired_velocity=np.zeros((2, 1, 1), np.float32),
             quaternions=np.zeros((2, 1, 4), np.float32))
    try:
        module.validate_vitfly_session(bad_vitfly)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("short VitFly trajectory unexpectedly passed")


def test_ordered_image_manifest_requires_exact_unique_digests(tmp_path):
    module = _module()
    first, second = tmp_path / "one.png", tmp_path / "two.png"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    manifest = tmp_path / "images.tsv"
    manifest.write_text(f"one.png\t{_sha(first)}\ntwo.png\t{_sha(second)}\n", encoding="utf-8")
    entries = module.parse_image_manifest(manifest, observations=2)
    assert [Path(value["path"]).name for value in entries] == ["one.png", "two.png"]
    manifest.write_text(f"one.png\t{_sha(first)}\none.png\t{_sha(first)}\n", encoding="utf-8")
    try:
        module.parse_image_manifest(manifest, observations=2)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate image input unexpectedly passed")


def test_resnet_zip_extraction_is_hash_pinned_ordered_and_byte_stable(tmp_path):
    module = _module()
    archive_path = tmp_path / "images.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("val2017/0003.jpg", b"third")
        archive.writestr("val2017/0001.jpg", b"first")
        archive.writestr("val2017/0002.jpg", b"second")
        archive.writestr("README.txt", b"ignored")
    config = {
        "dataset": "fixture/coco", "url": "https://example.invalid/images.zip",
        "archive_sha256": _sha(archive_path), "member_prefix": "val2017/",
        "selection": "first_2_jpegs_in_lexicographic_member_order", "observations": 2,
    }
    one = module.extract_resnet_images(archive_path, config, output_dir=tmp_path / "one")
    two = module.extract_resnet_images(archive_path, config, output_dir=tmp_path / "two")

    assert [row["member"] for row in one["members"]] == [
        "val2017/0001.jpg", "val2017/0002.jpg"]
    assert [row["sha256"] for row in one["members"]] == [
        hashlib.sha256(b"first").hexdigest(), hashlib.sha256(b"second").hexdigest()]
    assert [row["sha256"] for row in one["members"]] == [
        row["sha256"] for row in two["members"]]
    assert module.parse_image_manifest(Path(one["manifest_path"]), observations=2)

    bad = dict(config, archive_sha256="0" * 64)
    try:
        module.extract_resnet_images(archive_path, bad, output_dir=tmp_path / "bad")
    except ValueError as error:
        assert "archive SHA-256" in str(error)
    else:
        raise AssertionError("unverified ResNet archive unexpectedly passed")


def test_vitfly_archive_extraction_is_deterministic_and_uses_first_eligible_trajectory(tmp_path):
    module = _module()
    try:
        import cv2
    except ImportError:
        return

    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for trajectory, observations in (("001", 2), ("002", 4), ("003", 5)):
            rows = ["index,timestamp,desired,qw,qx,qy,qz"]
            for index in range(observations):
                timestamp = float(index + 1)
                rows.append(f"{index},{timestamp:.3f},{4 + index},1,0,0,0")
                image = np.full((60, 90), index + int(trajectory), dtype=np.uint8)
                okay, encoded = cv2.imencode(".png", image)
                assert okay
                archive.writestr(f"{trajectory}/{timestamp:.3f}.png", encoded.tobytes())
            archive.writestr(f"{trajectory}/data.csv", "\n".join(rows) + "\n")

    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    raw = tmp_path / "raw"
    with zipfile.ZipFile(archive_path) as archive:
        one = module.extract_vitfly_trajectory(
            archive, first, observations=3,
            archive_provenance={"sha1": "fixture"}, raw_output_dir=raw)
    with zipfile.ZipFile(io.BytesIO(archive_path.read_bytes())) as archive:
        two = module.extract_vitfly_trajectory(
            archive, second, observations=3, archive_provenance={"sha1": "fixture"})

    assert one["trajectory"] == two["trajectory"] == "002"
    assert one["output_sha256"] == two["output_sha256"]
    assert first.read_bytes() == second.read_bytes()
    assert len(one["members"]) == 3
    assert (raw / "data.csv").is_file()
    with np.load(first, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["desired_velocity"][:, 0, 0], [4, 5, 6])
        np.testing.assert_array_equal(data["quaternions"][:, 0], [[1, 0, 0, 0]] * 3)
        assert data["frames"].shape == (3, 1, 1, 60, 90)
