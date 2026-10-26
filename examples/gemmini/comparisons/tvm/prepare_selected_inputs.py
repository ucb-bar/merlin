"""Prepare public EL4 inputs and calibration data without editing frozen manifests."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from merlin.model_check.input_worker import _smolvla_config, save_npz
from merlin.model_check.inputs import _articles, stored


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tiny_inputs(public, source, output):
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    train = source / "tiny/train.parquet"
    expected = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
    if digest(train) != expected:
        raise ValueError("Pinned WikiText training source differs")
    tokenizer = AutoTokenizer.from_pretrained(source / "tiny/model", local_files_only=True)
    articles = _articles(parquet.read_table(train, columns=["text"]).column("text").to_pylist())
    selected = []
    for article in articles:
        ids = tokenizer(article, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        if len(ids) >= 160:
            selected.append(ids[:160])
        if len(selected) == 128:
            break
    tokens = np.asarray(selected, np.int64)
    roster = json.loads((public / "calibration/tiny_llama.json").read_text())["items"]
    if tokens.shape != (128, 160) or len(roster) != 128:
        raise ValueError("Expected exactly 128 attributed 160-token training segments")
    for ids, row in zip(tokens, roster, strict=True):
        if hashlib.sha256(ids.tobytes()).hexdigest() != row["sha256"]:
            raise ValueError("Calibration article differs from the frozen roster")
    save_npz(str(output / "tiny-calibration.npz"), tokens=tokens)
    ids = np.asarray(json.loads((public / "inputs/tiny_llama/visible.json").read_text())["v00"]["tokens"], np.int64)
    save_npz(str(output / "tiny-v00.npz"), tokens=ids)
    if digest(output / "tiny-v00.npz") != "065d978c1f302bc4c2ee8f74dc6823d72b81224baf30a51b877919cbfa450629":
        raise ValueError("Selected TinyLlama v00 input does not reproduce the protocol hash")
    return dict(calibration_segments=128, calibration_roster_exact=True, selected_input_exact=True)


def smol_config(models, metadata):
    from lerobot.configs import FeatureType
    from lerobot.utils.feature_utils import dataset_to_policy_features

    config = _smolvla_config(str(models / "smolvla"))
    config.device = "cpu"
    config.vlm_model_name = str((models / "smolvlm").resolve())
    features = dataset_to_policy_features(json.loads((metadata / "meta/info.json").read_text())["features"])
    config.input_features = {k: v for k, v in features.items() if v.type is not FeatureType.ACTION}
    config.output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    return config


def smol_inputs(public, source, models, metadata, output):
    import av
    import pyarrow.parquet as parquet
    import torch
    from lerobot.datasets.io_utils import load_stats
    from lerobot.policies.smolvla import SmolVLAPolicy, make_smolvla_pre_post_processors

    config = smol_config(models, metadata)
    processor, _ = make_smolvla_pre_post_processors(config, load_stats(metadata))
    shell = object.__new__(SmolVLAPolicy)
    shell.config = config
    visible = stored(public, "smolvla", "visible")["v00"]
    with av.open(str(source / "aloha/top.mp4")) as video:
        first_frame = next(video.decode(video=0)).to_ndarray(format="rgb24")
    rows = parquet.read_table(source / "aloha/data.parquet").to_pylist()
    row = next(row for row in rows if row["episode_index"] == 0 and row["frame_index"] == 0)
    cases = dict(
        calibration=(first_frame, np.asarray(row["observation.state"], np.float32), 0),
        v00=(visible["arrays"]["frame"], visible["arrays"]["state"], visible["meta"]["noise_seed"]),
    )
    for name, (frame, state, seed) in cases.items():
        sample = {
            "task": visible["meta"]["task"],
            "observation.images.top": torch.from_numpy(frame.copy()).permute(2, 0, 1).contiguous().float() / 255,
            "observation.state": torch.from_numpy(state.copy()),
        }
        processed = processor(sample)
        images, masks = SmolVLAPolicy.prepare_images(shell, processed)
        if len(images) != 1 or len(masks) != 1:
            raise ValueError("Selected profile requires one actual camera")
        arrays = dict(
            image=images[0].detach().numpy().astype(np.float32),
            image_mask=masks[0].detach().numpy().astype(np.bool_),
            language_tokens=processed["observation.language.tokens"].detach().numpy().astype(np.int64),
            language_mask=processed["observation.language.attention_mask"].detach().numpy().astype(np.bool_),
            state=SmolVLAPolicy.prepare_state(shell, processed).detach().numpy().astype(np.float32),
            noise=torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(seed)).numpy(),
        )
        save_npz(str(output / ("smol-" + name + ".npz")), **arrays)
    expected = "8c4c0dca184abf8da7af74c187531cecc172c3eb50f51aa7b6d6d0c61fef3daf"
    return dict(
        capture=dict(episode=0, frame=0, noise_seed=0),
        evaluation=dict(episode=0, frame=100, noise_seed=1000),
        expected_historical_input_sha256=expected,
        current_input_sha256=digest(output / "smol-v00.npz"),
        historical_input_exact=digest(output / "smol-v00.npz") == expected,
        release="Fresh source-bound preparation; preserve historical input hash discrepancy",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "source", "models", "metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = dict(tiny_llama=tiny_inputs(args.public, args.source, args.output))
    (args.output / "preparation.json").write_text(json.dumps(reports, indent=2) + "\n")
    reports["smolvla"] = smol_inputs(args.public, args.source, args.models, args.metadata, args.output)
    for name in ("meta/info.json", "meta/stats.json"):
        (args.output / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.metadata / name, args.output / name)
    reports["files"] = {p.name: digest(p) for p in sorted(args.output.glob("*.npz"))}
    (args.output / "preparation.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
