"""Model-side input preparation, run INSIDE a model's own pinned environment (``python input_worker.py``).

Each task reads a JSON request from argv[1] and writes its outputs where the request says. It imports
only the standard library, numpy and that environment's framework (torchvision, transformers or
LeRobot), never Merlin, so the same file runs in every model venv. The preprocessing is each
framework's own: torchvision's ``IMAGENET1K_V2`` transform, the checkpoint's tokenizer, LeRobot's
SmolVLA processors. Arrays are written with :func:`save_npz`, whose bytes depend only on the arrays.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def save_npz(path: str, **arrays: np.ndarray) -> None:
    """A compressed ``.npz`` whose bytes depend only on the arrays (no timestamps), so its sha256 is stable."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue())


def resnet_crops(request: dict) -> dict:
    """Resize (232, bilinear, antialias) and center-crop (224) each image to uint8, as IMAGENET1K_V2 does.

    The model input is rebuilt from the STORED crop -- ``(u8 / 255 - mean) / std`` in float32 -- and must
    equal torchvision's own full transform of the source image bit for bit, so the stored form provably
    determines the input.
    """
    import torch
    from PIL import Image
    from torchvision.models import ResNet50_Weights
    from torchvision.transforms import functional as F

    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    mean, std = np.asarray(transform.mean, np.float32), np.asarray(transform.std, np.float32)
    out = {}
    for item in request["images"]:
        image = Image.open(item["path"]).convert("RGB")
        resized = F.resize(image, transform.resize_size, interpolation=transform.interpolation, antialias=True)
        cropped = np.asarray(F.center_crop(resized, transform.crop_size), dtype=np.uint8)
        save_npz(item["stored"], image=cropped)
        with np.load(item["stored"]) as stored:
            pixels = np.asarray(stored["image"], dtype=np.uint8).astype(np.float32)
        x = (pixels / np.float32(255.0) - mean) / std
        tensor = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None]).astype(np.float32)
        reference = transform(image).unsqueeze(0).to(torch.float32).numpy()
        if not np.array_equal(tensor, reference):
            raise ValueError(
                f"{item['id']}: the stored crop does not reproduce the IMAGENET1K_V2 tensor exactly "
                f"(max |diff| {float(np.max(np.abs(tensor - reference)))})"
            )
        save_npz(item["inputs"], in0=tensor)
        out[item["id"]] = {"stored": item["stored"], "inputs": item["inputs"]}
    return {
        "transform": {
            "resize": list(transform.resize_size),
            "crop": list(transform.crop_size),
            "mean": list(transform.mean),
            "std": list(transform.std),
            "interpolation": str(transform.interpolation),
        },
        "images": out,
    }


def parquet_rows(request: dict) -> dict:
    import pyarrow.parquet as parquet

    table = parquet.read_table(request["path"], columns=["text"])
    rows = table.column("text").combine_chunks().to_pylist()
    Path(request["out"]).write_text(json.dumps(rows))
    return {"rows": len(rows)}


def tokenize_segments(request: dict) -> dict:
    """Token ids for each text, from the checkpoint's own tokenizer, no special tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(request["tokenizer"], local_files_only=True)
    out = [
        tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"] for text in request["texts"]
    ]
    Path(request["out"]).write_text(json.dumps(out))
    return {"segments": len(out)}


def _smolvla_config(policy_snapshot: str):
    """The pinned config through draccus, as the attributed extractor reads it (LeRobot 0.6's
    subclass/choice parser does not accept the checkpoint's ``type`` key)."""
    import draccus
    from lerobot.policies.smolvla import SmolVLAConfig

    raw = json.loads((Path(policy_snapshot) / "config.json").read_text())
    if raw.pop("type", None) != "smolvla":
        raise ValueError("pinned policy config is not SmolVLA")
    with tempfile.NamedTemporaryFile("w+", suffix=".json") as stream:
        json.dump(raw, stream)
        stream.flush()
        with draccus.config_type("json"):
            return draccus.parse(SmolVLAConfig, stream.name, args=[])


def smolvla_frames(request: dict) -> dict:
    """Raw frames/state and the preprocessed model inputs for (episode, frame) pairs, via LeRobot."""
    import torch
    import transformers.utils

    if not hasattr(transformers.utils, "torch_compilable_check"):
        transformers.utils.torch_compilable_check = lambda function: function
    from lerobot.configs import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla import SmolVLAPolicy, make_smolvla_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features

    policy = _smolvla_config(request["policy_snapshot"])
    policy.device = "cpu"
    episodes = sorted({int(item["episode"]) for item in request["items"]})
    dataset = LeRobotDataset(
        request["dataset"],
        root=request["dataset_root"],
        episodes=episodes,
        revision=request["revision"],
        download_videos=False,
        video_backend="pyav",
    )
    features = dataset_to_policy_features(dataset.meta.features)
    policy.input_features = {n: v for n, v in features.items() if v.type is not FeatureType.ACTION}
    policy.output_features = {n: v for n, v in features.items() if v.type is FeatureType.ACTION}
    policy.vlm_model_name = request["vlm_snapshot"]
    preprocessor, _ = make_smolvla_pre_post_processors(policy, dataset.meta.stats)
    shell = object.__new__(SmolVLAPolicy)
    shell.config = policy
    index_of = {}
    for i in range(len(dataset)):
        row = dataset.hf_dataset[i]
        index_of[(int(row["episode_index"]), int(row["frame_index"]))] = i
    out = {}
    for item in request["items"]:
        sample = dataset[index_of[(int(item["episode"]), int(item["frame"]))]]
        raw = sample["observation.images.top"].detach().cpu().numpy()
        frame = np.clip(np.rint(np.transpose(raw, (1, 2, 0)) * 255.0), 0, 255).astype(np.uint8)
        save_npz(item["stored"], frame=frame, state=sample["observation.state"].detach().cpu().numpy())
        processed = preprocessor(sample)
        images, masks = SmolVLAPolicy.prepare_images(shell, processed)
        state = SmolVLAPolicy.prepare_state(shell, processed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(item["noise_seed"]))
        noise = torch.randn(
            (1, int(policy.chunk_size), int(policy.max_action_dim)), generator=generator, dtype=torch.float32
        )
        save_npz(
            item["arrays"],
            image=images[0].detach().cpu().numpy().astype(np.float32),
            image_mask=masks[0].detach().cpu().numpy().astype(np.bool_),
            language_tokens=processed["observation.language.tokens"].detach().cpu().numpy().astype(np.int64),
            language_mask=processed["observation.language.attention_mask"].detach().cpu().numpy().astype(np.bool_),
            state=state.detach().cpu().numpy().astype(np.float32),
            noise=noise.numpy(),
        )
        out[item["id"]] = {"task": str(sample["task"]), "raw_state": sample["observation.state"].tolist()}
    return {"items": out}


TASKS = {
    "resnet_crops": resnet_crops,
    "parquet_rows": parquet_rows,
    "tokenize_segments": tokenize_segments,
    "smolvla_frames": smolvla_frames,
}


def main(argv: list[str]) -> int:
    request = json.loads(Path(argv[1]).read_text())
    result = TASKS[request["task"]](request)
    Path(request["result"]).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
