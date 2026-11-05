#!/usr/bin/env python3
"""Prepare attributed SO100 observations using the original checkpoint processor.

This is an offline preparation recipe, not model/task qualification. The chosen
camera assignment is explicit; its identity during base-model training remains
unverified. No artificial third view or dataset-wide normalization is introduced.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import smolvla_session as factory


DATASET_ID = "lerobot/svla_so100_pickplace"
# Before the July 2026 physical-angle conversion; includes the integer-fps fix.
DATASET_REVISION = "3d6d687a25cdf1565cdf24550814f72d999a861d"
DATASET_SOURCE = f"https://huggingface.co/datasets/{DATASET_ID}/tree/{DATASET_REVISION}"
PINNED_FILES = {
    "data/chunk-000/file-000.parquet": "615e9fafcd9378b5df82e8969936e48ad521848e02f843b3a1715e3b64e1f262",
    "meta/episodes/chunk-000/file-000.parquet": "e1890fd52588a937cbab76f1efd759623f2b2f77326d1133249a9cd1d924bdc0",
}
PIN_SOURCE = "https://huggingface.co/datasets/lerobot/svla_so100_pickplace/commit/728583b5eaf9e739a7f119e2def466fa1d552402"
CAMERA_MAP = {"observation.images.top": "observation.images.camera1",
              "observation.images.wrist": "observation.images.camera2"}
STATE_NAMES = ["main_shoulder_pan", "main_shoulder_lift", "main_elbow_flex",
               "main_wrist_flex", "main_wrist_roll", "main_gripper"]


def episode_split(episode):
    """Disjoint whole episodes: 30 TRAIN, 10 development, 10 held out.

    Interleaving spreads selections across the recording. Position-group identity
    is not established, so this does not claim stratification or unseen positions.
    """
    if type(episode) is not int or not 0 <= episode < 50:
        raise ValueError("Expected an official SO100 episode index in [0,50)")
    return "train" if episode % 5 < 3 else "development" if episode % 5 == 3 else "heldout"


def validate_schema(info):
    if any(info.get(key) != value for key, value in {
            "codebase_version": "v3.0", "robot_type": "so100", "fps": 30,
            "total_episodes": 50, "total_frames": 19631, "total_tasks": 1}.items()):
        raise ValueError("Official SO100 dataset geometry/version differs")
    features = info.get("features", {})
    cameras = {name for name in features if name.startswith("observation.images.")}
    if cameras != set(CAMERA_MAP):
        raise ValueError("SO100 requires exactly its recorded top and wrist cameras")
    for name in CAMERA_MAP:
        feature = features[name]
        if feature.get("dtype") != "video" or feature.get("shape") != [480, 640, 3] or \
                feature.get("names") != ["height", "width", "channels"]:
            raise ValueError("Recorded camera schema differs")
    for name in ("observation.state", "action"):
        feature = features.get(name, {})
        if feature.get("dtype") != "float32" or feature.get("shape") != [6] or feature.get("names") != STATE_NAMES:
            raise ValueError("SO100 state/action coordinate schema differs")


def verify_snapshot(root):
    root = factory.safe_path(root)
    if not root.is_dir():
        raise FileNotFoundError("Supply the complete pinned dataset snapshot locally")
    for name, expected in PINNED_FILES.items():
        identity = factory.file_identity(root / name, factory.safe_path)
        if identity["sha256"] != expected:
            raise ValueError("Dataset bytes differ from the pinned preconversion SO100 revision")
    info = json.loads(factory.safe_path(root / "meta/info.json").read_text())
    validate_schema(info)
    files = factory.directory_artifacts(root)
    for camera in CAMERA_MAP:
        video = factory.safe_path(root / "videos" / camera / "chunk-000/file-000.mp4")
        if not video.is_file():
            raise FileNotFoundError(f"Recorded camera video is absent: {camera}")
    return info, files


def prepare_tensors(policy, preprocessor, observation, *, noise_seed, torch):
    """Use checkpoint preprocessing and upstream image/state preparation exactly."""
    if type(noise_seed) is not int or not 0 <= noise_seed < 2**63:
        raise ValueError("Noise seed must be an integer in [0,2**63)")
    task = observation.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Recorded observation requires its attributed nonempty task text")
    batch = {"task": task}
    state = observation["observation.state"]
    if state.shape != (6,) or state.dtype != torch.float32 or not torch.isfinite(state).all():
        raise ValueError("Recorded state must preserve six finite FP32 coordinates")
    batch["observation.state"] = state.clone()
    configured = set(policy.config.image_features)
    if not set(CAMERA_MAP.values()).issubset(configured) or policy.config.empty_cameras != 0:
        raise ValueError("Recipe requires configured camera1/2 and no invented empty cameras")
    for source, destination in CAMERA_MAP.items():
        image = observation[source]
        if image.shape != (3, 480, 640) or image.dtype != torch.float32 or \
                not torch.isfinite(image).all() or not torch.all((image >= 0) & (image <= 1)):
            raise ValueError("Decoded recorded cameras must preserve finite FP32 CHW [0,1] pixels")
        batch[destination] = image.clone()
    processed = preprocessor(batch)
    # Omit action labels from the processor/model observation input.
    images, masks = policy.prepare_images(processed)
    if len(images) != 2 or len(masks) != 2:
        raise ValueError("Upstream preparation did not preserve exactly two recorded cameras")
    inputs = (torch.stack(images, dim=1), torch.stack(masks, dim=1),
              processed["observation.language.tokens"], processed["observation.language.attention_mask"],
              policy.prepare_state(processed), torch.randn(
                  1, policy.config.chunk_size, policy.config.max_action_dim,
                  generator=torch.Generator(device="cpu").manual_seed(noise_seed)))
    return inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, required=True, help="Global recorded frame index")
    parser.add_argument("--split", choices=("train", "development", "heldout"), required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav",
                        help="Explicit supported LeRobot decoder; recorded in the manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("dataset_root", "checkpoint_dir", "backbone_dir", "output_dir"):
        setattr(args, name, factory.safe_path(getattr(args, name)))
    from merlin.common.paths import artifacts_dir
    if not args.output_dir.is_relative_to(factory.safe_path(artifacts_dir())) or args.output_dir.exists():
        raise ValueError("Choose a new output directory inside the configured artifacts root")
    if importlib.metadata.version("lerobot") != "0.5.1":
        raise ValueError("This observation recipe pins lerobot==0.5.1")
    config_json, checkpoint_files = factory.checkpoint_artifacts(args.checkpoint_dir)
    info, dataset_files = verify_snapshot(args.dataset_root)
    if not 0 <= args.frame_index < info["total_frames"]:
        raise ValueError("Global frame index is outside the pinned recording")
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla import processor_smolvla  # Register the saved newline step.
    from lerobot.processor import PolicyProcessorPipeline
    from verify_session import offline

    with offline():
        dataset = LeRobotDataset(DATASET_ID, root=args.dataset_root, revision=DATASET_REVISION,
                                 image_transforms=None, delta_timestamps=None, video_backend=args.video_backend)
        observation = dataset[args.frame_index]
        episode = int(observation["episode_index"].item())
        if episode_split(episode) != args.split:
            raise ValueError("Recorded episode belongs to a different frozen preparation split")
        config = PreTrainedConfig.from_pretrained(str(args.checkpoint_dir), local_files_only=True)
        config.device, config.vlm_model_name = "cpu", str(args.backbone_dir)
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            args.checkpoint_dir, config_filename="policy_preprocessor.json", local_files_only=True,
            overrides={"device_processor": {"device": "cpu"},
                       "tokenizer_processor": {"tokenizer_name": str(args.backbone_dir)}})
        # Preparation uses upstream methods without loading a second full model.
        policy = SimpleNamespace(config=config)
        policy.prepare_images = lambda batch: SmolVLAPolicy.prepare_images(policy, batch)
        policy.prepare_state = lambda batch: SmolVLAPolicy.prepare_state(policy, batch)
        inputs = prepare_tensors(policy, preprocessor, observation, noise_seed=args.noise_seed, torch=torch)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = factory.safe_path(args.output_dir / "observation.npz")
    names = ("images", "image_masks", "language_tokens", "language_mask", "state", "noise")
    np.savez_compressed(output, **{name: value.cpu().numpy() for name, value in zip(names, inputs, strict=True)})
    checked = factory.fixture_inputs(config, 2, input_npz=output)
    if any(not torch.equal(left, right) for left, right in zip(inputs, checked, strict=True)):
        raise AssertionError("Prepared observation did not preserve its tensor ABI on reload")
    sources = [factory.file_identity(path, factory.safe_path) for path in (
        __file__, factory.__file__, processor_smolvla.__file__,
        __import__(SmolVLAPolicy.__module__, fromlist=["__file__"]).__file__)]
    backbone_files = factory.directory_artifacts(args.backbone_dir)
    bindings = checkpoint_files + dataset_files + backbone_files + sources
    if any(factory.file_identity(item["path"], factory.safe_path) != item for item in bindings):
        raise AssertionError("Observation/checkpoint/processor source bytes changed during preparation")
    manifest = {
        "version": 1, "fixture_kind": "dataset", "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION, "dataset_source": DATASET_SOURCE,
        "pinned_parquet_verified": True, "pin_source": PIN_SOURCE,
        "video_upstream_hashes_verified": False, "source_closure_verified": False,
        "video_decoder": {"backend": args.video_backend,
                          "dependencies": {name: importlib.metadata.version(name) for name in
                                           ("av" if args.video_backend == "pyav" else "torchcodec", "torchvision", "torch", "lerobot")}},
        "schema_verified": True, "selected_cameras": list(CAMERA_MAP.values()),
        "camera_mapping": CAMERA_MAP, "training_camera_mapping_verified": False,
        "state_names": STATE_NAMES, "split": args.split,
        "split_rule": "episode_index modulo5:0/1/2 TRAIN,3 development,4 heldout; no position stratification claim",
        "episode_index": episode, "frame_index": int(observation["frame_index"].item()),
        "global_index": args.frame_index, "task": observation["task"], "noise_seed": args.noise_seed,
        "checkpoint_revision": factory.MODEL_REVISION, "precision": "original_lerobot",
        "normalization": "saved checkpoint processor/state; no dataset-wide statistics refit",
        "processor_overrides": {"device": "cpu", "tokenizer_name": str(args.backbone_dir)},
        "backbone_revision_verified": False, "input_fixture": factory.file_identity(output, factory.safe_path),
        "bindings": bindings, "model_quality_qualified": False, "active_gate_changed": False,
    }
    factory.safe_path(args.output_dir / "observation.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"prepared": str(output), "split": args.split, "cameras": manifest["selected_cameras"]}))
    return manifest


if __name__ == "__main__":
    main()
