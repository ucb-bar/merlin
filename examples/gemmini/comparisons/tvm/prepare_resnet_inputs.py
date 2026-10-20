"""Prepare source-bound Imagenette320 development inputs without downloads or extraction."""

import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import tarfile


CLASSES = (
    ("n01440764", 0, "tench"), ("n02102040", 217, "English springer"),
    ("n02979186", 482, "cassette player"), ("n03000684", 491, "chain saw"),
    ("n03028079", 497, "church"), ("n03394916", 566, "French horn"),
    ("n03417042", 569, "garbage truck"), ("n03425413", 571, "gas pump"),
    ("n03445777", 574, "golf ball"), ("n03888257", 701, "parachute"),
)
ARCHIVE_ROOT = "imagenette2-320"
MAX_ARCHIVE_BYTES = 512 * 1024**2
MAX_MEMBERS = 30000
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3
MAX_IMAGE_BYTES = 16 * 1024**2
MAX_IMAGE_PIXELS = 16 * 1024**2
MAX_SELECTED_IMAGES = 200


class BoundedTarInfo(tarfile.TarInfo):
    @classmethod
    def frombuf(cls, buf, encoding, errors):
        member = super().frombuf(buf, encoding, errors)
        if member.size < 0 or member.size > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("archive member bytes exceed bound")
        metadata = (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK)
        if member.type in metadata and member.size > 64 * 1024:
            raise ValueError("archive extended metadata exceeds bound")
        return member


def safe_path(value, seen=None):
    """Reject restricted components before filesystem operations, including symlink targets."""
    path = Path(value)
    if any(token in part.lower() for part in path.parts for token in ("hammer", "vlsi")):
        raise ValueError("restricted path component")
    path = Path(os.path.abspath(path))
    seen = set() if seen is None else seen
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            if str(current) in seen:
                raise ValueError("symlink cycle")
            seen.add(str(current))
            target = Path(os.readlink(current))
            current = safe_path(target if target.is_absolute() else current.parent / target, seen)
    return current


def file_identity(path):
    path = safe_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def tensor_hash(value):
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode() + b"\0" + json.dumps(list(value.shape)).encode() + b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def inventory(archive):
    """Admit the entire header inventory before opening any image payload."""
    groups = {(split, wnid): [] for split in ("train", "val") for wnid, _, _ in CLASSES}
    names, total = set(), 0
    known_classes = {wnid for wnid, _, _ in CLASSES}
    for count, member in enumerate(archive, 1):
        if count > MAX_MEMBERS:
            raise ValueError("archive member count exceeds bound")
        parts = member.name.rstrip("/").split("/")
        if (not parts or any(part in ("", ".", "..") or "\\" in part for part in parts)
                or any(token in part.lower() for part in parts for token in ("hammer", "vlsi"))):
            raise ValueError("unsafe archive member name")
        name = "/".join(parts)
        if name in names:
            raise ValueError("duplicate archive member name")
        names.add(name)
        if parts[0] != ARCHIVE_ROOT:
            raise ValueError("expected Imagenette320 archive root")
        if not (member.isdir() or member.isfile()) or member.sparse is not None:
            raise ValueError("archive links and special members are forbidden")
        if type(member.size) is not int or member.size < 0:
            raise ValueError("invalid archive member size")
        total += member.size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("archive uncompressed bytes exceed bound")
        if member.isdir():
            if (len(parts) > 3 or (len(parts) >= 2 and parts[1] not in ("train", "val"))
                    or (len(parts) == 3 and parts[2] not in known_classes)):
                raise ValueError("unexpected archive directory")
            continue
        if len(parts) == 2 and Path(parts[1]).suffix.lower() in (".csv", ".txt"):
            continue
        if (len(parts) != 4 or parts[1] not in ("train", "val") or parts[2] not in known_classes
                or Path(parts[3]).suffix.lower() not in (".jpg", ".jpeg")):
            raise ValueError("unexpected image member")
        if not 0 < member.size <= MAX_IMAGE_BYTES:
            raise ValueError("image member bytes exceed bound")
        groups[(parts[1], parts[2])].append(member)
    return groups, {"members": len(names), "uncompressed_bytes": total}


def transform_contract(weights):
    transform = weights.transforms()
    categories = list(weights.meta["categories"])
    if len(categories) != 1000 or any(not isinstance(name, str) or not name for name in categories):
        raise ValueError("expected 1000 ImageNet category names")
    for wnid, index, name in CLASSES:
        if categories[index] != name:
            raise ValueError(f"category mapping mismatch for {wnid}")
    contract = {
        "weights": "ResNet50_Weights.IMAGENET1K_V2", "resize_size": list(transform.resize_size),
        "crop_size": list(transform.crop_size), "mean": list(transform.mean), "std": list(transform.std),
        "interpolation": transform.interpolation.value, "antialias": transform.antialias,
        "input_mode": "PIL RGB; no augmentation or EXIF orientation adjustment", "output_dtype": "float32",
    }
    if (contract["resize_size"] != [232] or contract["crop_size"] != [224]
            or contract["mean"] != [0.485, 0.456, 0.406] or contract["std"] != [0.229, 0.224, 0.225]
            or contract["interpolation"] != "bilinear" or contract["antialias"] is not True):
        raise ValueError("installed V2 transform differs from the declared contract")
    return transform, categories, contract


def dependency_paths(torch):
    modules = (
        "torchvision.models.resnet", "torchvision.models._meta", "torchvision.transforms._presets",
        "torchvision.transforms.functional", "torchvision.transforms._functional_pil",
        "torchvision.transforms._functional_tensor", "PIL.Image", "PIL._imaging", "torch",
        "numpy", "numpy._core._multiarray_umath", "tarfile", "gzip",
    )
    paths = [safe_path(__file__)]
    paths.extend(safe_path(importlib.import_module(name).__file__) for name in modules)
    paths.append(safe_path(Path(torch.__file__).parent / "lib/libtorch_cpu.so"))
    return sorted(set(paths), key=str)


def select_members(groups, calibration_per_class, evaluation_per_class, final_per_class, selection_seed):
    """Keep lexical development VAL; select TRAIN/final by source-name hashes.

    Final candidates exclude every development member before deterministic
    selection. Membership uses no image labels, predictions or quality result.
    """
    selected = {"calibration": [], "evaluation": []}
    if final_per_class:
        selected["final"] = []
    def ranked(members, role):
        if selection_seed is None:
            return sorted(members, key=lambda member: member.name)
        return sorted(members, key=lambda member: (hashlib.sha256((selection_seed + "\0" + role + "\0" + member.name).encode()).hexdigest(), member.name))
    for wnid, label, category in CLASSES:
        train = ranked(groups[("train", wnid)], "calibration")
        val = sorted(groups[("val", wnid)], key=lambda member: member.name)
        if len(train) < calibration_per_class or len(val) < evaluation_per_class + final_per_class:
            raise ValueError(f"missing samples: train/val/{wnid}")
        roles = {"calibration": train[:calibration_per_class], "evaluation": val[:evaluation_per_class]}
        if final_per_class:
            roles["final"] = ranked(val[evaluation_per_class:], "final")[:final_per_class]
        for role, members in roles.items():
            selected[role].extend((member, wnid, label, category) for member in members)
    return selected


def prepare_inputs(archive_path, output_dir, *, calibration_per_class=1, evaluation_per_class=2, final_per_class=0, selection_seed=None):
    for count in (calibration_per_class, evaluation_per_class):
        if type(count) is not int or count <= 0:
            raise ValueError("per-class counts must be positive integers")
    if type(final_per_class) is not int or final_per_class < 0:
        raise ValueError("final per-class count must be a nonnegative integer")
    if selection_seed is not None and (not isinstance(selection_seed, str) or not selection_seed.strip()):
        raise ValueError("selection seed must be a nonempty string")
    if len(CLASSES) * (calibration_per_class + evaluation_per_class + final_per_class) > MAX_SELECTED_IMAGES:
        raise ValueError("selected image count exceeds bound")
    archive_path, output_dir = safe_path(archive_path), safe_path(output_dir)
    if output_dir.exists():
        raise ValueError("output directory must be new")
    if not archive_path.is_file() or not 0 < archive_path.stat().st_size <= MAX_ARCHIVE_BYTES:
        raise ValueError("local archive size exceeds bound or is not a file")
    before_archive = file_identity(archive_path)
    with tarfile.open(archive_path, "r:gz", tarinfo=BoundedTarInfo) as archive:
        groups, archive_inventory = inventory(archive)
        selected = select_members(groups, calibration_per_class, evaluation_per_class, final_per_class, selection_seed)

        import numpy as np
        import PIL
        from PIL import Image
        import torch
        import torchvision
        from torchvision.models import ResNet50_Weights

        transform, categories, contract = transform_contract(ResNet50_Weights.IMAGENET1K_V2)
        dependencies = dependency_paths(torch)
        before_sources = [file_identity(path) for path in dependencies]
        arrays = {role: np.empty((len(entries), 1, 3, 224, 224), dtype=np.float32) for role, entries in selected.items()}
        samples = {role: [None] * len(entries) for role, entries in selected.items()}
        observed = {kind: {} for kind in ("jpeg", "decoded", "transformed")}
        # One rewind and monotonic gzip reads, regardless of the declared sample
        # order. Seeking each randomly selected member separately can decompress
        # the full archive hundreds of times. Arrays retain the selection order.
        payload_order = sorted((entry[0].offset_data, role, index, entry)
                               for role, entries in selected.items() for index, entry in enumerate(entries))
        for _, role, index, (member, wnid, label, category) in payload_order:
            with archive.extractfile(member) as source:
                payload = source.read(MAX_IMAGE_BYTES + 1)
            if len(payload) != member.size:
                raise ValueError("image payload length does not match archive header")
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "JPEG" or not 0 < image.width * image.height <= MAX_IMAGE_PIXELS:
                    raise ValueError("expected bounded JPEG image")
                original_size = list(image.size)
                rgb = image.convert("RGB")
                pixels = np.array(rgb, dtype=np.uint8, copy=True)
                with torch.no_grad():
                    value = transform(rgb).cpu().numpy().copy()
            if value.dtype != np.float32 or value.shape != (3, 224, 224) or not np.isfinite(value).all():
                raise ValueError("transform output violates float32 RGB224 contract")
            hashes = {"jpeg": hashlib.sha256(payload).hexdigest(), "decoded": tensor_hash(pixels), "transformed": tensor_hash(value)}
            for kind, digest in hashes.items():
                if digest in observed[kind]:
                    raise ValueError(f"duplicate or overlapping {kind} content: {member.name} and {observed[kind][digest]}")
                observed[kind][digest] = member.name
            arrays[role][index, 0] = value
            samples[role][index] = {"id": member.name, "split": "train" if role == "calibration" else "val", "wnid": wnid,
                "label": label, "category": category, "member_bytes": member.size, "original_size": original_size,
                "jpeg_sha256": hashes["jpeg"], "decoded_rgb_sha256": hashes["decoded"], "transformed_sha256": hashes["transformed"]}

    def check_bindings():
        if file_identity(archive_path) != before_archive:
            raise ValueError("archive changed during preparation")
        if [file_identity(path) for path in dependencies] != before_sources:
            raise ValueError("source or transform dependency changed during preparation")

    check_bindings()
    output_dir.mkdir(parents=True)
    for role, images in arrays.items():
        np.savez(output_dir / f"{role}.npz", images=images)
    outputs = {role: file_identity(output_dir / f"{role}.npz") for role in arrays}
    class_ids = [f"imagenet1k:{index:04d}:{category}" for index, category in enumerate(categories)]
    for role in ("evaluation", "final"):
        if role not in samples:
            continue
        labels = {"input_sha256": outputs[role]["sha256"], "class_ids": class_ids,
                  "samples": [{"id": item["id"], "label": item["label"]} for item in samples[role]]}
        label_name = "labels" if role == "evaluation" else "final_labels"
        labels_path = output_dir / f"{label_name}.json"
        labels_path.write_text(json.dumps(labels, indent=2) + "\n")
        outputs[label_name] = file_identity(labels_path)
    check_bindings()
    report = {
        "status": "prepared", "development_only": True, "paper_quality_approved": False,
        "scope": "Imagenette320 ten-class development subset; not full ImageNet accuracy qualification",
        "dataset_declaration": ARCHIVE_ROOT, "archive": before_archive, "archive_inventory": archive_inventory,
        "selection": "class table order, then lexicographically sorted archive member names; no random sampling",
        "calibration_per_class": calibration_per_class, "evaluation_per_class": evaluation_per_class,
        "final_per_class": final_per_class, "selection_seed": selection_seed,
        "role_contract": {"calibration": "TRAIN only; recipe fitting", "evaluation": "lexical VAL development; may already have been exercised", "final": "remaining VAL after excluding development; freeze before evaluating; no recipe fitting"},
        "content_disjointness": "JPEG bytes, decoded RGB pixels, and transformed float32 tensors are unique across both streams",
        "class_head": {"classes": 1000, "categories": categories, "class_ids": class_ids,
                       "subset": [{"wnid": wnid, "index": index, "category": name} for wnid, index, name in CLASSES]},
        "transform": contract, "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__, "Pillow": PIL.__version__},
        "source_bindings": before_sources, "bindings_checked_before_and_after": True, "samples": samples, "outputs": outputs,
    }
    if selection_seed is not None:
        report["selection"] = "class table order; development VAL lexical; TRAIN and remaining final VAL ordered by SHA256(seed + NUL + role + NUL + member_name), name tie-break"
    (output_dir / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="local imagenette2-320.tgz; never downloaded or extracted")
    parser.add_argument("--output-dir", required=True, help="new directory for NPZ streams and manifests")
    parser.add_argument("--calibration-per-class", type=int, default=1)
    parser.add_argument("--evaluation-per-class", type=int, default=2)
    parser.add_argument("--final-per-class", type=int, default=0, help="separate untouched VAL stream after excluding development members")
    parser.add_argument("--selection-seed", help="stable source-name hash selection for TRAIN and final VAL; development VAL stays lexical")
    args = parser.parse_args()
    report = prepare_inputs(args.archive, args.output_dir, calibration_per_class=args.calibration_per_class, evaluation_per_class=args.evaluation_per_class, final_per_class=args.final_per_class, selection_seed=args.selection_seed)
    print(json.dumps({"status": report["status"], "output_dir": str(safe_path(args.output_dir)), "paper_quality_approved": False}))


if __name__ == "__main__":
    main()
