"""Native ResNet checkpoint and image admission, with no compiler dependency.

The numerical admission functions retain the frozen verifier's behavior. Physical
paths use the current Atlas examples' stricter lexical and symlink checks.
"""

import hashlib
import os
from pathlib import Path
import stat


def safe_path(path):
    path = Path(path).absolute()
    # Check the whole lexical path before the first filesystem operation.
    if any(bad in part.lower() for part in path.parts for bad in ("hammer", "vlsi")):
        raise ValueError("Forbidden path component")
    if ".." in path.parts:
        raise ValueError("Parent traversal is not an admitted physical input path")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError("Symlink components are not admitted for capture inputs or outputs")
    return path


def file_identity(path, allowed_path):
    path = allowed_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def load_images(path, np):
    identity = file_identity(path, safe_path)
    with np.load(identity["path"], allow_pickle=False) as data:
        if "images" not in data.files:
            raise ValueError("Input NPZ must contain an images array")
        images = data["images"]
    if images.dtype != np.float32:
        raise ValueError("Input images must be float32; preprocessing and dtype conversion must be explicit")
    if images.ndim == 4:
        images = images[:, None, :, :, :]
    if images.ndim != 5 or images.shape[0] < 1 or images.shape[1:] != (1, 3, 224, 224):
        raise ValueError("Expected images shaped [N,3,224,224] or [N,1,3,224,224], with N positive")
    if not np.isfinite(images).all():
        raise ValueError("Input images contain nonfinite values")
    return np.ascontiguousarray(images), identity


def load_checkpoint(model, path, torch):
    identity = file_identity(path, safe_path)
    with Path(identity["path"]).open("rb") as stream:
        state = torch.load(stream, map_location="cpu", weights_only=True)
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Checkpoint must be a plain state_dict with exactly the model's parameter and buffer keys")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError(f"Checkpoint entry {name} must be a dense tensor")
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"Checkpoint shape or dtype mismatch for {name}; implicit conversion is not permitted")
        if not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint entry {name} contains nonfinite values")
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    if any(not torch.equal(loaded[name], value) for name, value in state.items()):
        raise AssertionError("Loaded model does not preserve the supplied state_dict")
    if file_identity(identity["path"], safe_path) != identity:
        raise ValueError("Checkpoint file changed while loading")
    return identity
