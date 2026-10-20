"""Reuse immutable single-forward C artifacts while rebinding typed runtime inputs.

The first call uses the ordinary generator and proves weights/call agreement with
the observed model build. Later calls replace only declarations independently
derived from the original argument manifest, signature and input-order mapping.
No candidate metadata grants authority to change an argument or a weight.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from merlin.model_check.full_build_identity import content_digest_bytes


def _bindings(stage: Path, inputs: Path) -> list[dict]:
    from merlin.llvmlower import c_runtime

    # Older captures without an explicit order retain ordinary regeneration.
    if not (stage / "input_order.json").is_file():
        raise ValueError("an explicit input_order.json is required for runtime template reuse")
    order = {name: int(index) for name, index in json.loads((stage / "input_order.json").read_text()).items()}
    manifest = json.loads((stage / "weights.safetensors.manifest.json").read_text())
    signature = c_runtime.parse_forward_signature(stage / "model.mlir")
    with np.load(inputs) as arrays:
        bindings, positional = [], 0
        for index, (shape, dtype) in enumerate(signature):
            entry = manifest[str(index)]
            name = entry.get("name", "")
            if entry["kind"] in {"param", "buffer"} or name.startswith("c_lifted_tensor_"):
                continue
            if name in order and f"in{order[name]}" in arrays.files:
                key = f"in{order[name]}"
            else:
                key = f"in{positional}"
                positional += 1
            value = np.asarray(arrays[key])
            if list(value.shape) != list(shape):
                raise ValueError(f"runtime template input {key} disagrees with the original signature")
            bindings.append({"argument": index, "key": key, "dtype": dtype, "shape": list(shape)})
    return bindings


def _declaration(binding: dict, arrays) -> str:
    from merlin.llvmlower import c_runtime

    value = np.asarray(arrays[binding["key"]])
    if list(value.shape) != binding["shape"]:
        raise ValueError("bound runtime input shape changed")
    return (f"static {c_runtime.C_OF[binding['dtype']]} merlin_in_{binding['argument']}[] = "
            + "{" + c_runtime._embed_array(np.ascontiguousarray(value), binding["dtype"]) + "};\n")


def _render(template: str, bindings: list[dict], arrays, *, initial: bool) -> str:
    from merlin.llvmlower import c_runtime

    prefixes = {
        f"static {c_runtime.C_OF[row['dtype']]} merlin_in_{row['argument']}[] = {{": row
        for row in bindings
    }
    seen, lines = set(), []
    for line in template.splitlines(keepends=True):
        matches = [prefix for prefix in prefixes if line.startswith(prefix)]
        if not matches:
            lines.append(line)
            continue
        prefix, = matches
        if prefix in seen or not line.endswith("};\n"):
            raise ValueError("runtime template declaration is duplicate or malformed")
        seen.add(prefix)
        replacement = _declaration(prefixes[prefix], arrays)
        if initial and replacement != line:
            raise ValueError("runtime template input does not match the ordinary C generator")
        lines.append(replacement)
    if seen != set(prefixes):
        raise ValueError("runtime template has no unique declaration for every bound input")
    return "".join(lines)


def generate(record: dict, stage: Path, shadow: Path, cgen: Path) -> None:
    """Generate current I/O; render large immutable weights only once per model build."""
    import fcntl

    from merlin.llvmlower import c_runtime
    from merlin.model_check.full_build import FullBuildError, sha256

    build_dir = Path(record["build_dir"])
    template = build_dir / "model_check_single_forward"
    stamp = {"key": content_digest_bytes(record["key"]), "recipe": record["compilation_recipe"]}
    with (build_dir / ".model_check_inputs.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = template / "template.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        if manifest is not None and manifest.get("stamp") != stamp:
            raise FullBuildError("single-forward runtime template belongs to another model build")
        if manifest is not None and manifest.get("bindings") != _bindings(stage, shadow / "inputs.npz"):
            raise FullBuildError("single-forward runtime template changed original input ownership")
        if manifest is None:
            shutil.rmtree(template, ignore_errors=True)
            template.mkdir()
            c_runtime.generate(shadow, template, shadow / "inputs.npz", prepared_dir=build_dir, max_session_steps=1)
            weights = template / "weights.bin"
            if (weights.stat().st_size != record["weights_bin_bytes"]
                    or sha256(weights) != record["weights_bin_sha256"]):
                raise FullBuildError("the regenerated weights blob differs from the build's; refusing to relink")
            weights.unlink()
            if (template / "model_call.c").read_bytes() != (build_dir / "cgen/model_call.c").read_bytes():
                raise FullBuildError("the regenerated argument-table call differs from the build's")
            try:
                bindings = _bindings(stage, shadow / "inputs.npz")
            except ValueError:
                # An old or unusual capture gets the complete ordinary path on
                # every step; no guessed argument ownership enters the fast path.
                for item in template.iterdir():
                    shutil.copy2(item, cgen / item.name)
                shutil.rmtree(template)
                return
            io = (template / "model_io.h").read_text()
            with np.load(shadow / "inputs.npz") as arrays:
                _render(io, bindings, arrays, initial=True)
            manifest = {"stamp": stamp, "bindings": bindings, "files": {
                item.name: sha256(item) for item in sorted(template.iterdir()) if item.is_file()
            }}
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        for name, digest in manifest["files"].items():
            item = template / name
            if Path(name).name != name or not item.is_file() or sha256(item) != digest:
                raise FullBuildError("single-forward runtime template bytes changed")
            if name != "model_io.h":
                shutil.copy2(item, cgen / name)
        with np.load(shadow / "inputs.npz") as arrays:
            io = _render((template / "model_io.h").read_text(), manifest["bindings"], arrays, initial=False)
        (cgen / "model_io.h").write_text(io)
