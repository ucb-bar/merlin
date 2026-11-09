"""Materialize and verify runtime bundles from frozen model-capsule assets.

This grader-owned adapter preserves the source, loader ABI, every input/output,
immutable weights, and before/after byte custody. It does not recapture models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _model_asset(capsule_root: Path, declared: object, role: str) -> Path:
    """Resolve one declared model-capsule asset without permitting aliases or escapes."""
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"model capsule does not declare its {role} asset")
    rel = Path(declared)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"model capsule {role} path must stay inside the capsule: {declared!r}")
    lexical = capsule_root / rel
    if lexical.is_symlink() or not lexical.is_file():
        raise ValueError(f"model capsule {role} asset is missing or a symlink: {declared!r}")
    resolved = lexical.resolve(strict=True)
    if capsule_root not in resolved.parents:
        raise ValueError(f"model capsule {role} asset escapes its frozen directory: {declared!r}")
    return resolved


def _file_identity(path: Path) -> dict[str, Any]:
    import hashlib

    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            size += len(block)
            h.update(block)
    return {"sha256": h.hexdigest(), "size_bytes": size}


def _content_identity(files: dict[str, Path], *, root: Path | None = None) -> dict[str, Any]:
    """Stable identity of a role-keyed set; paths do not participate in its content hash."""
    import hashlib

    artifacts = {}
    for role, path in sorted(files.items()):
        rel = str(path.relative_to(root)) if root is not None else path.name
        artifacts[role] = {"path": rel, **_file_identity(path)}
    canonical = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
    return {
        "version": 1,
        "content_sha256": hashlib.sha256(canonical).hexdigest(),
        "artifacts": artifacts,
        "missing": [],
    }


def _write_deterministic_npz(path: Path, arrays: dict[str, Any]) -> None:
    """Write an np.load-compatible archive whose bytes do not contain wall-clock timestamps."""
    import io
    import zipfile

    import numpy as np

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in sorted(arrays.items()):
            payload = io.BytesIO()
            np.save(payload, np.ascontiguousarray(array), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o100444 << 16
            archive.writestr(member, payload.getvalue())


def _numpy_dtype(dtype: str):
    import numpy as np

    dtypes = {
        "f64": np.float64,
        "f32": np.float32,
        "f16": np.float16,
        "i64": np.int64,
        "i32": np.int32,
        "i16": np.int16,
        "i8": np.int8,
        "i1": np.int8,
    }
    if dtype not in dtypes:
        # A bf16 array cannot be reconstructed from decoded YAML values without an explicit bit-level
        # representation.  Guessing by casting to f32 would make the runtime consume different bytes.
        raise ValueError(f"capsule runtime-bundle adapter cannot losslessly encode dtype {dtype!r}")
    return dtypes[dtype]


def _model_runtime_bundle(capsule: dict, *, timeout: int, artifact_root: Path | None = None, target: str | None = None):
    """Context manager yielding a read-only capture bundle made only from frozen capsule assets.

    The model capsule intentionally exposes a linalg interface, external weights and an independent
    golden instead of a live model2MLIR capture directory.  Whole-model execution needs the equivalent
    runtime files.  This adapter reconstructs those files deterministically, validates their ABI against
    the capsule's frozen loader via torch.export, and checks every source/bundle digest both before and
    after compilation.  It never resolves ``out/artifacts/recaptures`` and never captures/downloads a
    model.
    """
    import contextlib
    import hashlib
    import shutil
    import subprocess
    import tempfile

    import numpy as np

    from merlin.targetgen.owned_host_slice_execution import runtime_bundle, selected_source_slice

    if selected_source_slice(capsule):
        return runtime_bundle(capsule, target=target, artifact_root=artifact_root)

    @contextlib.contextmanager
    def _materialized():
        raw_root = capsule.get("__dir__")
        if not raw_root:
            raise ValueError("model capsule has no frozen source directory")
        lexical_root = Path(raw_root)
        if lexical_root.is_symlink() or not lexical_root.is_dir():
            raise ValueError("model capsule source directory is missing or a symlink")
        root = lexical_root.resolve(strict=True)
        op_attrs = (capsule.get("operation") or {}).get("attributes") or {}
        ref = capsule.get("pytorch_ref") or {}

        interface_names = [str(x) for x in (capsule.get("linalg_mlir"), capsule.get("interface_mlir")) if x]
        if not interface_names or len(set(interface_names)) != 1:
            raise ValueError("model capsule linalg_mlir/interface_mlir declarations are absent or inconsistent")
        interface = _model_asset(root, interface_names[0], "linalg interface")
        loader = _model_asset(root, ref.get("loader"), "PyTorch loader")
        weights = _model_asset(root, op_attrs.get("weights"), "external weights")
        manifest_name = op_attrs.get("weights_manifest")
        manifest_asset = _model_asset(root, manifest_name, "capture-time weight manifest") if manifest_name else None
        profile_name = op_attrs.get("loader_profile")
        profile_sha = op_attrs.get("loader_profile_sha256")
        loader_profile = None
        if profile_name is not None or profile_sha is not None:
            loader_profile = _model_asset(root, profile_name, "selected loader profile")
            if (
                loader_profile != loader.with_name("profile.json")
                or not isinstance(profile_sha, str)
                or hashlib.sha256(loader_profile.read_bytes()).hexdigest() != profile_sha
            ):
                raise ValueError("selected loader profile differs from its immutable capsule binding")
        elif loader.with_name("profile.json").exists():
            raise ValueError("adjacent loader profile has no immutable capsule binding")
        dependency_name = op_attrs.get("loader_dependencies")
        dependency_root = None
        if dependency_name:
            dependency_lexical = root / str(dependency_name)
            if (
                Path(str(dependency_name)).is_absolute()
                or ".." in Path(str(dependency_name)).parts
                or dependency_lexical.is_symlink()
                or not dependency_lexical.is_dir()
            ):
                raise ValueError("model loader dependency tree must be a real capsule-local directory")
            dependency_root = dependency_lexical.resolve(strict=True)
            if root not in dependency_root.parents:
                raise ValueError("model loader dependency tree must stay inside the capsule")
        golden_yaml = _model_asset(root, "golden.yaml", "independent golden")
        capsule_yaml = _model_asset(root, "capsule.yaml", "capsule declaration")
        source_files = {
            "capsule_declaration": capsule_yaml,
            "loader": loader,
            "linalg_interface": interface,
            "external_weights": weights,
            "independent_golden": golden_yaml,
        }
        if loader_profile is not None:
            source_files["loader_profile"] = loader_profile
        if manifest_asset is not None:
            source_files["capture_manifest"] = manifest_asset
        from merlin.targetgen import golden_store as _GS

        if (root / _GS.ARRAYS).exists() or (root / _GS.ARRAYS).is_symlink():
            source_files["independent_golden_arrays"] = _model_asset(root, _GS.ARRAYS, "independent golden arrays")
        if dependency_root is not None:
            for dependency in sorted(dependency_root.rglob("*.py")):
                if dependency.is_symlink() or not dependency.is_file():
                    raise ValueError("model loader dependency source is missing or a symlink")
                source_files[f"loader_dependency::{dependency.relative_to(dependency_root)}"] = dependency
        source_before = _content_identity(source_files, root=root)

        golden_doc = _GS.load_golden(root)
        if not isinstance(golden_doc, dict):
            raise ValueError("model capsule golden.yaml is not a mapping")
        prov = golden_doc.get("oracle_provenance") or {}
        if prov.get("pytorch_source") != loader.name or prov.get("linalg_mlir") != interface.name:
            raise ValueError("golden provenance does not name the capsule's declared loader/interface")
        input_specs = list(capsule.get("inputs") or [])
        input_names = [str(x.get("name")) for x in input_specs]
        output_names = [str(x) for x in (op_attrs.get("outs") or [])]
        if not output_names:
            output_names = [str(op_attrs.get("out") or "")]
        declared_order = list(op_attrs.get("arg_order") or [])
        golden_order = list(prov.get("arg_order") or [])
        if declared_order != input_names + output_names or golden_order != declared_order:
            raise ValueError("capsule/golden argument order is inconsistent")
        golden_inputs = prov.get("inputs") or {}
        outputs = golden_doc.get("outputs") or {}
        if set(golden_inputs) != set(input_names) or set(outputs) != set(output_names):
            raise ValueError("capsule golden input/output names do not exactly match its declaration")

        from merlin.common.mlir_query import forward_signature
        from merlin.llvmlower.weights_pack import load_safetensors_header

        signature, output_signature = forward_signature(interface)
        header, _payload = load_safetensors_header(weights)
        if len(signature) != len(header) + len(input_names):
            raise ValueError(
                f"interface has {len(signature)} arguments but frozen weights+inputs account for "
                f"{len(header)}+{len(input_names)}"
            )
        if len(output_signature) != len(output_names):
            raise ValueError(
                f"capsule declares {len(output_names)} outputs but interface returns {len(output_signature)}"
            )

        arrays: dict[str, np.ndarray] = {}
        first_input = len(signature) - len(input_names)
        for offset, (decl, name) in enumerate(zip(input_specs, input_names)):
            shape, dtype = signature[first_input + offset]
            source = golden_inputs[name]
            if list(decl.get("shape") or []) != list(shape) or list(source.get("shape") or []) != list(shape):
                raise ValueError(f"frozen input {name!r} shape disagrees with the interface")
            if str(decl.get("dtype") or "") != dtype:
                raise ValueError(
                    f"frozen input {name!r} declares dtype {decl.get('dtype')!r} but the executable "
                    f"interface requires {dtype!r}"
                )
            decoded = source.get("decoded")
            if decoded is None:
                raise ValueError(f"frozen input {name!r} has no decoded row-major values")
            array = np.asarray(decoded, dtype=_numpy_dtype(dtype))
            if array.size != int(np.prod(shape, dtype=np.int64)):
                raise ValueError(f"frozen input {name!r} element count disagrees with its shape")
            arrays[f"in{offset}"] = np.ascontiguousarray(array.reshape(shape))

        output_arrays: dict[str, np.ndarray] = {}
        for output_name, (out_shape, out_dtype) in zip(output_names, output_signature, strict=True):
            output_array = np.asarray(outputs[output_name], dtype=_numpy_dtype(out_dtype))
            if output_array.size != int(np.prod(out_shape, dtype=np.int64)):
                raise ValueError(f"frozen golden {output_name!r} element count disagrees with interface shape")
            output_arrays[output_name] = np.ascontiguousarray(output_array.reshape(out_shape))

        if artifact_root is None:
            temp = Path(tempfile.mkdtemp(prefix="merlin_frozen_capsule_bundle_"))
        else:
            temp = artifact_root
            if not temp.is_absolute() or temp.exists() or temp.is_symlink():
                raise ValueError("retained public runtime bundle requires a fresh private artifact root")
            temp.mkdir(mode=0o700)
        bundle = temp / "bundle"
        bundle.mkdir(mode=0o700)
        try:
            shutil.copyfile(interface, bundle / "model.mlir")
            shutil.copyfile(weights, bundle / "weights.safetensors")
            # The linalg's prov.weights_file is capsule-relative and remains byte-identical.  Keep an
            # alias with that declared name instead of rewriting incompatible/unproven IR.
            shutil.copyfile(weights, bundle / weights.name)
            _write_deterministic_npz(bundle / "inputs.npz", arrays)
            with (bundle / "golden.npy").open("wb") as fh:
                np.save(fh, output_arrays[output_names[0]], allow_pickle=False)
            _write_deterministic_npz(bundle / "goldens.npz", output_arrays)
            (bundle / "output_order.json").write_text(json.dumps(output_names, separators=(",", ":")), encoding="utf-8")
            if manifest_asset is not None:
                shutil.copyfile(manifest_asset, bundle / "weights.safetensors.manifest.json")

            from merlin.targetgen.capsule_source import _m2m_python

            torch_python = _m2m_python()
            worker = Path(__file__).with_name("bundle_worker.py").resolve(strict=True)
            if not torch_python.is_file():
                raise ValueError(f"torch interpreter unavailable for frozen loader validation: {torch_python}")
            request = {
                "loader": str(loader),
                "weights": str(weights),
                "inputs_npz": str(bundle / "inputs.npz"),
                "golden_npy": str(bundle / "golden.npy"),
                "signature": [{"shape": list(shape), "dtype": dtype} for shape, dtype in signature],
                "output_signature": [{"shape": list(shape), "dtype": dtype} for shape, dtype in output_signature],
                "goldens_npz": str(bundle / "goldens.npz"),
                "output_order": output_names,
                "torch_seed": int(op_attrs.get("torch_seed", 0)),
                "numeric_policy": capsule.get("numeric_policy") or {},
            }
            if loader_profile is not None:
                request["loader_profile"] = {"path": str(loader_profile), "sha256": profile_sha}
            if manifest_asset is not None:
                request["captured_manifest"] = str(manifest_asset)
            if dependency_root is not None:
                request["loader_dependencies"] = str(dependency_root)
            request_path = temp / "request.json"
            request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
            env = dict(os.environ)
            # The worker has no hub code path.  These additionally force common model clients offline
            # if a future frozen loader imports one; an attempted checkout must fail, not mutate inputs.
            env.update(
                {
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                    "WANDB_MODE": "offline",
                }
            )
            run = subprocess.run(
                [str(torch_python), "-B", str(worker), "--request", str(request_path)],
                cwd=str(root),
                env=env,
                capture_output=True,
                text=True,
                timeout=max(30, min(int(timeout), 300)),
            )
            marker = "__CAPSULE_BUNDLE_ABI__ "
            line = next((ln[len(marker) :] for ln in run.stdout.splitlines() if ln.startswith(marker)), None)
            if run.returncode != 0 or line is None:
                detail = (run.stderr or run.stdout or "no diagnostic")[-1200:]
                raise ValueError(
                    f"frozen loader/weights/interface/golden validation failed (rc={run.returncode}): {detail}"
                )
            report = json.loads(line)
            if (
                report.get("loader_sha256") != source_before["artifacts"]["loader"]["sha256"]
                or report.get("weights_sha256") != source_before["artifacts"]["external_weights"]["sha256"]
            ):
                raise ValueError("loader or weights changed while the export ABI was being validated")
            manifest = report.get("manifest")
            order = report.get("input_order")
            if not isinstance(manifest, dict) or set(manifest) != {str(i) for i in range(len(signature))}:
                raise ValueError("loader validator returned an incomplete forward-argument manifest")
            if not isinstance(order, dict) or sorted(order.values()) != list(range(len(input_names))):
                raise ValueError("loader validator returned an invalid input order")
            if manifest_asset is None:
                (bundle / "weights.safetensors.manifest.json").write_text(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
                )
            (bundle / "input_order.json").write_text(
                json.dumps(order, sort_keys=True, separators=(",", ":")), encoding="utf-8"
            )

            bundle_files = {p.name: p for p in bundle.iterdir() if p.is_file()}
            bundle_before = _content_identity(bundle_files)
            source_after_materialize = _content_identity(source_files, root=root)
            if source_after_materialize["content_sha256"] != source_before["content_sha256"]:
                raise ValueError("frozen capsule assets changed while constructing the runtime bundle")
            for path in bundle_files.values():
                path.chmod(0o444)
            bundle.chmod(0o555)
            provenance = {
                "version": 2 if manifest_asset is not None else 1,
                "source": source_before,
                "bundle": bundle_before,
                "construction": (
                    "frozen_capsule_assets_v2" if manifest_asset is not None else "frozen_capsule_assets_v1"
                ),
                "interface_reused_byte_exact": True,
                "live_recapture_used": False,
                "network_or_model_checkout_used": False,
                "validation": {
                    k: report.get(k)
                    for k in (
                        "torch_export",
                        "capture_manifest_validated",
                        "golden_validated",
                        "golden_value_replay",
                        "weights_validated_exact",
                        "loader_input_count",
                    )
                },
                "tool": {
                    "worker_sha256": _file_identity(worker)["sha256"],
                    "python": report.get("python"),
                    "python_version": report.get("python_version"),
                    "torch_version": report.get("torch_version"),
                },
            }

            def verify_unchanged() -> None:
                source_after = _content_identity(source_files, root=root)
                bundle_after = _content_identity(bundle_files)
                if source_after["content_sha256"] != source_before["content_sha256"]:
                    raise ValueError("frozen capsule assets changed during whole-model compilation")
                if bundle_after["content_sha256"] != bundle_before["content_sha256"]:
                    raise ValueError("constructed runtime bundle changed during whole-model compilation")

            yield bundle, provenance, verify_unchanged
        finally:
            if artifact_root is None:
                if bundle.exists():
                    bundle.chmod(0o700)
                    for path in bundle.iterdir():
                        if path.is_file() and not path.is_symlink():
                            path.chmod(0o600)
                shutil.rmtree(temp, ignore_errors=True)

    return _materialized()
