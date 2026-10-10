"""Answer-free physical caller layout from the explicitly selected harness authority.

This inspection does not build, execute, grade, or authorize a compiler. The selected
provider owns the layout calculation; the host permits only a strict typed projection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_ROW_FIELDS = frozenset(
    {
        "tensor",
        "dtype",
        "logical_shape",
        "physical_extents",
        "logical_strides_elements",
        "storage_elements",
        "offset_elements",
    }
)
_MAX_DECLARATION_BYTES = 16 * 1024 * 1024


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _ordinary_member(root: Path, member: str) -> Path:
    """The broker may read only an ordinary submission member, never a host path."""
    relative = Path(member)
    if not member or relative.is_absolute() or any(part in ("", ".", "..") for part in member.split("/")):
        raise ValueError("caller layout requires a submission-relative command-buffer member")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("caller layout submission is not an ordinary directory")
    owner = root.resolve()
    path = root / relative
    if not path.resolve().is_relative_to(owner):
        raise ValueError("caller layout command buffer escapes its submission")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("caller layout command buffer traverses a symlink")
    if not path.is_file() or path.stat().st_size > _MAX_DECLARATION_BYTES:
        raise ValueError("caller layout command buffer is absent or too large")
    return path


def _provider_digest(info: Any, module: Any) -> str:
    paths = getattr(module, "caller_layout_source_paths", None)
    if not callable(paths):
        raise ValueError("selected harness provider has no complete caller-layout source declaration")
    root = Path(info.base).resolve()
    declared = tuple(paths())
    if not declared or len(declared) != len(set(declared)):
        raise ValueError("selected caller-layout source roster is empty or duplicated")
    contract = Path(info.contract_path)
    records = []
    from merlin.targetgen.plugins import is_core_module_source

    for path in (*declared, contract, root / "provider.yaml"):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("selected caller-layout source is absent or indirect")
        if path.resolve().is_relative_to(root):
            records.append((path.resolve().relative_to(root).as_posix(), _sha256(path.read_bytes())))
        elif path in declared and is_core_module_source(path):
            # A data-only provider is served by a GENERIC core backend: its layout code is installed
            # core, pinned by bytes beside the provider's own data -- never an arbitrary outside file.
            records.append((f"<core>/{path.name}", _sha256(path.read_bytes())))
        else:
            raise ValueError("selected caller-layout source is outside its provider and the installed core")
    if len(records) != len(set(name for name, _ in records)):
        raise ValueError("selected caller-layout source identity is duplicated")
    return _sha256(_canonical(sorted(records)))


def _core_digest(paths: tuple[Path, ...]) -> str:
    records = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("installed caller-layout source is absent or indirect")
        records.append((path.name, _sha256(path.read_bytes())))
    return _sha256(_canonical(records))


def _checked_projection(layout: Any, command_buffer: dict) -> dict:
    if not isinstance(layout, dict) or set(layout) != {"schema", "policy", "tensors"}:
        raise ValueError("selected harness returned an unrecognized caller-layout projection")
    if layout["schema"] != "caller_storage_layout_v1" or not isinstance(layout["tensors"], list):
        raise ValueError("selected harness returned an unsupported caller-layout schema")
    policy = layout["policy"]
    if not isinstance(policy, dict):
        raise ValueError("selected harness caller-layout policy is malformed")
    params = command_buffer.get("params") or {}
    if not isinstance(params, dict) or (policy.get("mode") == "declared_grouped_axes_storage_v1") != (
        "storage_encodings" in params
    ):
        raise ValueError("selected caller-layout policy differs from the command buffer")
    if policy.get("mode") == "legacy_aligned_row_major_v1":
        if set(policy) != {"mode", "row_alignment_elements"} or type(policy["row_alignment_elements"]) is not int:
            raise ValueError("selected legacy caller-layout policy is incomplete")
        if not 0 < policy["row_alignment_elements"] <= 1_000_000:
            raise ValueError("selected caller row alignment is not bounded and positive")
    elif policy != {"mode": "declared_grouped_axes_storage_v1"}:
        raise ValueError("selected explicit caller-layout policy is unsupported")
    abi = command_buffer.get("kernel_abi") or {}
    tensors = command_buffer.get("tensors") or {}
    args = abi.get("args") or []
    if abi.get("kind") != "whole_program" or not isinstance(args, list) or not isinstance(tensors, dict):
        raise ValueError("caller layout requires a complete whole-program pointer ABI")
    names = [arg.get("tensor") for arg in args if isinstance(arg, dict)]
    if len(names) != len(args) or len(names) != len(set(names)) or set(names) != set(tensors):
        raise ValueError("caller layout pointer roster differs from the command buffer")
    if len(layout["tensors"]) != len(names):
        raise ValueError("selected caller layout omits a pointer")
    checked = []
    for name, row in zip(names, layout["tensors"], strict=True):
        if not isinstance(row, dict) or set(row) != _ROW_FIELDS or row["tensor"] != name:
            raise ValueError("selected caller layout has an extra field or wrong pointer order")
        spec = tensors[name]
        if not isinstance(spec, dict) or row["dtype"] != spec.get("dtype"):
            raise ValueError("selected caller layout dtype differs from the command buffer")
        shape = row["logical_shape"]
        extents = row["physical_extents"]
        strides = row["logical_strides_elements"]
        size, offset = row["storage_elements"], row["offset_elements"]
        if (
            not isinstance(shape, list)
            or not isinstance(extents, list)
            or not isinstance(strides, list)
            or len(shape) != len(strides)
            or any(type(v) is not int or v <= 0 for v in (*shape, *extents))
            or any(type(v) is not int or v < 0 for v in strides)
            or type(size) is not int
            or type(offset) is not int
            or size <= 0
            or size > 256 * 1024 * 1024
            or offset < 0
            or offset + sum((dim - 1) * stride for dim, stride in zip(shape, strides, strict=True)) >= size
        ):
            raise ValueError("selected caller layout has invalid physical bounds")
        if policy["mode"] == "legacy_aligned_row_major_v1":
            if shape != spec.get("shape") or len(extents) != 2 or offset != 0:
                raise ValueError("legacy caller layout differs from the declared logical tensor")
        else:
            encoding = (params.get("storage_encodings") or {}).get(name)
            if not isinstance(encoding, dict):
                raise ValueError("explicit caller layout has no declared encoding")
            from merlin.perf.storage_encoding import GroupedAxesStorage

            selected = GroupedAxesStorage.from_dict(encoding)
            if (
                shape != list(selected.logical_shape)
                or extents != list(selected.physical_shape)
                or strides != list(selected.logical_strides_elements)
                or size != selected.storage_elements
                or offset != selected.offset_elements
            ):
                raise ValueError("explicit caller layout differs from its declared encoding")
        checked.append(row)
    return {"policy": policy, "tensors": checked}


def _declared_core_backend(info: Any, module: Any, plugins: Any) -> bool:
    """Whether ``module`` is the installed core backend the provider's own contract names."""
    block = info.plugin()
    reference = block.get("backend")
    if not isinstance(reference, str) or not plugins.is_core_module_source(module.__file__):
        return False
    root = plugins.provider_root(info.base, block.get("path"))
    return plugins.resolve_reference(root, reference, "module") == Path(module.__file__).resolve()


def inspect_caller_layout(*, submission: Path, command_buffer_member: str, target: str, facts_path: Path) -> dict:
    """Describe only physical pointer storage under one selected, source-pinned provider."""
    from merlin.perf import storage_encoding
    from merlin.runtime import storage_binding
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import plugins

    cb_path = _ordinary_member(Path(submission), command_buffer_member)
    cb_bytes = cb_path.read_bytes()
    cb = json.loads(cb_bytes)
    if not isinstance(cb, dict) or cb.get("target") != target:
        raise ValueError("caller layout command buffer names another target")
    facts_path = Path(facts_path)
    if facts_path.is_symlink() or not facts_path.is_file() or facts_path.stat().st_size > _MAX_DECLARATION_BYTES:
        raise ValueError("selected caller-layout RTL facts are absent, indirect, or too large")
    facts_bytes = facts_path.read_bytes()
    facts = json.loads(facts_bytes)
    if not isinstance(facts, dict) or (facts.get("inputs") or {}).get("target") != target:
        raise ValueError("selected caller-layout RTL facts name another target")
    info = plugins.resolve_support(target)
    module = backends.get_backend(target)
    if not Path(module.__file__).resolve().is_relative_to(Path(info.base).resolve()) and not (
        _declared_core_backend(info, module, plugins)
    ):
        raise ValueError("selected harness backend differs from explicit support provider")
    describe = getattr(module, "describe_caller_layout", None)
    if not callable(describe):
        raise ValueError("selected harness provider does not expose caller-layout inspection")
    core_sources = (
        Path(storage_binding.__file__),
        Path(storage_encoding.__file__),
        Path(plugins.__file__),
        Path(backends.__file__),
        Path(__file__),
    )
    core_before = _core_digest(core_sources)
    provider_before = _provider_digest(info, module)
    projection = _checked_projection(describe(cb, target=target, facts=facts), cb)
    if _provider_digest(info, module) != provider_before or cb_path.read_bytes() != cb_bytes:
        raise ValueError("selected caller-layout source or command buffer changed during inspection")
    if facts_path.read_bytes() != facts_bytes:
        raise ValueError("selected caller-layout facts changed during inspection")
    if _core_digest(core_sources) != core_before:
        raise ValueError("installed caller-layout source changed during inspection")
    return {
        "schema": "public_caller_layout_receipt_v1",
        "status": "layout_only",
        "scope": "caller storage format only; not source placement, numerics, execution or certification",
        "target": target,
        "command_buffer_sha256": _sha256(cb_bytes),
        "rtl_facts_sha256": _sha256(facts_bytes),
        "provider_sha256": provider_before,
        "core_layout_sha256": core_before,
        "policy_sha256": _sha256(_canonical(projection["policy"])),
        **projection,
    }
