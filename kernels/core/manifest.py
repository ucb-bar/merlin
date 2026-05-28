"""Kernel manifest schema + loader.

A manifest is JSON; the schema is intentionally narrow because everything that
makes a particular kernel deployable lives in `kernels/match/<name>.match.mlir`
(the linalg-DAG pattern) and `kernels/src/<name>.<lang>` (the source). The
manifest just glues them together with declared metadata: name, signature,
target list, and pre-compute hints from KernelBlaster.

Schema v1 (single source-of-truth — JSON Schema not used to keep deps minimal):

    {
      "schema_version": 1,
      "kernels": [
        {
          "name":           "<unique kernel id>",
          "source":         "src/<file>",                  # relative to manifest dir
          "source_lang":    "cl" | "glsl" | "spirv" | "c" | "cpp" | "ll" | "qnn-context-binary",
          "entry_symbol":   "<C/CL/GLSL function name>",
          "signature": {
            "operands": [
              {"role": "in"|"out", "tensor": "<MLIR tensor type>"},
              ...
            ]
          },
          "match": {
            "kind":      "linalg_dag" | "named_sequence",
            "spec_path": "match/<name>.match.mlir"        # relative to manifest dir
          },
          "targets":        ["<hal target key>", ...],     # eg "qualcomm-adreno-vulkan"
          "workgroup_size": [WGX, WGY, WGZ],               # optional, required for SPIR-V
          "kernelblaster":  { "perf_us": 12.4, ... }        # optional provenance
        }
      ]
    }

Manifest paths are resolved relative to the manifest file's directory so the
whole `kernels/` subtree is relocatable.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any


@dataclasses.dataclass(frozen=True)
class Operand:
    role: str
    tensor: str  # MLIR type literal, e.g. "tensor<512x512xf16>"


@dataclasses.dataclass(frozen=True)
class ConstantSource:
    """How to derive a scalar push-constant value at dispatch time. The only
    form supported today is `dim(input_index, dim_index)` — i.e., extract a
    runtime dim from one of the matched input tensors. Extend by adding
    sibling constructors when other forms (literal, attribute, computed)
    are needed.
    """

    input_index: int
    dim_index: int


@dataclasses.dataclass(frozen=True)
class Constant:
    name: str  # short MLIR-friendly identifier (e.g. "M", "K", "N")
    type: str  # "i32" today (only int constants supported)
    source: ConstantSource
    # Other (input_index, dim_index) pairs that the matched DAG guarantees
    # equal this constant. For matmul, K = input[0].dim(1) = input[1].dim(0)
    # — the constant is sourced from the first, and the second is an alias.
    aliases: tuple[ConstantSource, ...] = ()


@dataclasses.dataclass(frozen=True)
class Signature:
    operands: tuple[Operand, ...]
    constants: tuple[Constant, ...] = ()
    # Optional list of constant names mapping output tensor dims, in order.
    # Lets the spec generator emit the correct shape annotation for the
    # output `flow.dispatch` and `hal.interface.binding.subspan` when the
    # output's dynamic dims aren't trivially `input[0].dim(k)`. For matmul:
    # `output_dims: ["M", "N"]`. For bias-add `(C,H,W) + (C) -> (C,H,W)`:
    # `output_dims: ["C", "H", "W"]`.
    output_dims: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class MatchSpec:
    kind: str
    # For "linalg_dag" / "named_sequence": path to the body MLIR file.
    # For "named_op": optional (no body needed); always None.
    spec_path: pathlib.Path | None
    # For "named_op": the linalg op name to match, e.g. "linalg.matmul",
    # "linalg.conv_2d_nchw_fchw", "linalg.pooling_nchw_max". Ignored for
    # other match kinds.
    op_name: str | None = None
    # For "named_op": when set, treat the input operand at this index as the
    # `outs` of the matched op (not synthesized via fill). Use this for ops
    # like conv where the framework feeds a broadcasted bias into `outs`
    # rather than a zero-fill. -1 means "synthesize outs via fill" (default).
    outs_from_input: int = -1
    # For "named_op": optional attribute string spliced into the matched op
    # verbatim, e.g. `{dilations = dense<1> : vector<2xi64>, strides = dense<2>
    # : vector<2xi64>}`. Required for named ops that carry structural attrs
    # (linalg.conv_2d_*, linalg.pooling_*); cast_compatible_dag matches attrs
    # by exact equality, so the synthesized match must declare them.
    op_attrs: str = ""


@dataclasses.dataclass(frozen=True)
class KernelEntry:
    name: str
    source: pathlib.Path  # absolute, resolved
    source_lang: str
    entry_symbol: str
    signature: Signature
    match: MatchSpec
    targets: tuple[str, ...]
    workgroup_size: tuple[int, int, int] | None
    extra: dict[str, Any]  # opaque (kernelblaster provenance, etc.)


@dataclasses.dataclass(frozen=True)
class Manifest:
    path: pathlib.Path  # absolute path to manifest.json
    schema_version: int
    kernels: tuple[KernelEntry, ...]
    # Optional opt-in selection. When non-empty, only kernels whose `name`
    # appears in this tuple are wired into the auto-generated transform spec
    # for a given compile — the rest stay in the catalog but inert. Empty
    # tuple = "select all" (backward-compatible default).
    select: tuple[str, ...] = ()

    @property
    def root(self) -> pathlib.Path:
        return self.path.parent

    def selected_kernels(self) -> tuple[KernelEntry, ...]:
        """Kernels actually enabled for compilation. Honors `select` when
        present; otherwise returns all kernels."""
        if not self.select:
            return self.kernels
        keep = set(self.select)
        return tuple(k for k in self.kernels if k.name in keep)


_VALID_SOURCE_LANGS = {"cl", "glsl", "spirv", "c", "cpp", "ll", "qnn-context-binary"}
_VALID_MATCH_KINDS = {"linalg_dag", "named_sequence", "named_op"}


def _required_str(obj: dict[str, Any], key: str, ctx: str) -> str:
    if key not in obj or not isinstance(obj[key], str):
        raise ValueError(f"{ctx}: missing required string field '{key}'")
    return obj[key]


def _parse_kernel(raw: dict[str, Any], root: pathlib.Path, idx: int) -> KernelEntry:
    ctx = f"kernels[{idx}]"
    name = _required_str(raw, "name", ctx)
    source_rel = _required_str(raw, "source", ctx)
    source_lang = _required_str(raw, "source_lang", ctx)
    if source_lang not in _VALID_SOURCE_LANGS:
        raise ValueError(
            f"{ctx} '{name}': source_lang must be one of " f"{sorted(_VALID_SOURCE_LANGS)}, got '{source_lang}'"
        )
    entry_symbol = _required_str(raw, "entry_symbol", ctx)
    targets_raw = raw.get("targets")
    if not isinstance(targets_raw, list) or not all(isinstance(t, str) for t in targets_raw):
        raise ValueError(f"{ctx} '{name}': 'targets' must be a list of strings")

    sig_raw = raw.get("signature")
    if not isinstance(sig_raw, dict):
        raise ValueError(f"{ctx} '{name}': missing 'signature' object")
    operands_raw = sig_raw.get("operands")
    if not isinstance(operands_raw, list):
        raise ValueError(f"{ctx} '{name}': signature.operands must be a list")
    operands: list[Operand] = []
    for j, op in enumerate(operands_raw):
        if not isinstance(op, dict):
            raise ValueError(f"{ctx} '{name}': signature.operands[{j}] must be an object")
        role = _required_str(op, "role", f"{ctx} '{name}' operand[{j}]")
        if role not in {"in", "out"}:
            raise ValueError(f"{ctx} '{name}' operand[{j}]: role must be 'in' or 'out', got '{role}'")
        tensor = _required_str(op, "tensor", f"{ctx} '{name}' operand[{j}]")
        operands.append(Operand(role=role, tensor=tensor))

    constants_raw = sig_raw.get("constants", [])
    if not isinstance(constants_raw, list):
        raise ValueError(f"{ctx} '{name}': signature.constants must be a list")
    n_inputs = sum(1 for op in operands if op.role == "in")
    constants: list[Constant] = []
    for j, c in enumerate(constants_raw):
        if not isinstance(c, dict):
            raise ValueError(f"{ctx} '{name}': signature.constants[{j}] must be an object")
        c_name = _required_str(c, "name", f"{ctx} '{name}' constant[{j}]")
        c_type = _required_str(c, "type", f"{ctx} '{name}' constant[{j}]")
        if c_type != "i32":
            raise ValueError(
                f"{ctx} '{name}' constant '{c_name}': only type 'i32' is " f"supported today, got '{c_type}'"
            )
        from_raw = c.get("from")
        if not isinstance(from_raw, dict):
            raise ValueError(
                f"{ctx} '{name}' constant '{c_name}': missing 'from' object " f'(use {{"input": N, "dim": M}})'
            )
        in_idx = from_raw.get("input")
        dim_idx = from_raw.get("dim")
        if not isinstance(in_idx, int) or not isinstance(dim_idx, int):
            raise ValueError(f"{ctx} '{name}' constant '{c_name}': 'from' must be " f'{{"input": int, "dim": int}}')
        if in_idx < 0 or in_idx >= n_inputs:
            raise ValueError(
                f"{ctx} '{name}' constant '{c_name}': input index {in_idx} "
                f"out of range (kernel has {n_inputs} inputs)"
            )
        aliases_raw = c.get("aliases", [])
        if not isinstance(aliases_raw, list):
            raise ValueError(f"{ctx} '{name}' constant '{c_name}': 'aliases' must be a list")
        aliases: list[ConstantSource] = []
        for k, alias_raw in enumerate(aliases_raw):
            if not isinstance(alias_raw, dict):
                raise ValueError(
                    f"{ctx} '{name}' constant '{c_name}' alias[{k}]: " f'must be {{"input": int, "dim": int}}'
                )
            a_in = alias_raw.get("input")
            a_dim = alias_raw.get("dim")
            if not isinstance(a_in, int) or not isinstance(a_dim, int):
                raise ValueError(f"{ctx} '{name}' constant '{c_name}' alias[{k}]: " f"'input' and 'dim' must be ints")
            aliases.append(ConstantSource(input_index=a_in, dim_index=a_dim))
        constants.append(
            Constant(
                name=c_name,
                type=c_type,
                source=ConstantSource(input_index=in_idx, dim_index=dim_idx),
                aliases=tuple(aliases),
            )
        )
    output_dims_raw = sig_raw.get("output_dims", [])
    if not isinstance(output_dims_raw, list) or not all(isinstance(x, str) for x in output_dims_raw):
        raise ValueError(f"{ctx} '{name}': signature.output_dims must be a list of strings")
    declared_const_names = {c.name for c in constants}
    for nm in output_dims_raw:
        if nm not in declared_const_names:
            raise ValueError(f"{ctx} '{name}': output_dims references undeclared " f"constant '{nm}'")
    signature = Signature(
        operands=tuple(operands),
        constants=tuple(constants),
        output_dims=tuple(output_dims_raw),
    )

    match_raw = raw.get("match")
    if not isinstance(match_raw, dict):
        raise ValueError(f"{ctx} '{name}': missing 'match' object")
    match_kind = _required_str(match_raw, "kind", f"{ctx} '{name}' match")
    if match_kind not in _VALID_MATCH_KINDS:
        raise ValueError(
            f"{ctx} '{name}' match: kind must be one of " f"{sorted(_VALID_MATCH_KINDS)}, got '{match_kind}'"
        )
    outs_from_input = -1
    op_attrs = ""
    if match_kind == "named_op":
        op_name = _required_str(match_raw, "op_name", f"{ctx} '{name}' match")
        match_spec_path = None
        ofi = match_raw.get("outs_from_input")
        if ofi is not None:
            if not isinstance(ofi, int):
                raise ValueError(f"{ctx} '{name}' match.outs_from_input must be an int")
            outs_from_input = ofi
        op_attrs_raw = match_raw.get("op_attrs", "")
        if not isinstance(op_attrs_raw, str):
            raise ValueError(f"{ctx} '{name}' match.op_attrs must be a string")
        op_attrs = op_attrs_raw
    else:
        op_name = None
        match_spec_rel = _required_str(match_raw, "spec_path", f"{ctx} '{name}' match")
        match_spec_path = (root / match_spec_rel).resolve()
        if not match_spec_path.exists():
            raise FileNotFoundError(f"{ctx} '{name}' match.spec_path does not exist: {match_spec_path}")

    source_path = (root / source_rel).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"{ctx} '{name}' source does not exist: {source_path}")

    workgroup_raw = raw.get("workgroup_size")
    workgroup: tuple[int, int, int] | None
    if workgroup_raw is None:
        workgroup = None
    else:
        if (
            not isinstance(workgroup_raw, list)
            or len(workgroup_raw) != 3
            or not all(isinstance(x, int) for x in workgroup_raw)
        ):
            raise ValueError(f"{ctx} '{name}': workgroup_size must be a 3-element int list")
        workgroup = (workgroup_raw[0], workgroup_raw[1], workgroup_raw[2])

    extra = {
        k: v
        for k, v in raw.items()
        if k
        not in {
            "name",
            "source",
            "source_lang",
            "entry_symbol",
            "signature",
            "match",
            "targets",
            "workgroup_size",
        }
    }

    return KernelEntry(
        name=name,
        source=source_path,
        source_lang=source_lang,
        entry_symbol=entry_symbol,
        signature=signature,
        match=MatchSpec(
            kind=match_kind,
            spec_path=match_spec_path,
            op_name=op_name,
            outs_from_input=outs_from_input,
            op_attrs=op_attrs,
        ),
        targets=tuple(targets_raw),
        workgroup_size=workgroup,
        extra=extra,
    )


def load(path: str | pathlib.Path) -> Manifest:
    """Load and validate a kernels manifest.

    All path fields are resolved relative to the manifest's directory.
    Raises ValueError on schema problems and FileNotFoundError on missing
    source / match files (validated eagerly so misconfiguration shows up
    early, not after a long iree-compile run).
    """
    path = pathlib.Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"manifest root must be an object: {path}")
    schema_version = raw.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError(f"manifest {path}: unsupported schema_version {schema_version} " f"(only 1 is supported)")
    kernels_raw = raw.get("kernels")
    if not isinstance(kernels_raw, list) or not kernels_raw:
        raise ValueError(f"manifest {path}: 'kernels' must be a non-empty list")
    root = path.parent
    kernels = tuple(_parse_kernel(k, root, i) for i, k in enumerate(kernels_raw))

    # Reject duplicate kernel names — they collide on the executable variant
    # symbol and on the auto-generated transform-spec named sequence.
    seen: set[str] = set()
    for k in kernels:
        if k.name in seen:
            raise ValueError(f"manifest {path}: duplicate kernel name '{k.name}'")
        seen.add(k.name)

    # Optional `select` — list of kernel names enabled for this compile.
    # Validates against the catalog so typos are caught early.
    select_raw = raw.get("select", [])
    if not isinstance(select_raw, list) or not all(isinstance(s, str) for s in select_raw):
        raise ValueError(f"manifest {path}: 'select' must be a list of kernel-name strings")
    for s in select_raw:
        if s not in seen:
            raise ValueError(f"manifest {path}: select references unknown kernel '{s}' " f"(not in catalog)")
    select = tuple(select_raw)

    return Manifest(
        path=path,
        schema_version=schema_version,
        kernels=kernels,
        select=select,
    )
