"""Resolve selected finite lazy Python exports without importing or executing source.

Only literal export tables and bounded literal comprehensions are evaluated. The getter must bind
the requested symbol through that table and pass the selected module to ``import_module``. This
accounts implementation bytes; it is not a capability grant and does not override answer masks.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Mapping

from merlin.common import digest as _mdigest
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

# ---- static_imports ----------------------------------------------------------------------------


@dataclass(frozen=True)
class LazyExportResolution:
    status: str
    module: str | None = None
    reason: str = ""


class _Unsupported(ValueError):
    pass


def _literal(node: ast.AST, variables: dict[str, Any], budget: list[int]) -> Any:
    budget[0] -= 1
    if budget[0] < 0:
        raise _Unsupported("lazy export table exceeds bounded static evaluation")
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, bool, type(None))):
        return node.value
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_literal(item, variables, budget) for item in node.elts)
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise _Unsupported("unpacked lazy export tables are not statically resolved")
        return {
            _literal(key, variables, budget): _literal(value, variables, budget)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.DictComp):
        result = {}

        def bind(target, value, scope):
            if isinstance(target, ast.Name):
                scope[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)):
                if len(target.elts) != len(value):
                    raise _Unsupported("lazy export comprehension unpacking does not match")
                for item, element in zip(target.elts, value):
                    bind(item, element, scope)
            else:
                raise _Unsupported("unsupported lazy export comprehension target")

        def expand(index, scope):
            if index == len(node.generators):
                result[_literal(node.key, scope, budget)] = _literal(node.value, scope, budget)
                return
            generator = node.generators[index]
            if generator.is_async or generator.ifs:
                raise _Unsupported("conditional or async lazy exports are unresolved")
            values = _literal(generator.iter, scope, budget)
            if not isinstance(values, (tuple, list)):
                raise _Unsupported("lazy export generator is not a literal sequence")
            for value in values:
                child = dict(scope)
                bind(generator.target, value, child)
                expand(index + 1, child)

        expand(0, variables)
        return result
    if isinstance(node, ast.JoinedStr):
        chunks = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                chunks.append(part.value)
            elif isinstance(part, ast.FormattedValue) and part.conversion == -1 and part.format_spec is None:
                value = _literal(part.value, variables, budget)
                if not isinstance(value, str):
                    raise _Unsupported("lazy import module interpolation is not a string")
                chunks.append(value)
            else:
                raise _Unsupported("unsupported lazy import module interpolation")
        return "".join(chunks)
    raise _Unsupported(f"unsupported static lazy export expression: {type(node).__name__}")


def resolve_lazy_export(source: bytes, *, package: str, symbol: str) -> LazyExportResolution:
    """Resolve one requested public export, never every sibling export of a lazy package."""
    if len(source) > 131072:
        return LazyExportResolution("unresolved", reason="package initializer exceeds static parsing policy")
    tree = ast.parse(source)
    # An eagerly bound attribute never invokes the package's __getattr__.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            return LazyExportResolution("not_lazy")
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (alias.asname or alias.name.split(".")[0]) == symbol for alias in node.names
        ):
            return LazyExportResolution("not_lazy")
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
            isinstance(target, ast.Name) and target.id == symbol
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        ):
            return LazyExportResolution("not_lazy")
    getters = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"]
    if not getters:
        return LazyExportResolution("not_lazy")
    if len(getters) != 1 or len(getters[0].args.args) != 1:
        return LazyExportResolution("unresolved", reason="lazy getter signature is not statically supported")
    getter = getters[0]
    requested = getter.args.args[0].arg
    table_name = module_variable = None
    for node in ast.walk(getter):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "get"
            and isinstance(node.value.func.value, ast.Name)
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == requested
        ):
            table_name, module_variable = node.value.func.value.id, node.targets[0].id
    if table_name is None:
        return LazyExportResolution("unresolved", reason="lazy getter has no recognized finite export table")
    assignments = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == table_name
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    if len(assignments) != 1:
        return LazyExportResolution("unresolved", reason="lazy export table is absent or reassigned")
    try:
        table = _literal(assignments[0].value, {}, [20000])
        if not isinstance(table, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in table.items()
        ):
            raise _Unsupported("lazy export table must map symbol names to module names")
        if symbol == "*":
            raise _Unsupported("wildcard lazy exports require explicit selected-symbol imports")
        if symbol not in table:
            return LazyExportResolution("not_exported")
        module_calls = []
        for node in ast.walk(getter):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if name != "import_module" or not any(
                isinstance(item, ast.Name) and item.id == module_variable for item in ast.walk(node.args[0])
            ):
                continue
            module_calls.append(node)
        if len(module_calls) != 1:
            raise _Unsupported("selected lazy module does not reach one supported import_module call")
        call = module_calls[0]
        module = _literal(call.args[0], {module_variable: table[symbol]}, [20000])
        if not isinstance(module, str):
            raise _Unsupported("lazy import module is not a static string")
        if module.startswith("."):
            if len(call.args) != 2 or not isinstance(call.args[1], ast.Name) or call.args[1].id != "__name__":
                raise _Unsupported("relative lazy import does not bind the declaring package")
            dots = len(module) - len(module.lstrip("."))
            components = package.split(".")
            if dots > len(components):
                raise _Unsupported("relative lazy import escapes the package hierarchy")
            module = ".".join((*components[: len(components) - dots + 1], module[dots:]))
        if not module or any(not component.isidentifier() for component in module.split(".")):
            raise _Unsupported("lazy import is not a Python module name")
        return LazyExportResolution("resolved", module=module)
    except _Unsupported as error:
        return LazyExportResolution("unresolved", reason=str(error))


def imported_attribute_paths(tree: ast.AST, bindings: dict[str, set[str]]) -> set[str]:
    """Find selected package attributes through imported aliases and constant getattr calls."""

    def paths(node):
        if isinstance(node, ast.Name):
            return bindings.get(node.id, set())
        if isinstance(node, ast.Attribute):
            return {base + "." + node.attr for base in paths(node.value)}
        return set()

    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            result.update(paths(node))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            result.update(base + "." + node.args[1].value for base in paths(node.args[0]))
    return result


# ---- external_objective ------------------------------------------------------------------------
# Host-pinned normalized model sources, separate from functional qualification.
#
# Only the explicitly supplied spec and MLIR file are read. Provenance pins are
# recorded, never interpreted as paths, callbacks, normalization or capture grants.
# The host declares completeness; parsing cannot prove that a capture is complete.

OBJECTIVE_DIRECTORY = "_external_objective"
OBJECTIVES_DIRECTORY = "_external_objectives"


def objective_directory(objective_id: str) -> Path:
    """Collision-free grant location for a member of a multi-model external portfolio."""
    _require(
        isinstance(objective_id, str)
        and objective_id
        and all(c.isascii() and (c.isalnum() or c in "_-.") for c in objective_id),
        "external objective id must be a simple nonempty identifier",
    )
    return Path(OBJECTIVES_DIRECTORY) / objective_id


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pin(value):
    _require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "objective pins must be lowercase SHA-256",
    )


_sha = _mdigest.sha256_bytes


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _read(path: Path, *, limit: int, digest: str) -> bytes:
    _pin(digest)
    _require(path.is_absolute(), "external objective paths must be explicit absolute paths")
    _require(not any(part == ".." for part in path.parts), "objective paths cannot contain parent traversal")
    # Reject symlink components as well as the leaf; do not follow a surprising
    # alternate source tree before freezing host-granted bytes.
    _require(not any(part.is_symlink() for part in (path, *path.parents)), "external objective path is linked")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode), "external objective requires a regular file")
        _require(0 < info.st_size <= limit, "external objective exceeds host input size bound or is empty")
        payload = stream.read(limit + 1)
    _require(len(payload) <= limit and _sha(payload) == digest, "external objective bytes differ from the pinned hash")
    return payload


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate external objective spec key")
        result[key] = value
    return result


@dataclass(frozen=True)
class ExternalObjective:
    """Validated immutable bytes; contains no authority to execute source code."""

    objective_id: str
    source_sha256: str
    spec_sha256: str
    source_bytes: bytes
    record_bytes: bytes
    descriptor_bytes: bytes

    def files(self) -> tuple[tuple[str, bytes], ...]:
        return (
            ("capsule.interface.mlir", self.source_bytes),
            ("capsule.yaml", self.descriptor_bytes),
            ("objective.json", self.record_bytes),
        )

    def record(self) -> dict:
        return json.loads(self.record_bytes)


def load_external_objective(spec_path: Path, *, spec_sha256: str, max_source_bytes: int) -> ExternalObjective:
    """Read one pinned host declaration and its exact already-normalized source.

    The spec contains ONLY schema/id/entry/source path+hash, nonempty provenance
    pins, and explicit normalized/complete-model assertions. It cannot introduce
    expected outputs, compiler options, scripts or additional readable files.
    """
    _require(type(max_source_bytes) is int and max_source_bytes > 0, "positive host source byte bound required")
    raw = _read(Path(spec_path), limit=1024 * 1024, digest=spec_sha256)
    spec = json.loads(
        raw,
        object_pairs_hook=_unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite objective spec")),
    )
    fields = {
        "schema",
        "id",
        "entry",
        "normalized_source",
        "source_sha256",
        "provenance_pins",
        "already_normalized",
        "complete_model",
    }
    _require(
        isinstance(spec, dict) and set(spec) == fields, "external objective spec fields are incomplete or unsupported"
    )
    _require(spec["schema"] == "external_full_model_objective_spec_v1", "unsupported external objective schema")
    for key in ("id", "entry"):
        value = spec[key]
        _require(
            isinstance(value, str) and value and all(c.isascii() and (c.isalnum() or c in "_-.") for c in value),
            f"external objective {key} must be a simple nonempty identifier",
        )
    _require(
        spec["already_normalized"] is True and spec["complete_model"] is True,
        "host must explicitly declare an already-normalized complete model",
    )
    provenance = spec["provenance_pins"]
    _require(isinstance(provenance, dict) and provenance, "external objective requires explicit provenance pins")
    for label, digest in provenance.items():
        _require(
            isinstance(label, str) and label and all(c.isascii() and (c.isalnum() or c in "_-.") for c in label),
            "provenance labels are identifiers, never paths",
        )
        _pin(digest)
    _require(isinstance(spec["normalized_source"], str), "normalized source must be an explicit path")
    source = _read(Path(spec["normalized_source"]), limit=max_source_bytes, digest=spec["source_sha256"])
    module = parse_mlir_text(source.decode("utf-8"))
    module.verify()
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    symbols = {op.sym_name.data: op for op in functions}
    _require(len(symbols) == len(functions), "duplicate source function symbols")
    entry = symbols.get(spec["entry"])
    _require(
        entry is not None and len(entry.body.blocks) == 1,
        "normalized objective requires its defined single-block entry",
    )
    _require(bool(entry.function_type.outputs), "external model entry must return declared outputs")
    operations = Counter(op.name for op in entry.walk() if op is not entry)
    _require(
        any(name != "func.return" for name in operations), "return-only source does not provide an optimization graph"
    )
    for operation in module.walk():
        if operation.name == "func.call":
            callee = operation.callee.root_reference.data
            _require(
                callee in symbols and bool(symbols[callee].body.blocks),
                "normalized objective has an unresolved external function call",
            )
    record = {
        "schema": "external_full_model_objective_v1",
        "id": spec["id"],
        "entry": spec["entry"],
        "spec_sha256": spec_sha256,
        "source_sha256": spec["source_sha256"],
        "provenance_pins": provenance,
        "source_bytes": len(source),
        "entry_argument_types": [str(arg.type) for arg in entry.body.block.args],
        "entry_output_types": [str(ty) for ty in entry.function_type.outputs],
        "entry_operation_counts": dict(sorted(operations.items())),
        "complete_model": "host_declared_not_independently_proved",
        "already_normalized": "host_declared_no_normalization_executed",
        "phase1_qualification_applies_to_external_model": False,
        "phase1_action": "reuse_unchanged_existing_qualification_and_waivers",
        "numerical_equivalence": "UNPROVEN",
        "full_model_execution_allowed": False,
        "scope": "separate pinned compile/static optimization objective; bounded changed-mechanism probes only",
    }
    descriptor = {
        "id": spec["id"],
        "interface_mlir": "capsule.interface.mlir",
        "entry": spec["entry"],
        "required_oracle_tiers": [],
        "external_objective_record": "objective.json",
        "performance": {"global_objective": True, "qualification": "UNPROVEN_EXTERNAL_MODEL"},
    }
    return ExternalObjective(spec["id"], spec["source_sha256"], spec_sha256, source, _json(record), _json(descriptor))


# ---- host_resources ----------------------------------------------------------------------------
# Target-neutral host memory admission for bounded compiler experiments.
#
# This module prices no target and knows nothing about a workload.  It only turns
# Linux's measured host-memory state into an explicit, caller-selected admission
# decision.  Experiment drivers choose the limits; keeping that policy outside
# the module avoids baking one machine's capacity into reusable compiler code.

_REQUIRED_FIELDS = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")


@dataclass(frozen=True)
class HostMemorySample:
    observed_at_unix_s: float
    memory_total_bytes: int
    memory_available_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int

    @property
    def swap_used_bytes(self) -> int:
        return self.swap_total_bytes - self.swap_free_bytes

    def record(self) -> dict[str, int | float]:
        return {
            "observed_at_unix_s": self.observed_at_unix_s,
            "memory_total_bytes": self.memory_total_bytes,
            "memory_available_bytes": self.memory_available_bytes,
            "swap_total_bytes": self.swap_total_bytes,
            "swap_free_bytes": self.swap_free_bytes,
            "swap_used_bytes": self.swap_used_bytes,
        }


def parse_proc_meminfo(text: str, *, observed_at_unix_s: float | None = None) -> HostMemorySample:
    """Parse the four kernel counters needed by the admission policy.

    ``/proc/meminfo`` reports these values in KiB.  Unknown units and missing or
    internally inconsistent counters are refused rather than guessed.
    """
    fields: dict[str, int] = {}
    for raw in text.splitlines():
        name, separator, remainder = raw.partition(":")
        if not separator or name not in _REQUIRED_FIELDS:
            continue
        parts = remainder.split()
        if len(parts) != 2 or parts[1] != "kB":
            raise ValueError(f"unsupported /proc/meminfo field: {name}")
        try:
            kib = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"non-integer /proc/meminfo field: {name}") from exc
        if kib < 0:
            raise ValueError(f"negative /proc/meminfo field: {name}")
        fields[name] = kib * 1024
    missing = tuple(name for name in _REQUIRED_FIELDS if name not in fields)
    if missing:
        raise ValueError(f"missing /proc/meminfo fields: {', '.join(missing)}")
    if fields["MemAvailable"] > fields["MemTotal"] or fields["SwapFree"] > fields["SwapTotal"]:
        raise ValueError("inconsistent /proc/meminfo counters")
    return HostMemorySample(
        observed_at_unix_s=time() if observed_at_unix_s is None else observed_at_unix_s,
        memory_total_bytes=fields["MemTotal"],
        memory_available_bytes=fields["MemAvailable"],
        swap_total_bytes=fields["SwapTotal"],
        swap_free_bytes=fields["SwapFree"],
    )


def sample_host_memory(path: Path = Path("/proc/meminfo")) -> HostMemorySample:
    return parse_proc_meminfo(path.read_text())


@dataclass(frozen=True)
class HostResourcePolicy:
    minimum_memory_available_bytes: int
    maximum_swap_used_bytes: int
    consecutive_violations_to_stop: int = 2

    def __post_init__(self) -> None:
        values = (
            self.minimum_memory_available_bytes,
            self.maximum_swap_used_bytes,
            self.consecutive_violations_to_stop,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("host resource limits must be integers")
        if min(self.minimum_memory_available_bytes, self.maximum_swap_used_bytes) < 0:
            raise ValueError("host resource byte limits must be nonnegative")
        if self.consecutive_violations_to_stop < 1:
            raise ValueError("consecutive violation limit must be positive")

    def record(self) -> dict[str, int]:
        return {
            "minimum_memory_available_bytes": self.minimum_memory_available_bytes,
            "maximum_swap_used_bytes": self.maximum_swap_used_bytes,
            "consecutive_violations_to_stop": self.consecutive_violations_to_stop,
        }


def violations(sample: HostMemorySample, policy: HostResourcePolicy) -> tuple[str, ...]:
    reasons = []
    if sample.memory_available_bytes < policy.minimum_memory_available_bytes:
        reasons.append("memory_available_below_limit")
    if sample.swap_used_bytes > policy.maximum_swap_used_bytes:
        reasons.append("swap_used_above_limit")
    return tuple(reasons)


class HostResourceTripwire:
    """Require consecutive bad samples before stopping a supervised workload."""

    def __init__(self, policy: HostResourcePolicy):
        self.policy = policy
        self.consecutive_violations = 0

    def observe(self, sample: HostMemorySample) -> Mapping[str, object]:
        reasons = violations(sample, self.policy)
        self.consecutive_violations = self.consecutive_violations + 1 if reasons else 0
        return {
            "status": (
                "stop"
                if reasons and self.consecutive_violations >= self.policy.consecutive_violations_to_stop
                else "continue"
            ),
            "reasons": list(reasons),
            "consecutive_violations": self.consecutive_violations,
            "sample": sample.record(),
            "policy": self.policy.record(),
        }


def summarize_samples(samples: list[HostMemorySample]) -> dict[str, object]:
    if not samples:
        return {"sample_count": 0}
    return {
        "sample_count": len(samples),
        "first": samples[0].record(),
        "last": samples[-1].record(),
        "minimum_memory_available_bytes": min(row.memory_available_bytes for row in samples),
        "maximum_swap_used_bytes": max(row.swap_used_bytes for row in samples),
    }
