"""Select checked host layouts from immutable original software permissions.

The operator supplies the software file, never a candidate permission override.
Prepared source is bound at the existing logical-host hook. Missing permission
or insufficient structural savings retains exact input bytes. Invalid inputs,
stale selections and conformance failures remain errors, not fallback reasons.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .host_optimization_request import HostOptimizationContext, context_from_software

SCHEMA = "merlin.host_graph_policy.v1"
_BUDGET = 1 << 20


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _read(path):
    with path.open("rb") as stream:
        payload = stream.read(_BUDGET + 1)
    if len(payload) > _BUDGET:
        raise ValueError("host software specification exceeds its byte budget")
    return payload


@dataclass(frozen=True)
class HostGraphPolicy:
    software_path: Path
    software_bytes: bytes
    target: str
    entry: str
    enabled: bool = True
    minimum_transpose_bytes_saved: int = 1
    maximum_added_operations: int = 8192
    physical_layout: bool = True
    maximum_layout_assignments: int = 100_000
    maximum_layout_permutations: int = 16
    connected_layout: bool = False
    context: HostOptimizationContext = field(init=False)
    sources: tuple[tuple[str, str], ...] = field(init=False)

    def __post_init__(self):
        from .graph_optimizations import _implementation_sources

        if not isinstance(self.software_path, Path) or type(self.software_bytes) is not bytes:
            raise ValueError("host graph policy requires original software bytes and path")
        if len(self.software_bytes) > _BUDGET or any(type(v) is not bool for v in (
                self.enabled, self.physical_layout, self.connected_layout)):
            raise ValueError("host graph policy requires bounded bytes and an explicit boolean choice")
        for value in (
            self.minimum_transpose_bytes_saved,
            self.maximum_added_operations,
            self.maximum_layout_assignments,
            self.maximum_layout_permutations,
        ):
            if type(value) is not int or not 1 <= value <= _BUDGET:
                raise ValueError("host graph policy requires bounded positive structural budgets")
        context = context_from_software(
            yaml.safe_load(self.software_bytes),
            semantic_sha256=_sha(self.software_bytes),
            target=self.target,
            entry=self.entry,
            allow_missing=True,
        )
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "sources", tuple(sorted(_implementation_sources().items())))

    def revalidate(self):
        from .graph_optimizations import _implementation_sources

        if _read(self.software_path) != self.software_bytes:
            raise ValueError("original host software specification changed; reselect and replan")
        if tuple(sorted(_implementation_sources().items())) != self.sources:
            raise ValueError("host graph policy implementation changed; reselect and replan")

    def record(self):
        return {
            "schema": SCHEMA,
            "software_spec": {
                "path": str(self.software_path),
                "sha256": _sha(self.software_bytes),
            },
            "target": self.target,
            "context": asdict(self.context),
            "enabled": self.enabled,
            "minimum_transpose_bytes_saved": self.minimum_transpose_bytes_saved,
            "maximum_added_operations": self.maximum_added_operations,
            "physical_layout": self.physical_layout,
            "maximum_layout_assignments": self.maximum_layout_assignments,
            "maximum_layout_permutations": self.maximum_layout_permutations,
            "connected_layout": self.connected_layout,
        }

    def retain(self, directory):
        self.revalidate()
        (Path(directory) / "host-graph-software.yaml").write_bytes(self.software_bytes)
        (Path(directory) / "host-graph-policy.json").write_text(json.dumps(self.record(), sort_keys=True))

    def optimize(self, source):
        from .graph_optimizations import GraphOptimizationContract, GraphOptimizationUnavailable, optimize_host_graph

        self.revalidate()
        allowed = (
            "logical_host" in self.context.permitted_stages
            and "propagate_layout" in self.context.permitted_optimizations
            and self.context.independent_element_order_unobserved
        )
        contract = GraphOptimizationContract(
            _sha(source.encode()),
            self.entry,
            propagate_layout=self.enabled and allowed,
            independent_element_order_unobserved=self.context.independent_element_order_unobserved,
            floating_environment_unobserved_nontrapping=self.context.floating_environment_unobserved_nontrapping,
            minimum_transpose_bytes_saved=self.minimum_transpose_bytes_saved,
            maximum_added_operations=self.maximum_added_operations,
            physical_layout=self.physical_layout,
            maximum_layout_assignments=self.maximum_layout_assignments,
            maximum_layout_permutations=self.maximum_layout_permutations,
            connected_layout=self.connected_layout,
        )
        reason = None if contract.propagate_layout else "disabled" if not self.enabled else "permission_not_admitted"
        try:
            selected, report = optimize_host_graph(source, contract)
        except GraphOptimizationUnavailable as exc:
            from dataclasses import replace

            reason = str(exc)
            selected, report = optimize_host_graph(source, replace(contract, propagate_layout=False))
        if reason is not None:
            selected = source
            report["transformed_sha256"] = _sha(source.encode())
        report["automatic_selection"] = {
            "policy": self.record(),
            "status": "applied" if reason is None else "unchanged",
            "reason": reason,
        }
        self.revalidate()
        return selected, report

    def recheck(self, directory, paths):
        self.revalidate()
        directory = Path(directory)
        paths = {
            **paths,
            "software_spec": directory / "host-graph-software.yaml",
            "policy": directory / "host-graph-policy.json",
        }
        binding = directory / "host-graph-source-binding.json"
        if binding.exists() or binding.is_symlink():
            paths["source_binding"] = binding
        if any(path.is_symlink() or not path.is_file() for path in paths.values()):
            raise ValueError("host graph policy artifacts are absent or indirect")
        if paths["software_spec"].read_bytes() != self.software_bytes or json.loads(
            paths["policy"].read_text()
        ) != json.loads(json.dumps(self.record())):
            raise ValueError("host graph original software or policy artifacts changed")
        if "source_binding" in paths:
            _require_projected_source(json.loads(binding.read_text()), paths["input"].read_text())
        selected, expected = self.optimize(paths["input"].read_text())
        if paths["selected"].read_text() != selected or json.loads(paths["report"].read_text()) != json.loads(
            json.dumps(expected)
        ):
            raise ValueError("host graph policy, prepared source or realized selection changed")
        return {role: {"path": str(path.resolve()), "sha256": _sha(path.read_bytes())} for role, path in paths.items()}


def read_host_graph_policy(software_path, *, target, entry, **choices):
    """Read explicitly selected original semantics; absent permissions stay false."""
    path = Path(software_path).resolve(strict=True)
    return HostGraphPolicy(path, _read(path), target, entry, **choices)


def _require_projected_source(binding, source):
    """Reopen the exact ordinary projection product; saved metadata grants nothing."""
    if (type(binding) is not dict or set(binding) != {"schema", "stage", "source"}
            or binding["schema"] != "merlin.host_layout_source_binding.v1"
            or binding["stage"] != "after_source_projection"
            or type(binding["source"]) is not dict or set(binding["source"]) != {"path", "sha256"}):
        raise ValueError("host layout source binding is malformed")
    pin = binding["source"]
    path = Path(pin["path"])
    if (not path.is_absolute() or any(part.is_symlink() for part in (path, *path.parents))
            or not path.is_file()):
        raise ValueError("host layout needs its original direct projected source")
    raw = path.read_bytes()
    # Ordinary projection writers already emit this exact generic IR, with
    # an optional final newline. Compare their complete bytes at every use;
    # reparsing a large unchanged model adds no correspondence information.
    canonical = source.encode()
    if _sha(raw) != pin["sha256"] or raw not in (canonical, canonical+b"\n"):
        raise ValueError("host layout projected source or exact input correspondence changed")


def prepare_group_layout(module, policy, directory, *, projected_source=None):
    """Apply the existing checked policy before source groups become device calls.

    Unknown operations remain coordinate/effect boundaries under the existing
    conformance checker. All original function signatures and observations stay
    intact; source group, object and numerical owners must still rejoin this IR.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    if type(policy) is not HostGraphPolicy:
        raise TypeError("pre-group layout requires original software-owned HostGraphPolicy")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    policy.retain(directory)
    source = text(module, generic=True)
    if projected_source is not None:
        path = Path(projected_source).absolute()
        binding = {"schema": "merlin.host_layout_source_binding.v1", "stage": "after_source_projection",
                   "source": {"path": str(path), "sha256": _sha(path.read_bytes())}}
        _require_projected_source(binding, source)
        (directory / "host-graph-source-binding.json").write_text(json.dumps(binding, sort_keys=True))
    selected, report = policy.optimize(source)
    (directory / "host-graph-input.mlir").write_text(source)
    (directory / "host-graph-optimized.mlir").write_text(selected)
    (directory / "host-graph-optimization.json").write_text(json.dumps(report, sort_keys=True))
    result = module if selected == source else parse_mlir_text(selected)
    result.verify()
    return result


def bind_group_layout_artifacts(device, work, compilation):
    """Retain the actual early policy consumption in the ordinary object recipe."""
    policy = getattr(device, "pre_group_layout", None)
    if policy is None:
        return
    from .graph_optimizations import recheck_host_graph_artifacts

    directory = Path(work) / "pre_group_layout"
    artifacts = recheck_host_graph_artifacts(directory, policy)
    if getattr(device, "sibling_source", None) is not None and "source_binding" not in artifacts:
        raise ValueError("composed host layout lost its original projected source binding")
    for role, pin in artifacts.items():
        compilation.bind_preparation("pre_group_layout_" + role, Path(pin["path"]))
