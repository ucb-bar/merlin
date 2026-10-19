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
    context: HostOptimizationContext = field(init=False)
    sources: tuple[tuple[str, str], ...] = field(init=False)

    def __post_init__(self):
        from .graph_optimizations import _implementation_sources

        if not isinstance(self.software_path, Path) or type(self.software_bytes) is not bytes:
            raise ValueError("host graph policy requires original software bytes and path")
        if len(self.software_bytes) > _BUDGET or any(type(v) is not bool for v in (self.enabled, self.physical_layout)):
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
        if any(path.is_symlink() or not path.is_file() for path in paths.values()):
            raise ValueError("host graph policy artifacts are absent or indirect")
        if paths["software_spec"].read_bytes() != self.software_bytes or json.loads(
            paths["policy"].read_text()
        ) != json.loads(json.dumps(self.record())):
            raise ValueError("host graph original software or policy artifacts changed")
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
