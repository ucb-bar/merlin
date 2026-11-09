"""Exact producer namespace roots for ExecuTorch paper package attribution."""

from __future__ import annotations

from pathlib import Path

from merlin.baselines import executorch as _executorch
from merlin.common.paths import module_source_path, repo_root

et_source_dir = _executorch.et_source_dir

_PRODUCER_MODULES = (
    "merlin.common.paths",
    "merlin.baselines.executorch",
    "merlin.compare.executorch_packages",
    _executorch.k1.__name__,
)


def selected_source_paths(declared: list[Path]) -> list[Path]:
    """Retain declared core closure; add only the actual selected producer namespaces."""
    paths = [path.resolve() for path in declared]
    required_core = (repo_root() / "merlin" / "python" / "merlin").resolve()
    if required_core not in paths:
        raise ValueError(
            "ExecuTorch framework_source_sha256 must cover the complete executed/imported "
            f"source closure; missing roots: {[str(required_core)]}"
        )
    for module in _PRODUCER_MODULES:
        source = module_source_path(module).resolve(strict=True)
        namespace = source.parents[1]
        if not source.is_file() or namespace.name != "merlin" or not namespace.is_dir():
            raise ValueError(f"ExecuTorch producer source has no merlin namespace root: {module}")
        if namespace not in paths:
            paths.append(namespace)
    external = et_source_dir().resolve(strict=True)
    if external not in paths:
        paths.append(external)
    return paths
