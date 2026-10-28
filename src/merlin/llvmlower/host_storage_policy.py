"""Consume original host-storage permissions on the actual compiled LLVM.

This is a late consumer of the existing arena binder, not a storage law inferred
from a candidate or a graph name. Refused allocation sites stay on the heap;
loop-carried state, escaping results and device completion are never guessed.
"""

import hashlib
import json
from pathlib import Path

from .host_graph_policy import HostGraphPolicy

SCHEMA = "merlin.host_intermediate_storage.v1"


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _sources():
    from merlin.xdsl_dialects.lowering import arena_plan

    from . import arena_bind, borrowed_arena, borrowed_buffer_calls

    return {name: _sha(Path(module.__file__).read_bytes()) for name, module in (
        ("arena_bind", arena_bind), ("arena_plan", arena_plan),
        ("borrowed_arena", borrowed_arena), ("borrowed_buffer_calls", borrowed_buffer_calls))} | {
            "host_storage_policy": _sha(Path(__file__).read_bytes())}


def _optimize(source, policy, blocked_reason, borrowed_lifetime):
    if type(policy) is not HostGraphPolicy:
        raise TypeError("intermediate storage needs the original software-owned policy")
    policy.revalidate()
    context = policy.context
    allowed = ("llvm" in context.permitted_stages and "static_arena" in context.permitted_optimizations
               and context.sequential_execution and context.successful_allocations and context.non_reentrant_entry)
    reason = "disabled" if not policy.enabled else "permission_not_admitted" if not allowed else blocked_reason
    selected, arena = source, None
    if reason is None:
        from .arena_bind import bind_arena

        selected, report = bind_arena(source, entry=context.entry, borrowed_lifetime=borrowed_lifetime)
        arena = report.to_dict()
        if selected == source:
            reason = "no_proven_reusable_allocations"
    policy.revalidate()
    return selected, {
        "schema": SCHEMA, "policy": policy.record(), "implementation_sources_sha256": _sources(),
        "input_sha256": _sha(source.encode()), "selected_sha256": _sha(selected.encode()),
        "automatic_selection": {"status": "applied" if selected != source else "unchanged", "reason": reason},
        "blocked_reason": blocked_reason, "arena": arena, "numerics": "not_established", "timing": "UNKNOWN",
    }


def prepare(source_path, policy, directory, compilation, *, blocked_reason=None, borrowed_lifetime=None):
    """Retain distinct original/selected products before the normal model compiler.

    The original caller supplies a concrete incompatible-pipeline reason, if
    applicable. Absence of a permission is a reported unchanged choice. Invalid
    LLVM, stale permissions and binder errors remain errors.
    A live ``borrowed_lifetime`` from the current ordinary lowering permits its
    verified synchronous borrowed calls. Replay must receive that original owner;
    saved report fields never reconstruct this permission.
    """
    source_path, directory = Path(source_path), Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    policy.retain(directory)
    source = source_path.read_text()
    selected, report = _optimize(source, policy, blocked_reason, borrowed_lifetime)
    (directory / "input.ll").write_text(source)
    (directory / "selected.ll").write_text(selected)
    (directory / "report.json").write_text(json.dumps(report, sort_keys=True))
    for role, pin in recheck(directory, policy, borrowed_lifetime=borrowed_lifetime).items():
        compilation.bind_preparation("intermediate_storage_" + role, Path(pin["path"]))
    return directory / "selected.ll" if selected != source else source_path


def recheck(directory, policy, *, borrowed_lifetime=None):
    """Replay allocation custody against original permissions and exact products."""
    directory = Path(directory)
    paths = {role: directory / name for role, name in (
        ("input", "input.ll"), ("selected", "selected.ll"), ("report", "report.json"),
        ("software_spec", "host-graph-software.yaml"), ("policy", "host-graph-policy.json"))}
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("intermediate storage products are absent or indirect")
    policy.revalidate()
    if (paths["software_spec"].read_bytes() != policy.software_bytes
            or json.loads(paths["policy"].read_text()) != json.loads(json.dumps(policy.record()))):
        raise ValueError("intermediate storage original software or policy changed")
    report = json.loads(paths["report"].read_text())
    selected, expected = _optimize(
        paths["input"].read_text(), policy, report["blocked_reason"], borrowed_lifetime
    )
    if selected != paths["selected"].read_text() or report != json.loads(json.dumps(expected)):
        raise ValueError("intermediate storage source, selection or implementation changed")
    return {role: {"path": str(path.resolve()), "sha256": _sha(path.read_bytes())} for role, path in paths.items()}
