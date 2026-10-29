"""Fresh full-module three-buffer projection/query preparation for scoped loops.

Python parsing/projection is trusted. Only a later actual Lean check may establish
its complete operational theorem. This module launches no Lean or SDK operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common.owned_process import run_owned

from .lean_graph_pipeline_gate_profile_prepare import _domain
from .lean_graph_pipeline_gate_projection import project_lean_sources
from .lean_graph_state_pipeline_emit import emit

DOMAIN = "static-nonnegative-index64-no-tail-unroll-ordered1d-segment-v1"
ROOT = "Merlin.GraphStatePipelineInstance.complete"


class Unsupported(ValueError):
    """Explicit domain/actual-producer mismatch; not a Lean refusal."""


class Unavailable(RuntimeError):
    """Host/parser/execution failure; not a successful refusal."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def file_ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": digest(data), "bytes": len(data)}


@dataclass(frozen=True)
class Settings:
    factor: int = 2
    tile: int = 3
    index_bits: int = 64

    def validate(self):
        if (
            type(self.factor) is not int
            or not 2 <= self.factor <= 4
            or type(self.tile) is not int
            or not 2 <= self.tile <= 16
            or type(self.index_bits) is not int
            or self.index_bits != 64
        ):
            raise Unsupported("factor2..4/tile2..16/index64 required")


@dataclass(frozen=True)
class PreparedLoop:
    buffers: tuple[str, str, str]
    settings: Settings
    capture_json: str
    query: str
    roots: tuple[str, ...]
    sources_json: str
    reference: str
    domain: str = DOMAIN

    @property
    def identity(self):
        return digest(canonical(asdict(self)))

    def describe(self):
        return {
            "schema": "merlin.loop-transform.prepared.v1",
            "identity": self.identity,
            "domain": self.domain,
            "buffers": [digest(x) for x in self.buffers],
            "settings": asdict(self.settings),
            "query_sha256": digest(self.query),
            "roots": list(self.roots),
            "authority": False,
        }


def _project(buffers, settings, reference, *, produce=False):
    bridge = Path(__file__).with_name("loop_transform_bridge.py")
    request = {"reference": str(reference), "buffers": buffers, "produce": produce, **asdict(settings)}
    environment = {**os.environ, "PYTHONPATH": str(reference / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        result = run_owned(
            [sys.executable, str(bridge)],
            input=canonical(request),
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Unavailable("producer bridge execution unavailable: " + str(exc)) from exc
    try:
        report = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise Unavailable("producer bridge did not return a complete result: " + result.stderr[-500:]) from exc
    if report.get("status") == "unsupported":
        raise Unsupported("host transformation attribution refused: " + report["reason"])
    if result.returncode or report.get("status") != "prepared":
        raise Unavailable("producer/projection failed: " + report.get("reason", result.stderr[-500:]))
    return report["capture"]


def prepare(buffers, *, reference, settings=Settings()):
    if type(settings) is not Settings:
        raise Unsupported("typed explicit settings required")
    settings.validate()
    if (
        type(buffers) is not tuple
        or len(buffers) != 3
        or any(type(x) is not str or len(x.encode()) > 4 * 1024 * 1024 for x in buffers)
    ):
        raise Unsupported("exact bounded3 complete buffer strings required")
    if buffers[0] == buffers[1] or buffers[1] == buffers[2]:
        raise Unsupported("unroll and segmentation must both actually change")
    reference = Path(reference).resolve()
    capture = _project(buffers, settings, reference)
    try:
        changed = _domain({"pipeline": capture})
    except ValueError as exc:
        raise Unsupported("scoped loop domain refused: " + str(exc)) from exc
    if len(changed) != 1:
        raise Unsupported("exactly one changed top-level loop function required")
    original = capture["stages"][0]["functions"][changed[0]]["before"]
    body = next(row["loop"]["body"] for row in original if "loop" in row)
    controls = {row["identity"] for row in capture["index_operations"]}
    external_calls = sum("effect" in row and row["effect"] not in controls for row in body)
    if settings.factor * external_calls > 16:
        raise Unsupported("at most sixteen retained external calls across the unrolled body required")
    query = emit(capture)
    # Every emitted theorem, including local body/carry/control/context obligations,
    # is audited; no caller-supplied root roster or local-only acceptance.
    roots = []
    namespace = []
    for line in query.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "namespace":
            namespace.append(words[1])
        elif words[0] == "end":
            namespace.pop()
        elif words[0] == "theorem":
            roots.append(".".join([*namespace, words[1]]))
    if not roots or roots[-1] != ROOT or len(query.encode()) > 4 * 1024 * 1024:
        raise Unavailable("complete bounded operational root missing")
    imports = [w for line in query.splitlines() if line.startswith("import ") for w in line.split()[1:]]
    sources = [asdict(row) for row in project_lean_sources(imports)]
    for name in (
        "loop_transform_intake.py",
        "loop_transform_bridge.py",
        "lean_graph_state_pipeline_emit.py",
        "lean_graph_state_pipeline.py",
        "lean_graph_state_pipeline_capture.py",
        "lean_graph_state_scalar_capture.py",
        "lean_graph_input_admission.py",
        "lean_graph_state_scalar.py",
        "lean_graph_pipeline_gate_profile_prepare.py",
        "lean_graph_pipeline_gate_projection.py",
        "loop_transform_gate.py",
        "loop_transform_cli.py",
    ):
        sources.append(file_ref(Path(__file__).with_name(name)))
    sources.append(file_ref(Path(__file__).parent / "lean/Merlin/GraphStateAudit.lean"))
    sources.append(file_ref(Path(__file__).parent.parent / "common/owned_process.py"))
    sources.extend(capture["producer_sources"])
    return PreparedLoop(buffers, settings, canonical(capture), query, tuple(roots), canonical(sources), str(reference))


def revalidate(prepared):
    if type(prepared) is not PreparedLoop or prepared.domain != DOMAIN:
        raise Unavailable("exact PreparedLoop required")
    for ref in json.loads(prepared.sources_json):
        actual = file_ref(ref["path"])
        if actual["sha256"] != ref["sha256"] or actual["bytes"] != ref.get("bytes", ref.get("size")):
            raise Unavailable("bound loop proof/host/producer source changed")


def produce(source, *, reference, settings=Settings()):
    """Run actual pinned producer without compiler-wide enforcement claims."""
    settings.validate()
    capture = _project((source,), settings, Path(reference).resolve(), produce=True)
    return tuple(capture["actual_buffers"])
