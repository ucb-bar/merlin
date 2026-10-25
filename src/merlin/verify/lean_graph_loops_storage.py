"""Bind actual counted/closed IR and freshly selected effects to Lean storage admission.

The parser, isolated Python adapter projection and physical backing declarations
are trusted. Acceptance describes the complete bounded loop representation and
its declared memory/completion safety, not a pipeline, numeric or hardware proof.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from merlin.sched.ir import parse_kernel

from .lean_graph_loops import (
    LeanSelection,
    _audit,
    _call_literal,
    _calls,
    _name,
    _projection_python_paths,
    _reference,
    project_pair,
)


@dataclass(frozen=True)
class ProducerSelection:
    source: Path
    revision: str
    example: str
    iterations: int
    source_files: tuple[Mapping, ...]
    adapter_identities: Mapping[str, str]
    initialized_external: tuple[str, ...]

    def __post_init__(self):
        # A frozen dataclass alone does not freeze dictionaries supplied by its
        # caller. Keep independent immutable selections throughout subprocesses.
        source = self.source.resolve(strict=True)
        object.__setattr__(self, "source", source)
        if not (source / self.example).resolve().is_relative_to(source / "examples"):
            raise ValueError("select an explicit producer example inside its pinned source")
        if (
            type(self.revision) is not str
            or len(self.revision) != 40
            or any(character not in "0123456789abcdef" for character in self.revision)
        ):
            raise ValueError("select an exact full producer revision")
        rows = []
        for original in self.source_files:
            row = dict(original)
            if set(row) != {"path", "sha256", "bytes", "git_blob"}:
                raise ValueError("producer source identities need exact file and Git blob bindings")
            if type(row["path"]) is not str or type(row["bytes"]) is not int or row["bytes"] <= 0:
                raise ValueError("producer source identities require a path and positive byte count")
            for field, size in (("sha256", 64), ("git_blob", 40)):
                value = row[field]
                if (
                    type(value) is not str
                    or len(value) != size
                    or any(char not in "0123456789abcdef" for char in value)
                ):
                    raise ValueError("producer source identities require exact hexadecimal digests")
            rows.append(MappingProxyType(row))
        identities = dict(self.adapter_identities)
        if set(identities) != {"configuration", "numerical_policy", "qualification"}:
            raise ValueError("select complete adapter semantic identities")
        inputs = tuple(self.initialized_external)
        for name in (*identities.values(), *inputs):
            _name(name)
        object.__setattr__(self, "source_files", tuple(rows))
        object.__setattr__(self, "adapter_identities", MappingProxyType(identities))
        object.__setattr__(self, "initialized_external", inputs)

    def dependencies(self) -> list[dict]:
        if type(self.iterations) is not int or self.iterations <= 0:
            raise ValueError("select the complete positive producer domain")
        source = self.source.resolve(strict=True)
        if (
            subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            != self.revision
        ):
            raise ValueError("selected producer revision changed")
        if not self.source_files or len({row["path"] for row in self.source_files}) != len(self.source_files):
            raise ValueError("select a complete unique producer dependency closure")
        paths = set()
        for row in self.source_files:
            path = Path(row["path"]).resolve(strict=True)
            if not path.is_relative_to(source):
                raise ValueError("selected producer dependency escaped its source checkout")
            if _reference(path) != {key: row[key] for key in ("path", "sha256", "bytes")}:
                raise ValueError("originally selected producer source changed")
            paths.add(path)
        if (source / self.example).resolve(strict=True) not in paths:
            raise ValueError("selected adapter example is absent from the dependency closure")
        if not self.initialized_external or len(set(self.initialized_external)) != len(self.initialized_external):
            raise ValueError("select explicit unique initialized external inputs")
        return [dict(row) for row in self.source_files]


def _natural(value, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError("storage geometry requires actual nonnegative integers and positive extents")
    return value


def _region(space, address, size):
    return f"⟨{_natural(space)}, {_natural(address)}, {_natural(size, positive=True)}⟩"


def _regions(rows):
    return "[" + ", ".join(rows) + "]"


def storage_obligation(projection: dict, effects: dict, *, initialized_external: tuple[str, ...]) -> str:
    """Project actual selected declarations; no call count/effect hints are used."""
    if effects["private_ownership"] is not True:
        raise ValueError("invocation-private ownership is an explicit required premise")
    banks, windows, objects = effects["banks"], effects["external_windows"], effects["objects"]
    names = [bank["name"] for bank in banks]
    external_names = [window["space"] for window in windows]
    if len(set(names)) != len(names) or len(set(external_names)) != len(external_names):
        raise ValueError("storage declarations require unique bank and external names")
    if set(names) & set(external_names) or not set(initialized_external) <= set(external_names):
        raise ValueError("private banks and selected initialized external windows must be distinct")
    if len({bank["backing"] for bank in banks}) != len(banks):
        raise ValueError("multiple typed views of one backing need a separately checked view geometry")
    for name in names + external_names + [bank["backing"] for bank in banks]:
        _name(name)
    spaces = {name: index for index, name in enumerate(names + external_names)}
    bank_map = {bank["name"]: bank for bank in banks}
    allocations = []
    if len({obj["name"] for obj in objects}) != len(objects):
        raise ValueError("private object declarations must be unique")
    for obj in objects:
        _name(obj["name"])
        bank = bank_map[obj["bank"]]
        if bank["unit"] != "bytes":
            raise ValueError("this storage family requires explicit byte-addressed geometry")
        base, size = _natural(obj["source_base"]), _natural(obj["size"], positive=True)
        alignment = math.lcm(_natural(obj["alignment"], positive=True), _natural(bank["alignment"], positive=True))
        region = _region(spaces[obj["bank"]], base, size)
        allocations.append(
            f"⟨{region}, {_natural(bank['start'])}, {_natural(bank['capacity'], positive=True)}, {alignment}⟩"
        )
    external = [_region(spaces[row["space"]], row["start"], row["size"]) for row in windows]
    initialized = [
        region for row, region in zip(windows, external, strict=True) if row["space"] in initialized_external
    ]
    blocks = effects["blocks"]
    if len(blocks) != projection["extent"] or any(len(block) != len(projection["body"]) for block in blocks):
        raise ValueError("effect rows must cover every call of every actual iteration")
    tokens = set()
    for block in blocks:
        for row in block:
            if row["token"] is not None:
                _name(row["token"])
                tokens.add(row["token"])
            for token in row["waits"]:
                _name(token)
                tokens.add(token)
    token_ids = {token: index for index, token in enumerate(sorted(tokens))}
    event_blocks = []
    offset = 0
    for block in blocks:
        events = []
        for row in block:
            if set(row) != {"accesses", "completion", "elementwise", "exact_copy", "semantic_key", "token", "waits"}:
                raise ValueError("unknown effect fields require an explicit projection extension")
            _name(row["semantic_key"])
            if type(row["exact_copy"]) is not bool or row["elementwise"] is not None:
                raise ValueError("unsupported effect semantic declarations")
            accesses = []
            for access in row["accesses"]:
                if set(access) != {"space", "start", "size", "writes"} or type(access["writes"]) is not bool:
                    raise ValueError("unknown or untyped actual access fields")
                if access["space"] not in spaces:
                    raise ValueError("actual access has no selected physical address window")
                region = _region(spaces[access["space"]], access["start"], access["size"])
                accesses.append(f"⟨{region}, {str(access['writes']).lower()}⟩")
            completion = row["completion"]
            if completion == "synchronous" and row["token"] is None and not row["waits"]:
                effect = f"(.synchronous {_regions(accesses)})"
            elif completion == "asynchronous" and row["token"] is not None and not row["waits"]:
                effect = f"(.asynchronous {token_ids[row['token']]} {_regions(accesses)})"
            elif completion == "wait" and row["token"] is None and not accesses and row["waits"]:
                effect = f"(.wait {_regions([str(token_ids[token]) for token in row['waits']])})"
            else:
                raise ValueError("unknown or inconsistent actual completion effect")
            call = _call_literal(projection["target"][offset], closed=True)
            events.append(f"⟨{call}, {effect}⟩")
            offset += 1
        event_blocks.append(_regions(events))
    source = (
        f"⟨{_name(projection['variable'])}, {projection['extent']}, {projection['bits']}, "
        f"{_calls(projection['body'], closed=False)}⟩"
    )
    roots = "\n".join("#print axioms " + root for root in STORAGE_ROOTS)
    return f"""import Merlin.GraphLoopsEpochs
import Merlin.GraphLoopsBounds
set_option autoImplicit false
open Merlin GraphLoops GraphLoops.Storage GraphLoops.Bound
namespace GraphLoopStorageObligation
def source : Counted := {source}
def actual : List ClosedCall := {_calls(projection["target"], closed=True)}
def allocations : List Allocation := {_regions(allocations)}
def external : List Region := {_regions(external)}
def initialized : List Region := {_regions(initialized)}
def events : List (List Event) := {_regions(event_blocks)}
theorem accepted : Epochs.check source actual allocations external initialized events = true := by cbv
theorem complete :
    actual = (events.map (List.map Event.call)).flatten ∧
    actual.length = source.extent * source.body.length ∧ events.length = source.extent ∧
    ∀ block ∈ events, block.length = source.body.length ∧
      ∃ final, Trace (external ++ allocations.map Allocation.region) (allocations.map Allocation.region)
        ⟨initialized, [], []⟩ (block.map Event.effect) final ∧ final.pending = [] ∧
        ∀ object ∈ allocations.map Allocation.region, Covered final.initialized object :=
  Bound.complete_boundary (Epochs.check_sound accepted).1
theorem preserves {{Value : Type}} (run : ClosedCall → StateEffects.State Value → StateEffects.State Value)
    (initial : StateEffects.State Value) :
    (events.map (List.map Event.call)).foldl
      (fun state block => StateEffects.execute (block.map run) state) initial =
      StateEffects.execute (actual.map run) initial := Bound.preserves (Epochs.check_sound accepted).1 run initial
theorem index_bounds : ∀ call ∈ actual, ClosedCall.Fits source.bits call :=
  checked_call_bounds (Bound.check_sound (Epochs.check_sound accepted).1).1
theorem complete_dynamic_drain :
    ∃ final, Epochs.Trace (external ++ allocations.map Allocation.region) (allocations.map Allocation.region)
      ⟨initialized, [], []⟩ (events.flatten.map Event.effect) final ∧ final.pending = [] :=
  (Epochs.check_sound accepted).2
end GraphLoopStorageObligation
{roots}
"""


STORAGE_ROOTS = (
    "GraphLoopStorageObligation.accepted",
    "GraphLoopStorageObligation.complete",
    "GraphLoopStorageObligation.preserves",
    "GraphLoopStorageObligation.index_bounds",
    "Merlin.GraphLoops.checked_call_bounds",
    "GraphLoopStorageObligation.complete_dynamic_drain",
    "Merlin.GraphLoops.Epochs.check_sound",
    "Merlin.GraphLoops.Epochs.execute_sound",
    "Merlin.GraphLoops.Epochs.Trace.event",
    "Merlin.GraphLoops.Epochs.active_token_refused",
    "Merlin.GraphLoops.Bound.check_sound",
    "Merlin.GraphLoops.Bound.complete_boundary",
    "Merlin.GraphLoops.Bound.preserves",
    "Merlin.GraphLoops.Bound.allocation_point",
    "Merlin.GraphLoops.Storage.block_domain",
    "Merlin.GraphLoops.Storage.periods_sound",
    "Merlin.GraphLoops.Storage.Trace.event",
    "Merlin.GraphLoops.Storage.step_safe",
    "Merlin.GraphLoops.Storage.release_sound",
    "Merlin.GraphLoops.Storage.waitAll_sound",
    "Merlin.GraphLoops.Storage.pending_no_overwrite",
)


def _storage_audit(stdout):
    return _audit(stdout, STORAGE_ROOTS)


def verify_storage_files(
    before_path,
    after_path,
    *,
    selection: LeanSelection,
    producer: ProducerSelection,
    output: Path,
    index_bits: int,
    max_dynamic_calls: int,
    effects_path: Path | None = None,
):
    """Retain early projection refusals as unavailable evidence, with no receipt."""
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _verify_storage_files(
            before_path,
            after_path,
            selection=selection,
            producer=producer,
            output=output,
            index_bits=index_bits,
            max_dynamic_calls=max_dynamic_calls,
            effects_path=effects_path,
        )
    except Exception as error:
        (output / "REFUSAL.json").write_text(
            json.dumps(
                {
                    "schema": "merlin.graph.loop_storage_refusal.v1",
                    "error": str(error),
                    "type": type(error).__name__,
                    "lean_storage_checked": False,
                    "complete_boundary_checked": False,
                    "pipeline_receipt_issued": False,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        raise


def _verify_storage_files(
    before_path,
    after_path,
    *,
    selection,
    producer,
    output,
    index_bits,
    max_dynamic_calls,
    effects_path,
):
    before_path, after_path = before_path.resolve(strict=True), after_path.resolve(strict=True)
    before_bytes, after_bytes = before_path.read_bytes(), after_path.read_bytes()
    originals = [_reference(before_path, before_bytes), _reference(after_path, after_bytes)]
    projection = project_pair(
        parse_kernel(before_bytes.decode()),
        parse_kernel(after_bytes.decode()),
        index_bits=index_bits,
        max_dynamic_calls=max_dynamic_calls,
    )
    dependencies, producer_dependencies = selection.dependencies(), producer.dependencies()
    if {attempt["module"] for attempt in json.loads(selection.build_record.read_bytes())["attempts"]} != {
        "Merlin.StateEffects",
        "Merlin.ResourceGeometry",
        "Merlin.GraphLoops",
        "Merlin.GraphLoopsStorage",
        "Merlin.GraphLoopsBound",
        "Merlin.GraphLoopsBounds",
        "Merlin.GraphLoopsEpochs",
    }:
        raise ValueError("select the complete existing-root loop/storage proof dependency closure")
    from . import lean_graph_loops_effects

    python_paths = [
        *_projection_python_paths(),
        Path(__file__).resolve(),
        Path(lean_graph_loops_effects.__file__).resolve(),
    ]
    python_dependencies = [_reference(path) for path in python_paths]
    effects_bytes = effects_path.read_bytes() if effects_path is not None else None
    effects_ref = _reference(effects_path, effects_bytes) if effects_path is not None else None
    before_copy, after_copy = output / "before.kernel", output / "after.kernel"
    before_copy.write_bytes(before_bytes)
    after_copy.write_bytes(after_bytes)
    fresh = output / "fresh-effects.json"
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(producer.source.resolve() / "src"), PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable,
        str(Path(lean_graph_loops_effects.__file__).resolve()),
        "--source",
        str(producer.source.resolve()),
        "--revision",
        producer.revision,
        "--example",
        producer.example,
        "--iterations",
        str(producer.iterations),
        "--before",
        str(before_copy.resolve()),
        "--after",
        str(after_copy.resolve()),
        "--output",
        str(fresh.resolve()),
        "--index-bits",
        str(index_bits),
        "--max-dynamic-calls",
        str(max_dynamic_calls),
    ]
    projected = subprocess.run(command, capture_output=True, text=True, env=environment, check=False, timeout=120)
    (output / "projection.stdout").write_text(projected.stdout)
    (output / "projection.stderr").write_text(projected.stderr)
    if projected.returncode:
        raise ValueError("actual selected adapter projection refused; raw logs retained")
    fresh_bytes = fresh.read_bytes()
    fresh_ref = _reference(fresh, fresh_bytes)
    effects = json.loads(fresh_bytes)
    selected_files = {row["path"]: row for row in producer_dependencies}
    fresh_files = effects["producer_source"]
    if (
        not fresh_files
        or any(selected_files.get(row["path"]) != row for row in fresh_files)
        or effects["adapter_identities"] != producer.adapter_identities
    ):
        raise ValueError("fresh adapter/source identity differs from the originally selected context")
    if effects["before"] != _reference(before_copy, before_bytes) or effects["after"] != _reference(
        after_copy, after_bytes
    ):
        raise ValueError("fresh effects are not bound to the actual checked input buffers")
    if effects_bytes is not None:
        candidate = json.loads(effects_bytes)
        for key in ("blocks", "objects", "banks", "external_windows", "private_ownership", "adapter_identities"):
            if candidate[key] != effects[key]:
                raise ValueError("saved effects/declarations differ from fresh actual selected projection")
    query = output / "Obligation.lean"
    query_bytes = storage_obligation(projection, effects, initialized_external=producer.initialized_external).encode()
    query.write_bytes(query_bytes)
    query_ref = _reference(query, query_bytes)
    environment.update(
        LEAN_PATH=str(selection.library), LEAN_SYSROOT=str(selection.executable.parent.parent), LEAN_NUM_THREADS="2"
    )
    lean_command = [
        str(selection.executable),
        "-DautoImplicit=false",
        "-DrelaxedAutoImplicit=false",
        str(query.resolve()),
    ]
    result = subprocess.run(lean_command, capture_output=True, text=True, env=environment, check=False, timeout=120)
    (output / "stdout.log").write_text(result.stdout)
    (output / "stderr.log").write_text(result.stderr)
    record = {
        "schema": "merlin.graph.loop_storage_check.v1",
        "originals": originals,
        "effects": effects_ref,
        "fresh_projection": fresh_ref,
        "query": query_ref,
        "projection_command": command,
        "command": lean_command,
        "returncode": result.returncode,
        "dependencies": dependencies,
        "producer_dependencies": producer_dependencies,
        "trusted_python_dependencies": python_dependencies,
        "source_admission": effects["source_admission"],
        "lean_storage_checked": False,
        "complete_boundary_checked": False,
        "python_projection_trusted": True,
        "adapter_independently_qualified": False,
        "initialized_external_premise": list(producer.initialized_external),
        "external_backing_nonaliasing_premise": True,
        "pipeline_receipt_issued": False,
        "numerical_gates_discharged": False,
        "external_output_coverage_proved": False,
        "renderer_or_hardware_proved": False,
    }
    if (
        originals != [_reference(before_path), _reference(after_path)]
        or dependencies != selection.dependencies()
        or producer_dependencies != producer.dependencies()
        or python_dependencies != [_reference(path) for path in python_paths]
        or effects_ref != (_reference(effects_path) if effects_path is not None else None)
        or before_copy.read_bytes() != before_bytes
        or after_copy.read_bytes() != after_bytes
        or fresh_ref != _reference(fresh)
        or query_ref != _reference(query)
    ):
        record["refusal"] = "actual inputs, effects or selected dependencies changed during checking"
    elif result.returncode:
        record["refusal"] = "Lean refused complete actual storage/call boundary"
    else:
        record["transitive_axioms"] = _storage_audit(result.stdout)
        record["lean_storage_checked"] = True
        record["complete_boundary_checked"] = effects["source_admission"]["accepted"]
        if not record["complete_boundary_checked"]:
            record["refusal"] = (
                "independent production admission refused; kernel result alone does not release this boundary"
            )
    (output / "CHECK.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record
