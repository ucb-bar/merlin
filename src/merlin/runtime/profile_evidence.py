"""External, byte-bound region reporting without executing or changing a model.

This reader joins explicit source labels to already observed counter intervals.
It keeps functional counts, modeled cycles and hardware cycles in separate
domains. Caller intervals and their overlap never establish physical unit busy
time, traffic, numerical equivalence or a common clock between independent runs.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from merlin.common.strict_json import loads

KINDS = frozenset({"functional_instructions", "modeled_cycles", "hardware_cycles"})
LABELS = ("model", "part", "stage", "fqn", "function", "variant", "route", "scope")


def _uint(value: Any, name: str, bits: int = 64) -> int:
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer")
    return value


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"explicit nonempty {name} required")
    return value


@dataclass(frozen=True)
class BoundFile:
    """Exact existing input bytes; a digest is identity, not evidence authority."""

    path: str
    sha256: str

    def read(self, *, max_bytes: int = 128 * 1024 * 1024) -> bytes:
        if not Path(self.path).is_absolute():
            raise ValueError("profile inputs require absolute selected paths")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValueError("profile inputs require lowercase SHA256 identities")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("positive input byte budget required")
        with Path(self.path).open("rb") as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("profile input exceeds the byte budget")
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError("profile input changed: " + self.path)
        return raw

    def verify(self) -> None:
        """Stream large existing ELFs without materializing them in memory."""
        if not Path(self.path).is_absolute():
            raise ValueError("profile inputs require absolute selected paths")
        with Path(self.path).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != self.sha256:
            raise ValueError("profile input changed: " + self.path)


@dataclass(frozen=True)
class CounterDomain:
    """Caller-selected counter meaning and optional independently qualified Hz.

    An absent Hz mapping keeps raw-counter display coordinates. A shared epoch
    is never inferred from equal frequencies or from matching hart numbers.
    """

    kind: str
    semantics: str
    bits: int = 64
    clock_hz: int | None = None

    def validate(self) -> None:
        if self.kind not in KINDS:
            raise ValueError("explicit functional, modeled or hardware counter kind required")
        _name(self.semantics, "counter semantics")
        if type(self.bits) is not int or self.bits not in (32, 64):
            raise ValueError("counter width must be 32 or 64 bits")
        if self.clock_hz is not None:
            if self.kind == "functional_instructions" or type(self.clock_hz) is not int or self.clock_hz <= 0:
                raise ValueError("Hz conversion requires explicit positive cycle-counter frequency")


@dataclass(frozen=True)
class Gate:
    """A selected existing pass field, with its numerical method left explicit.

    The caller owns association of this gate with the execution. This reducer
    verifies bytes and the selected true value; it does not infer independent
    golden coverage from a filename, checksum, console exit or pass boolean.
    """

    file: BoundFile
    passed_path: tuple[str | int, ...]
    method: str
    coverage: str

    def verify(self) -> dict:
        _name(self.method, "numerical gate method")
        _name(self.coverage, "numerical gate coverage")
        if not self.passed_path:
            raise ValueError("explicit numerical-gate pass-field path required")
        document = loads(self.file.read())
        if _select(document, self.passed_path) is not True:
            raise ValueError("selected existing numerical gate did not pass")
        return asdict(self)


@dataclass(frozen=True)
class RegionSource:
    """One existing execution and its explicit, nontransferable identities.

    ``labels`` is optional JSON with ``elf_sha256`` and an ``entries`` array.
    Each entry names case_id/region/hart and optional LABELS. Labels are declared
    source/call-site attribution, not PC sampling or function self time.
    """

    name: str
    records: BoundFile
    elf: BoundFile
    gate: Gate
    domain: CounterDomain
    expected_regions: int
    record_path: tuple[str | int, ...] = ()
    console_prefix: str | None = None
    labels: BoundFile | None = None


def _select(value: Any, path: tuple[str | int, ...]) -> Any:
    for key in path:
        if isinstance(value, dict) and isinstance(key, str) and key in value:
            value = value[key]
        elif isinstance(value, list) and type(key) is int and 0 <= key < len(value):
            value = value[key]
        else:
            raise ValueError("selected profile field does not exist")
    return value


def parse_console(console: str, *, prefix: str, max_regions: int) -> list[dict]:
    """Read caller-named JSON records, with no target-specific marker assumed."""
    _name(prefix, "record prefix")
    if any(c.isspace() for c in prefix):
        raise ValueError("record prefix must be one complete token")
    if type(max_regions) is not int or max_regions < 1:
        raise ValueError("positive region budget required")
    rows = []
    for line in console.splitlines():
        token, separator, body = line.strip().partition(" ")
        if token != prefix:
            continue
        if not separator or not body.strip():
            raise ValueError("missing JSON region payload")
        row = loads(body)
        if type(row) is not dict:
            raise ValueError("region payload must be a JSON object")
        rows.append(row)
        if len(rows) > max_regions:
            raise ValueError("observed regions exceed the declared budget")
    return rows


def _identity(row: dict) -> tuple[str, str, int]:
    return (
        _name(row.get("case_id"), "case id"),
        _name(row.get("region"), "region name"),
        _uint(row.get("hart"), "hart"),
    )


def _labels(source: RegionSource) -> dict:
    if source.labels is None:
        return {}
    document = loads(source.labels.read())
    if type(document) is not dict or document.get("elf_sha256") != source.elf.sha256:
        raise ValueError("attribution sidecar must bind the exact ELF")
    entries = document.get("entries")
    if type(entries) is not list:
        raise ValueError("attribution sidecar entries must be an array")
    output = {}
    for row in entries:
        if type(row) is not dict:
            raise ValueError("attribution entries must be objects")
        key = _identity(row)
        if key in output:
            raise ValueError("duplicate attribution identity")
        labels = {name: row.get(name) for name in LABELS}
        if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in labels.values()):
            raise ValueError("attribution labels must be nonempty strings or null")
        output[key] = labels
    return output


def interval_coverage(intervals: list[tuple[int, int]]) -> dict:
    """Inclusive sums, union and observed overlap, without double counting.

    Overlap means overlapping recorded software intervals, including nesting.
    It does not imply concurrent execution by physical engines.
    """
    edges: dict[int, int] = defaultdict(int)
    inclusive = 0
    for start, end in intervals:
        _uint(start, "interval start")
        _uint(end, "interval end")
        if end < start:
            raise ValueError("reversed interval or counter wrap requires separate admission")
        inclusive += end - start
        edges[start] += 1
        edges[end] -= 1
    active = union = overlap = maximum = 0
    previous = None
    for position, delta in sorted(edges.items()):
        if previous is not None:
            span = position - previous
            union += span if active else 0
            overlap += span if active > 1 else 0
        active += delta
        maximum = max(maximum, active)
        previous = position
    return {
        "inclusive_ticks": inclusive,
        "union_ticks": union,
        "overlapping_interval_ticks": overlap,
        "maximum_recorded_intervals": maximum,
        "scope": "observed caller intervals; not physical engine overlap",
    }


def collect(sources: list[RegionSource], *, max_regions: int = 100_000) -> tuple[dict, dict]:
    """Validate and reduce existing executions into JSON and Chrome/Perfetto.

    Inputs are verified again after collection. No binary is executed, no time
    is modeled, no counter is programmed, and no missing metric becomes zero.
    """
    if type(max_regions) is not int or max_regions < 1 or not sources:
        raise ValueError("nonempty source roster and positive region budget required")
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("source names must be unique; independent runs cannot share a domain")
    records, source_reports, artifacts = [], [], []
    for source in sources:
        _name(source.name, "execution source name")
        source.domain.validate()
        if (
            type(source.expected_regions) is not int
            or source.expected_regions < 1
            or source.expected_regions > max_regions - len(records)
        ):
            raise ValueError("positive complete region count within the total budget required")
        source.elf.verify()
        gate = source.gate.verify()
        raw = source.records.read()
        if source.console_prefix is not None:
            if source.record_path:
                raise ValueError("console and JSON path selection are mutually exclusive")
            rows = parse_console(raw.decode("utf-8"), prefix=source.console_prefix, max_regions=source.expected_regions)
        else:
            rows = _select(loads(raw), source.record_path)
        if type(rows) is not list or len(rows) != source.expected_regions:
            raise ValueError("complete expected region roster is missing")
        labels = _labels(source)
        seen, own = set(), []
        for raw_row in rows:
            if type(raw_row) is not dict:
                raise ValueError("region records must be objects")
            identity = _identity(raw_row)
            if identity in seen:
                raise ValueError("duplicate region identity; repeated trials require unique ids")
            seen.add(identity)
            start = _uint(raw_row.get("start_cycle"), "counter start", source.domain.bits)
            end = _uint(raw_row.get("end_cycle"), "counter end", source.domain.bits)
            if end < start:
                raise ValueError("counter wrap, reset or reversed region is not admitted")
            if raw_row.get("pass") is not True:
                raise ValueError("explicit passed region correctness observation required")
            if "expected_checksum" in raw_row:
                if _uint(raw_row.get("checksum"), "checksum") != _uint(
                    raw_row["expected_checksum"], "expected checksum"
                ):
                    raise ValueError("region checksum differs from its declared oracle")
            retired = raw_row.get("instret_delta")
            if retired is not None:
                _uint(retired, "retirement delta")
            endpoints = "start_instret" in raw_row, "end_instret" in raw_row
            if any(endpoints):
                if not all(endpoints):
                    raise ValueError("both retirement endpoints required")
                first = _uint(raw_row["start_instret"], "retirement start")
                last = _uint(raw_row["end_instret"], "retirement end")
                if last < first or retired != last - first:
                    raise ValueError("retirement endpoints reset, wrap or disagree with delta")
            attribution = labels.get(identity, dict.fromkeys(LABELS))
            row = {
                "source": source.name,
                "case_id": identity[0],
                "region": identity[1],
                "hart": identity[2],
                "start": start,
                "end": end,
                "ticks": end - start,
                "kind": source.domain.kind,
                "counter_semantics": source.domain.semantics,
                "cycles": None if source.domain.kind == "functional_instructions" else end - start,
                "retired_instructions": retired,
                "retired_per_cycle": (
                    retired / (end - start)
                    if retired is not None and end > start and source.domain.kind != "functional_instructions"
                    else None
                ),
                "attribution": attribution,
                "attribution_basis": "explicit ELF-bound sidecar" if identity in labels else "unknown",
                "function_self_time": None,
                "physical_busy": None,
                "physical_memory_traffic_bytes": None,
                "raw_record": dict(raw_row),
            }
            records.append(row)
            own.append(row)
        if labels.keys() - seen:
            raise ValueError("attribution sidecar contains an unobserved region")
        source_reports.append(
            {
                "source": source.name,
                "elf": asdict(source.elf),
                "domain": asdict(source.domain),
                "regions": len(own),
                "numerical_gate": gate,
                "numerical_gate_association": "caller-selected; no independent authority inferred",
                "FQN_labelled_regions": sum(r["attribution"]["fqn"] is not None for r in own),
                "unattributed_regions": sum(r["attribution"]["part"] is None for r in own),
            }
        )
        artifacts.extend([source.records, source.elf, source.gate.file])
        if source.labels is not None:
            artifacts.append(source.labels)
    groups: dict[tuple, list] = defaultdict(list)
    tracks: dict[tuple, list] = defaultdict(list)
    for row in records:
        label = row["attribution"]
        groups[(row["source"], row["hart"], *(label[name] for name in LABELS))].append(row)
        tracks[(row["source"], row["hart"])].append(row)
    parts = []
    for key, rows in groups.items():
        source, hart, *values = key
        parts.append(
            {
                "source": source,
                "hart": hart,
                **dict(zip(LABELS, values, strict=True)),
                "regions": len(rows),
                "kind": rows[0]["kind"],
                "counter_semantics": rows[0]["counter_semantics"],
                "tick_samples": [r["ticks"] for r in rows],
                "median_ticks": statistics.median(r["ticks"] for r in rows),
                "min_ticks": min(r["ticks"] for r in rows),
                "max_ticks": max(r["ticks"] for r in rows),
                "cycles_median": (
                    statistics.median(r["ticks"] for r in rows)
                    if rows[0]["kind"] != "functional_instructions"
                    else None
                ),
                **interval_coverage([(r["start"], r["end"]) for r in rows]),
                "inclusive_retired_instructions": (
                    sum(r["retired_instructions"] for r in rows)
                    if all(r["retired_instructions"] is not None for r in rows)
                    else None
                ),
                "physical_busy": None,
                "physical_memory_traffic_bytes": None,
            }
        )
    events = []
    domains = {source.name: source.domain for source in sources}
    track_reports = []
    for pid, ((source, hart), rows) in enumerate(tracks.items(), 1):
        domain = domains[source]
        origin = min(r["start"] for r in rows)
        scale = 1_000_000 / domain.clock_hz if domain.clock_hz is not None else 1
        events.append(
            {
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "name": "process_name",
                "args": {"name": f"{source} hart {hart}: {domain.kind}"},
            }
        )
        for row in rows:
            label = row["attribution"]
            name = label["fqn"] or label["part"] or row["case_id"]
            if label["variant"]:
                name += " / " + label["variant"]
            events.append(
                {
                    "ph": "X",
                    "pid": pid,
                    "tid": 0,
                    "name": name,
                    "cat": domain.kind,
                    "ts": (row["start"] - origin) * scale,
                    "dur": row["ticks"] * scale,
                    "args": {
                        **label,
                        "region": row["region"],
                        "case_id": row["case_id"],
                        "raw_start": row["start"],
                        "raw_end": row["end"],
                        "raw_ticks": row["ticks"],
                        "counter_semantics": domain.semantics,
                        "coordinate_is_elapsed_time": domain.clock_hz is not None,
                    },
                }
            )
        track_reports.append(
            {
                "source": source,
                "hart": hart,
                "origin": origin,
                "independent_origin": True,
                **interval_coverage([(r["start"], r["end"]) for r in rows]),
            }
        )
    for artifact in artifacts:
        artifact.verify()
    report = {
        "schema": "merlin.external_profile_evidence.v1",
        "sources": source_reports,
        "regions": records,
        "parts": parts,
        "tracks": track_reports,
        "input_sha256": {a.path: a.sha256 for a in artifacts},
        "physical_busy_available": False,
        "physical_traffic_available": False,
        "cross_hart_epoch_alignment_established": False,
        "timing_accuracy_certified": False,
        "scope": "external reduction of existing caller observations; no execution or fitting",
    }
    trace = {
        "traceEvents": events,
        "displayTimeUnit": "ns",
        "otherData": {
            "schema": report["schema"],
            "sources": source_reports,
            "coordinates": (
                "Each source/hart has its own origin. Without explicit clock_hz, one raw counter tick "
                "is displayed as one microsecond coordinate, not elapsed time."
            ),
            "cross_hart_epoch_alignment_established": False,
            "overlap_scope": "Recorded software intervals only, not hardware-unit busy overlap",
        },
    }
    return report, trace


def attribute_pc_census(
    elf: BoundFile,
    census: BoundFile,
    symbolizer: BoundFile,
    *,
    companion: BoundFile | None = None,
    max_elf_bytes: int = 128 * 1024 * 1024,
) -> dict:
    """Join an existing PC census to exact-ELF symbolizer records.

    Reuses the existing structural companion admission and complete-PC reducer.
    This attributes sampled/executed PC counts, never region duration or function
    self time. Source labels in ``collect`` remain an independent contract.
    """
    from merlin.perf.debug_companion import attribute_symbolized_pcs, verify_debug_companion

    raw = elf.read(max_bytes=max_elf_bytes)
    elf_admission = verify_debug_companion(raw, raw)
    proof = None
    selected = elf
    if companion is not None:
        proof = verify_debug_companion(raw, companion.read(max_bytes=max_elf_bytes))
        selected = companion
    count_doc, symbol_doc = loads(census.read()), loads(symbolizer.read())
    if (
        type(count_doc) is not dict
        or count_doc.get("elf_sha256") != elf.sha256
        or type(symbol_doc) is not dict
        or symbol_doc.get("elf_sha256") != selected.sha256
    ):
        raise ValueError("PC and symbolizer sidecars must bind their exact ELF bytes")
    counts = {}
    if type(count_doc.get("pc_counts")) is not list:
        raise ValueError("explicit PC/count roster required")
    for item in count_doc["pc_counts"]:
        if type(item) is not dict:
            raise ValueError("PC counts must be objects")
        pc, count = _uint(item.get("pc"), "PC"), _uint(item.get("count"), "PC count")
        if pc in counts:
            raise ValueError("duplicate PC census identity")
        counts[pc] = count
    if type(symbol_doc.get("records")) is not list:
        raise ValueError("complete structural symbolizer records required")
    result = attribute_symbolized_pcs(counts, symbol_doc["records"])
    for artifact in (elf, census, symbolizer, *([companion] if companion else [])):
        artifact.verify()
    return {
        "schema": "merlin.external_PC_attribution.v1",
        "total_counts": result["total"],
        "functions": result["functions"],
        "source_lines": [{"file": key[0], "line": key[1], "count": value} for key, value in result["lines"].items()],
        "inline_stacks": [
            {"frames": [list(frame) for frame in key], "count": value} for key, value in result["inline_stacks"].items()
        ],
        "unknown_function_counts": result["functions"].get("", 0),
        "elf": asdict(elf),
        "symbolizer_ELF": asdict(selected),
        "ELF_admission": elf_admission,
        "companion_admission": proof,
        "input_sha256": {
            artifact.path: artifact.sha256
            for artifact in (elf, census, symbolizer, *([companion] if companion else []))
        },
        "scope": "existing PC counts; not elapsed function time, physical busy or traffic",
    }


def write_report(output: str | Path, report: dict, trace: dict) -> None:
    """Create a fresh report directory; existing results are never overwritten."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (directory / "trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
