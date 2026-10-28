"""Per-case Spike cycle-model timing of a frozen compiler through an export's own grader.

Each case runs in a fresh child (``timing_child.py``) whose ``PYTHONPATH`` is the
export's, so grading, the timing observer and the model pin are exactly what the
runs used.  The child's CPU set, I/O class and niceness come from the caller's
``--cpus``; nothing here picks a CPU.  A case whose complete timing-model
estimate does not finish in its budget gets the calibrated fast estimate from
its emitted command census (labelled as such), so no row is left without cycles
when the census exists.  Only a case with neither stays ``UNKNOWN``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .structure import FAMILY_WORDS

CHILD = Path(__file__).with_name("timing_child.py")
STORE_CLASSES = ("store", "pool_store")
COMPUTE_CLASS = "compute"
LOAD_CLASS = "load"
FENCE_KEY = "target.fences"
CLASS_PREFIX = "target.class."
PUBLICATION_PREFIX = "target.load_publication."
EVIDENCE_TOKENS = ("refus", "fallback", "capacity")


@dataclass
class Grader:
    """Everything a child needs to grade one case exactly as the runs do."""

    export: Path
    descriptor: Path
    suite_root: Path
    python: str
    rtl_facts: Path | None = None
    compiler_library: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    cpus: str | None = None
    nice: int = 19
    budget_s: int = 900
    slack_s: int = 600

    def child_env(self, work: Path) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.env)
        roots = [self.export / "src", self.export / "packages" / "merlin-experiments" / "src",
                 self.export / "packages" / "merlin-timing" / "src"]
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in roots)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TMPDIR"] = str(work)
        # Never inherit a live campaign's state: its deadline and observation ledger belong to that run.
        env.pop("MERLIN_JOINT_CAMPAIGN_STATE", None)
        return env

    def prefix(self) -> list[str]:
        if not self.cpus:
            return []
        return ["ionice", "-c3", "nice", "-n", str(self.nice), "taskset", "-c", self.cpus]


def case_family(case: str) -> str:
    """Operation family of a case id from its words (generic op words, never target names)."""
    words = case.lower().split("_")
    for word in words:
        if word.startswith("m1k") and word[3:4].isdigit():
            return "decode"
        for family, vocabulary in FAMILY_WORDS:
            if word in vocabulary or (family == "conv" and word.startswith("conv")):
                return family
    for word in words:
        if word.startswith("m") and "k" in word and "n" in word and word[1:2].isdigit():
            return "dense"
    return "other"


def _rmtree(path: Path) -> None:
    if not path.exists():
        return
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            try:
                os.chmod(os.path.join(root, name), 0o700)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def run_child(grader: Grader, submission: Path, case: str, work: Path) -> dict:
    """Grade one case in a fresh child; returns the child's JSON plus its output directory."""
    work.mkdir(parents=True, exist_ok=True)
    out, result = work / "grade", work / "result.json"
    _rmtree(out)
    result.unlink(missing_ok=True)
    argv = grader.prefix() + [
        grader.python, "-P", str(CHILD), "--export", str(grader.export), "--descriptor", str(grader.descriptor),
        "--suite-root", str(grader.suite_root), "--submission", str(submission), "--case", case,
        "--out", str(out), "--result", str(result), "--timeout", str(grader.budget_s),
    ]
    if grader.rtl_facts:
        argv += ["--rtl-facts", str(grader.rtl_facts)]
    if grader.compiler_library:
        argv += ["--compiler-library", str(grader.compiler_library)]
    started = time.monotonic()
    doc: dict
    try:
        with (work / "child.log").open("wb") as log:
            subprocess.run(argv, cwd=work, env=grader.child_env(work), stdout=log, stderr=subprocess.STDOUT,
                           timeout=grader.budget_s + grader.slack_s, check=False)
        doc = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else {"error": "child wrote no result"}
    except subprocess.TimeoutExpired:
        doc = {"error": f"child exceeded {grader.budget_s + grader.slack_s} s"}
    except (OSError, ValueError) as error:
        doc = {"error": f"{type(error).__name__}: {error}"}
    doc.setdefault("wall_s", round(time.monotonic() - started, 2))
    doc["grade_dir"] = str(out)
    return doc


def case_run_dir(grade_dir: Path, case: str) -> Path | None:
    for path in grade_dir.glob(f"runs/*/*/{case}"):
        if (path / "generated").is_dir():
            return path
    return None


def _sum_classes(counts: dict, names: tuple[str, ...]) -> int:
    return sum(int(v) for k, v in counts.items()
               if k.startswith(CLASS_PREFIX) and k[len(CLASS_PREFIX):].partition("_")[0] in names
               and isinstance(v, (int, float)))


def _roofline(structural: dict) -> tuple[float | None, str | None]:
    terms: dict[str, float] = {}
    compute = structural.get("structural_compute_floor_cycles")
    if isinstance(compute, (int, float)) and compute > 0:
        terms["compute"] = float(compute)
    for side in ("read", "write"):
        value = ((structural.get("logical_request_ideal_service") or {}).get(side) or {}).get(
            "ideal_requested_service_cycles")
        if isinstance(value, (int, float)) and value > 0:
            terms[f"{side} requests"] = float(value)
    if not terms:
        return None, None
    name = max(terms, key=terms.get)
    return terms[name], name


def _evidence_lines(run_dir: Path | None, limit: int = 6) -> list[str]:
    """Lines of the compiler's own stderr that mention a refusal, fallback or capacity decision."""
    if run_dir is None:
        return []
    hits: list[str] = []
    for path in sorted((run_dir / "generated" / "invocations").glob("*/stderr.bin")):
        try:
            text = path.read_bytes()[: 1 << 20].decode("utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            low = line.lower()
            if any(token in low for token in EVIDENCE_TOKENS):
                hits.append(line.strip()[:200])
                if len(hits) >= limit:
                    return hits
    return hits


def _fast_estimate(run_dir: Path | None, performance: dict) -> dict | None:
    if run_dir is None:
        return None
    from ..feedback import fast_estimate

    block = fast_estimate.for_case(run_dir, performance.get("structural_roofline") or {},
                                   facts_sha256=performance.get("facts_sha256"))
    return block if block.get("status") == "ESTIMATED" else dict(block)


def summarize(case: str, doc: dict) -> dict:
    """One baseline row from a child's result."""
    observed = doc.get("observed") or {}
    perf = observed.get("performance") or {}
    counts = perf.get("feature_counts") or {}
    run_dir = case_run_dir(Path(doc["grade_dir"]), case) if doc.get("grade_dir") else None
    structural = perf.get("structural_roofline") or {}
    bound, bound_by = _roofline(structural)
    cycles = perf.get("estimated_whole_soc_cycles")
    complete = perf.get("status") == "PROVISIONAL" and isinstance(cycles, int) and cycles > 0
    row: dict = {
        "case": case,
        "family": case_family(case),
        "numerics": ((observed.get("tiers") or {}).get("L2") or {}).get("status"),
        "wall_s": doc.get("wall_s"),
        "roofline_cycles": round(bound, 1) if bound else None,
        "bound_by": bound_by,
        "model_review": perf.get("model_review"),
        "model_sha256": perf.get("model_sha256"),
        "elf_sha256": perf.get("elf_sha256"),
        "timing_reason": perf.get("reason") or doc.get("error") or (observed.get("failure") or {}).get("detail"),
    }
    estimate = None if complete else _fast_estimate(run_dir, perf)
    if complete:
        row.update(status="timed", evidence="complete Spike timing model", model_cycles=cycles)
    elif estimate and estimate.get("status") == "ESTIMATED":
        row.update(status="fast_estimate", evidence="fast census estimate (band "
                   f"{estimate.get('low_cycles')}-{estimate.get('high_cycles')})", model_cycles=estimate["cycles"],
                   fast_estimate={k: estimate.get(k) for k in ("cycles", "low_cycles", "high_cycles",
                                                               "lower_bound", "limited_by")})
    else:
        row.update(status="UNKNOWN", evidence=None, model_cycles=None,
                   unknown_reason=(estimate or {}).get("reason") or row["timing_reason"])
    model = row["model_cycles"]
    peak, useful = structural.get("structural_peak_macs_per_cycle"), structural.get("useful_macs")
    row["utilization_pct"] = (round(100.0 * useful / (model * peak), 2)
                              if model and isinstance(peak, (int, float)) and peak and useful else None)
    row["headroom_cycles"] = int(model - bound) if model and bound else None
    stores = _sum_classes(counts, STORE_CLASSES)
    loads = _sum_classes(counts, (LOAD_CLASS,))
    fences = counts.get(FENCE_KEY)
    row["loads"] = loads if counts else None
    row["computes"] = _sum_classes(counts, (COMPUTE_CLASS,)) if counts else None
    row["stores"] = stores if counts else None
    row["fences"] = fences
    row["fences_per_output_tile"] = round(fences / stores, 3) if stores and isinstance(fences, (int, float)) else None
    widths: dict = {}
    if loads and isinstance(counts.get("target.requested_read_bytes"), (int, float)):
        widths["read_bytes_per_load"] = round(counts["target.requested_read_bytes"] / loads, 1)
    if stores and isinstance(counts.get("target.requested_write_bytes"), (int, float)):
        widths["write_bytes_per_store"] = round(counts["target.requested_write_bytes"] / stores, 1)
    for key, value in counts.items():
        if key.startswith(PUBLICATION_PREFIX) and value:
            widths["publication." + key[len(PUBLICATION_PREFIX):]] = value
    row["transfer_widths"] = widths
    fallback = counts.get(PUBLICATION_PREFIX + "aggregate_fallback")
    evidence = _evidence_lines(run_dir)
    row["capacity_fallback"] = (bool(fallback) or bool(evidence)) if counts or evidence else None
    row["refusal_records"] = evidence
    return row


def emission_digest(submission: Path, capsule_mlir: Path, work: Path, grader: Grader,
                    timeout_s: int = 900) -> str | None:
    """Digest of what the compiler emits for one case, or ``None`` when it does not emit.

    The command is the submission's own ``commands.emit_analysis_bundle`` (target artifact on stdout
    plus the command buffer), the same call the grader makes before the export's fixed LLVM lowering,
    so two compilers with equal digests produce the same executable for that case.
    """
    import yaml

    try:
        manifest = yaml.safe_load((submission / "manifest.yaml").read_text(encoding="utf-8"))
        argv = list(manifest["commands"]["emit_analysis_bundle"]["argv"])
        tool = manifest["entrypoints"]["tool"]
    except (OSError, KeyError, TypeError, ValueError):
        return None
    work.mkdir(parents=True, exist_ok=True)
    bundle = work / "command_buffer.json"
    bundle.unlink(missing_ok=True)
    mapping = {"{tool}": str(submission / tool), "{input_mlir}": str(capsule_mlir), "{output_json}": str(bundle)}
    resolved = []
    for arg in argv:
        for key, value in mapping.items():
            arg = arg.replace(key, value)
        resolved.append(grader.python if arg == "python3" else arg)
    env = grader.child_env(work)
    env["PYTHONPATH"] = os.pathsep.join([str(submission / "compiler_library"), env["PYTHONPATH"]])
    try:
        proc = subprocess.run(grader.prefix() + resolved, cwd=submission, env=env, capture_output=True,
                              timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    digest = hashlib.sha256(proc.stdout)
    if bundle.is_file():
        digest.update(bundle.read_bytes())
    return digest.hexdigest()


def run_cases(grader: Grader, submission: Path, cases: list[str], work: Path, workers: int,
              progress=None) -> dict[str, dict]:
    """Time ``cases`` with ``workers`` concurrent children; returns rows by case."""
    rows: dict[str, dict] = {}

    def one(case: str) -> tuple[str, dict]:
        doc = run_child(grader, submission, case, work / case)
        row = summarize(case, doc)
        if progress:
            progress(row)
        return case, row

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for case, row in pool.map(one, cases):
            rows[case] = row
    return rows


def default_python() -> str:
    return sys.executable
