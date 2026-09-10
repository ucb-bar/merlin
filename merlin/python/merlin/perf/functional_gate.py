"""Execute a candidate's whole-model emission and grade it with the model's own gate.

The phase-2 authoring loop scores a candidate by static counters (host operations, payload bytes).
Static counters cannot tell a -40% win from a miscompile: every performance lever accepted by hand
so far was made SAFE by one further step -- build the whole-model ELF from the candidate's emitted
artifact, run it on an ISA simulator, and read the harness's own ``MERLIN_RESULT`` lines against
the model's expected values. This module is that step, as a callable the loop can run per
iteration.

Everything target- and model-specific is a PARAMETER supplied by the caller:

* the toolchain (paths to ``mlir-translate``/``clang``/the simulator, the clang target triple,
  ``-march``/``-mabi``/code model, the simulator's ISA string and optional extension) --
  derived by the caller from the target descriptor or an explicit config, never assumed here;
* the model payload directory, whose OWN shipped link script (``build_elf.sh`` beside
  ``payload/`` and ``runtime/``) links the harness, constant blob and goldens -- the link recipe
  belongs to the payload, not to this module;
* the gate specification: which ``MERLIN_RESULT`` fields must hold which values. Equality is the
  bit-exact default; a bounded gate (``max``/``min``) is declared in the same place without
  touching this code.

Three outcomes, kept distinct because conflating them is how a loop lies to itself:

``passed``   the simulator ran, printed the result lines, every gated field matched and the
             harness printed its PASS verdict;
``failed``   the simulator ran and the evidence says the candidate is wrong (a gated field
             mismatched, the harness printed FAIL, or it never reached a verdict), or the
             candidate's own artifact would not translate, compile or link;
``not_run``  no evidence either way -- toolchain absent, payload absent, the compiler declined
             the program, the transcript carries no result line, or the simulator timed out.
             ``not_run`` is NEVER success; it is surfaced so a loop that silently stopped
             executing candidates is visible as such.

The transcript is parsed structurally (line prefix, whitespace split, ``key=value`` partition);
no regular expressions, per the repository rule.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESULT_SCHEMA = "merlin_functional_gate_result_v1"
CONFIG_SCHEMA = "merlin_functional_gate_config_v1"
RESULT_PREFIX = "MERLIN_RESULT"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_NOT_RUN = "not_run"
STATUSES = (STATUS_PASSED, STATUS_FAILED, STATUS_NOT_RUN)
_PASS_TOKENS = frozenset({"PASS"})
_FAIL_TOKENS = frozenset({"FAIL", "FAILED"})
_ARTIFACT_FILE = "model.llvm.mlir"
_OBJECT_FILE = "kernel.o"
_LOG_TAIL_BYTES = 4000


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-_LOG_TAIL_BYTES:].decode("utf-8", errors="replace")


# --------------------------------------------------------------------------------------------
# Transcript parsing
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedTranscript:
    """The ``MERLIN_RESULT`` fields and the harness verdict read out of a simulator transcript."""

    fields: dict[str, str]
    result_lines: tuple[str, ...]
    verdict: str | None                 # "PASS", "FAIL" or None when the harness printed neither
    verdict_line: str | None
    duplicate_fields: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"fields": dict(self.fields), "result_lines": list(self.result_lines),
                "verdict": self.verdict, "verdict_line": self.verdict_line,
                "duplicate_fields": {k: list(v) for k, v in self.duplicate_fields.items()}}


def parse_simulator_transcript(text: str) -> ParsedTranscript:
    """Read every ``MERLIN_RESULT key=value ...`` line and the PASS/FAIL verdict, structurally.

    A result line is one whose first whitespace token is exactly ``MERLIN_RESULT``; each further
    token is split on its first ``=``. Tokens without ``=`` are ignored. A field printed twice is
    kept at its FIRST value and every value is recorded under ``duplicate_fields`` so a transcript
    that contradicts itself is visible rather than silently last-wins.

    The verdict is the harness's own: a line whose first token (after stripping ``*`` and ``:``)
    is ``PASS``, or whose first two tokens include ``FAIL``/``FAILED`` (the riscv-tests
    ``*** FAILED *** (tohost = N)`` spelling). A FAIL anywhere dominates a PASS.
    """
    fields: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    result_lines: list[str] = []
    pass_line: str | None = None
    fail_line: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        if tokens[0] == RESULT_PREFIX:
            result_lines.append(line)
            for token in tokens[1:]:
                key, sep, value = token.partition("=")
                if not sep or not key:
                    continue
                if key in fields:
                    duplicates.setdefault(key, [fields[key]]).append(value)
                else:
                    fields[key] = value
            continue
        head = [token.strip("*:") for token in tokens[:2]]
        if any(token in _FAIL_TOKENS for token in head):
            if fail_line is None:
                fail_line = line
        elif head and head[0] in _PASS_TOKENS and pass_line is None:
            pass_line = line
    if fail_line is not None:
        verdict, verdict_line = "FAIL", fail_line
    elif pass_line is not None:
        verdict, verdict_line = "PASS", pass_line
    else:
        verdict, verdict_line = None, None
    return ParsedTranscript(fields=fields, result_lines=tuple(result_lines), verdict=verdict,
                            verdict_line=verdict_line, duplicate_fields=duplicates)


# --------------------------------------------------------------------------------------------
# Gate specification and evaluation
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FunctionalGateSpec:
    """Which result fields must hold which values.

    ``expectations`` maps a ``MERLIN_RESULT`` field name to either a scalar (equality: an ``int``
    is compared as a decimal integer, a ``str`` case-insensitively as text -- so a hex checksum is
    a string and a count is an int) or a mapping with any of ``equals``, ``max``, ``min`` and an
    optional ``radix`` (default 10) for parsing the actual value. Every named field is REQUIRED
    to be present; the gate cannot pass on a field it never saw.
    """

    expectations: Mapping[str, Any]
    require_pass_line: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.expectations, Mapping) or not self.expectations:
            raise ValueError("a functional gate must name at least one MERLIN_RESULT field")
        for name, expected in self.expectations.items():
            if not isinstance(name, str) or not name:
                raise ValueError("gate field names must be non-empty strings")
            if isinstance(expected, bool):
                raise ValueError(f"gate field {name!r}: booleans are not result values")
            if isinstance(expected, Mapping):
                if not set(expected) & {"equals", "max", "min"}:
                    raise ValueError(f"gate field {name!r}: bound needs equals/max/min")
                if set(expected) - {"equals", "max", "min", "radix"}:
                    raise ValueError(f"gate field {name!r}: unknown bound keys")
            elif not isinstance(expected, (int, str)):
                raise ValueError(f"gate field {name!r}: expected an int, str or bound mapping")

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> "FunctionalGateSpec":
        if not isinstance(document, Mapping):
            raise ValueError("gate specification must be a mapping")
        expectations = document.get("expectations", document.get("expect"))
        if expectations is None:
            # A bare mapping of field -> value is the compact spelling.
            expectations = {k: v for k, v in document.items() if k != "require_pass_line"}
        return cls(expectations=dict(expectations),
                   require_pass_line=bool(document.get("require_pass_line", True)))

    def to_dict(self) -> dict[str, Any]:
        return {"expectations": {k: (dict(v) if isinstance(v, Mapping) else v)
                                 for k, v in self.expectations.items()},
                "require_pass_line": self.require_pass_line}


def _parse_int(text: str, radix: int) -> int | None:
    try:
        return int(text.strip(), radix)
    except (TypeError, ValueError):
        return None


def _compare_field(name: str, expected: Any, actual: str | None) -> dict[str, Any]:
    row: dict[str, Any] = {"field": name, "expected": (dict(expected) if isinstance(expected, Mapping)
                                                        else expected), "actual": actual}
    if actual is None:
        row.update(ok=False, present=False, reason="field absent from the MERLIN_RESULT lines")
        return row
    row["present"] = True
    if isinstance(expected, Mapping):
        radix = int(expected.get("radix", 10))
        equals = expected.get("equals")
        if isinstance(equals, str):
            ok = actual.strip().lower() == equals.strip().lower()
            row.update(ok=ok, reason=None if ok else "text mismatch")
            return row
        value = _parse_int(actual, radix)
        if value is None:
            row.update(ok=False, reason=f"actual value is not an integer in radix {radix}")
            return row
        problems = []
        if equals is not None and value != int(equals):
            problems.append(f"{value} != {int(equals)}")
        if "max" in expected and value > int(expected["max"]):
            problems.append(f"{value} > max {int(expected['max'])}")
        if "min" in expected and value < int(expected["min"]):
            problems.append(f"{value} < min {int(expected['min'])}")
        row.update(ok=not problems, reason="; ".join(problems) or None)
        return row
    if isinstance(expected, int):
        value = _parse_int(actual, 10)
        if value is None:
            row.update(ok=False, reason="actual value is not a decimal integer")
            return row
        ok = value == expected
        row.update(ok=ok, reason=None if ok else f"{value} != {expected}")
        return row
    ok = actual.strip().lower() == str(expected).strip().lower()
    row.update(ok=ok, reason=None if ok else "text mismatch")
    return row


def evaluate_transcript(parsed: ParsedTranscript, gate_spec: FunctionalGateSpec,
                        ) -> tuple[str, str, list[dict[str, Any]]]:
    """``(status, reason, comparisons)`` for one parsed transcript against the gate.

    Order of judgement: no result line at all is ``not_run`` (nothing was measured); a FAIL verdict
    or a mismatched field is ``failed``; a gated field that is absent while the harness still
    printed PASS is ``not_run`` (the gate names a field this harness does not print -- a
    configuration defect, not a candidate one); an absent field with NO verdict is ``failed``
    (the program stopped mid-report); everything matching plus PASS is ``passed``; everything
    matching without a PASS verdict is ``failed`` when the gate requires the verdict.
    """
    if not parsed.result_lines:
        return STATUS_NOT_RUN, f"transcript carries no {RESULT_PREFIX} line", []
    comparisons = [_compare_field(name, expected, parsed.fields.get(name))
                   for name, expected in gate_spec.expectations.items()]
    mismatched = [row for row in comparisons if row["present"] and not row["ok"]]
    absent = [row["field"] for row in comparisons if not row["present"]]
    if parsed.verdict == "FAIL":
        detail = "; ".join(f"{row['field']}: {row['reason']}" for row in mismatched)
        return STATUS_FAILED, ("harness verdict FAIL: " + parsed.verdict_line
                               + (f" ({detail})" if detail else "")), comparisons
    if mismatched:
        detail = "; ".join(f"{row['field']}={row['actual']} ({row['reason']})" for row in mismatched)
        return STATUS_FAILED, "gated field mismatch: " + detail, comparisons
    if absent:
        if parsed.verdict == "PASS":
            return STATUS_NOT_RUN, ("gate names field(s) the harness never printed: "
                                    + ", ".join(absent)), comparisons
        return STATUS_FAILED, ("program stopped before reporting: " + ", ".join(absent)
                               + " (no verdict line)"), comparisons
    if parsed.verdict == "PASS":
        return STATUS_PASSED, "every gated field matched and the harness printed PASS", comparisons
    if gate_spec.require_pass_line:
        return STATUS_FAILED, ("every gated field matched but the harness printed no PASS verdict"
                               " (the program did not complete)"), comparisons
    return STATUS_PASSED, "every gated field matched (PASS verdict not required)", comparisons


# --------------------------------------------------------------------------------------------
# Toolchain and result records
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FunctionalGateToolchain:
    """Every executable and ISA string the build+simulate recipe needs, supplied by the caller."""

    mlir_translate: Path
    clang: Path
    simulator: Path
    clang_target: str
    march: str
    mabi: str
    simulator_isa: str
    code_model: str = "medany"
    simulator_extension: str | None = None
    link_script: Path | None = None            # default: <model_payload_dir>/build_elf.sh
    link_script_args: tuple[str, ...] = ()
    clang_flags: tuple[str, ...] = ("-O2", "-ffunction-sections")
    simulator_args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)

    _REQUIRED = ("mlir_translate", "clang", "simulator", "clang_target", "march", "mabi",
                 "simulator_isa")

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> "FunctionalGateToolchain":
        if not isinstance(document, Mapping):
            raise ValueError("toolchain must be a mapping")
        missing = [name for name in cls._REQUIRED if not document.get(name)]
        if missing:
            raise ValueError("toolchain is missing: " + ", ".join(missing))
        link_script = document.get("link_script")
        return cls(
            mlir_translate=Path(document["mlir_translate"]), clang=Path(document["clang"]),
            simulator=Path(document["simulator"]), clang_target=str(document["clang_target"]),
            march=str(document["march"]), mabi=str(document["mabi"]),
            simulator_isa=str(document["simulator_isa"]),
            code_model=str(document.get("code_model", "medany")),
            simulator_extension=(str(document["simulator_extension"])
                                 if document.get("simulator_extension") else None),
            link_script=Path(link_script) if link_script else None,
            link_script_args=tuple(str(item) for item in document.get("link_script_args", ())),
            clang_flags=tuple(str(item) for item in document.get(
                "clang_flags", ("-O2", "-ffunction-sections"))),
            simulator_args=tuple(str(item) for item in document.get("simulator_args", ())),
            env={str(k): str(v) for k, v in (document.get("env") or {}).items()})

    def to_dict(self) -> dict[str, Any]:
        return {"mlir_translate": str(self.mlir_translate), "clang": str(self.clang),
                "simulator": str(self.simulator), "clang_target": self.clang_target,
                "march": self.march, "mabi": self.mabi, "code_model": self.code_model,
                "simulator_isa": self.simulator_isa,
                "simulator_extension": self.simulator_extension,
                "link_script": str(self.link_script) if self.link_script else None,
                "link_script_args": list(self.link_script_args),
                "clang_flags": list(self.clang_flags),
                "simulator_args": list(self.simulator_args), "env": dict(self.env)}

    def missing_executables(self) -> list[str]:
        return [f"{name}={path}" for name, path in (
            ("mlir_translate", self.mlir_translate), ("clang", self.clang),
            ("simulator", self.simulator)) if not (path.is_file() and os.access(path, os.X_OK))]


@dataclass
class FunctionalGateResult:
    """What one gate run established, including exactly what it compared against."""

    status: str
    reason: str
    stage: str                                  # preflight|translate|compile|link|simulate|evaluate
    fields: dict[str, str] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    verdict: str | None = None
    verdict_line: str | None = None
    simulation_executed: bool = False
    simulator_returncode: int | None = None
    elapsed_seconds: float = 0.0
    artifact_sha256: str | None = None
    elf_sha256: str | None = None
    transcript_path: str | None = None
    transcript_sha256: str | None = None
    workdir: str | None = None
    toolchain: dict[str, Any] = field(default_factory=dict)
    model_payload_dir: str | None = None
    executable_emission: dict[str, Any] = field(default_factory=dict)
    log_tail: str | None = None
    duplicate_fields: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"functional gate status must be one of {STATUSES}")

    @property
    def passed(self) -> bool:
        return self.status == STATUS_PASSED

    @property
    def failed(self) -> bool:
        return self.status == STATUS_FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA, "status": self.status, "reason": self.reason,
            "stage": self.stage, "fields": dict(self.fields), "expected": dict(self.expected),
            "comparisons": list(self.comparisons), "verdict": self.verdict,
            "verdict_line": self.verdict_line, "simulation_executed": self.simulation_executed,
            "simulator_returncode": self.simulator_returncode,
            "elapsed_seconds": self.elapsed_seconds, "artifact_sha256": self.artifact_sha256,
            "elf_sha256": self.elf_sha256, "transcript_path": self.transcript_path,
            "transcript_sha256": self.transcript_sha256, "workdir": self.workdir,
            "toolchain": dict(self.toolchain), "model_payload_dir": self.model_payload_dir,
            "executable_emission": dict(self.executable_emission), "log_tail": self.log_tail,
            "duplicate_fields": {k: list(v) for k, v in self.duplicate_fields.items()},
            # Read this, not `status`, when deciding whether a candidate may be selected: a gate
            # that could not run is not a pass, and a gate that was never configured is not one.
            "excludes_candidate": self.status == STATUS_FAILED,
        }


@dataclass(frozen=True)
class FunctionalGateConfig:
    """The launch-time description of one gate: payload, toolchain, expectations, budget."""

    model_payload_dir: Path
    toolchain: FunctionalGateToolchain
    gate_spec: FunctionalGateSpec
    timeout_seconds: float
    keep_elf: bool = False
    source_path: Path | None = None
    source_sha256: str | None = None

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any], *, source_path: Path | None = None,
                     source_sha256: str | None = None) -> "FunctionalGateConfig":
        if not isinstance(document, Mapping):
            raise ValueError("functional gate config must be a mapping")
        if document.get("schema", CONFIG_SCHEMA) != CONFIG_SCHEMA:
            raise ValueError(f"functional gate config schema must be {CONFIG_SCHEMA}")
        for name in ("model_payload_dir", "toolchain", "gate"):
            if name not in document:
                raise ValueError(f"functional gate config lacks {name!r}")
        timeout = document.get("timeout_seconds", 600)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("functional gate timeout_seconds must be a positive number")
        return cls(model_payload_dir=Path(document["model_payload_dir"]),
                   toolchain=FunctionalGateToolchain.from_mapping(document["toolchain"]),
                   gate_spec=FunctionalGateSpec.from_mapping(document["gate"]),
                   timeout_seconds=float(timeout), keep_elf=bool(document.get("keep_elf", False)),
                   source_path=source_path, source_sha256=source_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CONFIG_SCHEMA, "model_payload_dir": str(self.model_payload_dir),
                "toolchain": self.toolchain.to_dict(), "gate": self.gate_spec.to_dict(),
                "timeout_seconds": self.timeout_seconds, "keep_elf": self.keep_elf,
                "source_path": str(self.source_path) if self.source_path else None,
                "source_sha256": self.source_sha256}


def load_functional_gate_config(path: Path) -> FunctionalGateConfig:
    """Read a ``--functional-gate`` JSON file; relative paths resolve against the file's dir."""
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    base = path.resolve().parent

    def absolute(value: Any) -> Any:
        return str(base / value) if isinstance(value, str) and value and not Path(value).is_absolute() else value

    document = dict(document)
    document["model_payload_dir"] = absolute(document.get("model_payload_dir"))
    toolchain = dict(document.get("toolchain") or {})
    for key in ("mlir_translate", "clang", "simulator", "link_script"):
        if key in toolchain:
            toolchain[key] = absolute(toolchain[key])
    document["toolchain"] = toolchain
    return FunctionalGateConfig.from_mapping(
        document, source_path=path.resolve(), source_sha256=hashlib.sha256(raw).hexdigest())


# --------------------------------------------------------------------------------------------
# The gate itself
# --------------------------------------------------------------------------------------------

def _payload_root(model_payload_dir: Path) -> Path:
    """Accept the bundle root or its ``payload/`` child; return the root holding ``payload/``."""
    root = Path(model_payload_dir)
    if root.name == "payload" and not (root / "payload").is_dir() and (root.parent / "payload").is_dir():
        return root.parent
    return root


def _run(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout: float,
         stdout: Path, stderr: Path) -> subprocess.CompletedProcess[bytes]:
    with stdout.open("wb") as out, stderr.open("wb") as err:
        return subprocess.run(list(command), cwd=str(cwd), env=dict(env), stdout=out, stderr=err,
                              timeout=timeout, check=False)


def run_functional_gate(artifact_text: str, command_buffer: Mapping[str, Any] | None, *,
                        model_payload_dir: Path, toolchain: FunctionalGateToolchain,
                        gate_spec: FunctionalGateSpec, workdir: Path, timeout: float,
                        keep_elf: bool = False) -> FunctionalGateResult:
    """Build the whole-model ELF from ``artifact_text`` and grade one simulator run.

    Recipe (the same one the hand validation used): ``mlir-translate --mlir-to-llvmir`` on the
    emitted artifact; ``clang --target=<triple> -march -mabi -mcmodel -c`` to ``compiler/kernel.o``;
    the payload's own link script, invoked from a copy of the payload dir with ``payload/`` and
    ``runtime/`` beside it; then the simulator with ``--isa=<isa>`` and, when declared,
    ``--extension=<ext>`` on the single ``*.elf`` the link produced. The whole recipe shares one
    ``timeout`` budget; running out of it at any stage is ``not_run``.
    """
    started = time.monotonic()
    deadline = started + float(timeout)
    workdir = Path(workdir)
    root = _payload_root(model_payload_dir)
    artifact_sha256 = _sha256_text(artifact_text)
    base = dict(status=STATUS_NOT_RUN, expected=gate_spec.to_dict(),
                artifact_sha256=artifact_sha256, workdir=str(workdir),
                toolchain=toolchain.to_dict(), model_payload_dir=str(root))

    def finish(**fields: Any) -> FunctionalGateResult:
        fields.setdefault("elapsed_seconds", round(time.monotonic() - started, 3))
        return FunctionalGateResult(**{**base, **fields})

    def remaining() -> float:
        return deadline - time.monotonic()

    # --- preflight: everything the recipe needs must exist before anything is spent ----------
    if isinstance(command_buffer, Mapping) and command_buffer.get("declined"):
        declined = command_buffer["declined"]
        why = str(declined.get("reason") or "") if isinstance(declined, Mapping) else ""
        return finish(stage="preflight", reason="the compiler declined this program"
                      + (f": {why[:400]}" if why else ""))
    if not artifact_text.strip():
        return finish(stage="preflight", reason="the emitted artifact is empty")
    missing = toolchain.missing_executables()
    if missing:
        return finish(stage="preflight", reason="toolchain executable(s) absent: " + ", ".join(missing))
    link_script = toolchain.link_script or (root / "build_elf.sh")
    absent = [str(path) for path in (root / "payload", link_script) if not path.exists()]
    if absent:
        return finish(stage="preflight", reason="model payload absent: " + ", ".join(absent))
    if not os.access(link_script, os.X_OK):
        return finish(stage="preflight", reason=f"link script is not executable: {link_script}")
    emission: dict[str, Any] = {}
    if isinstance(command_buffer, Mapping):
        try:
            from merlin.targetgen.bundle_harness import is_executable_emission
            ok, why_not = is_executable_emission(command_buffer, artifact_text=artifact_text)
            emission = {"ok": ok, "why_not": why_not or None, "advisory": True}
        except Exception as exc:  # the emission check is advisory; the run itself is the evidence
            emission = {"ok": None, "why_not": f"{type(exc).__name__}: {exc}", "advisory": True}
    base["executable_emission"] = emission

    # --- stage the payload copy and the artifact ---------------------------------------------
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    for child in ("payload", "runtime"):
        source = root / child
        if source.is_dir():
            shutil.copytree(source, workdir / child)
    staged_link = workdir / link_script.name
    shutil.copy2(link_script, staged_link)
    compiler = workdir / "compiler"
    compiler.mkdir()
    artifact = compiler / _ARTIFACT_FILE
    artifact.write_text(artifact_text, encoding="utf-8")
    env = {**os.environ, **toolchain.env}

    # --- translate ---------------------------------------------------------------------------
    if remaining() <= 0:
        return finish(stage="translate", reason="budget exhausted before translation")
    llvm_ir = compiler / "kernel.ll"
    try:
        proc = _run([str(toolchain.mlir_translate), "--mlir-to-llvmir", str(artifact), "-o",
                     str(llvm_ir)], cwd=workdir, env=env, timeout=remaining(),
                    stdout=workdir / "translate.out", stderr=workdir / "translate.err")
    except subprocess.TimeoutExpired:
        return finish(stage="translate", reason="mlir-translate timed out")
    if proc.returncode != 0 or not llvm_ir.is_file():
        return finish(status=STATUS_FAILED, stage="translate",
                      reason=f"mlir-translate exited {proc.returncode} on the emitted artifact",
                      log_tail=_tail(workdir / "translate.err"))

    # --- compile -----------------------------------------------------------------------------
    if remaining() <= 0:
        return finish(stage="compile", reason="budget exhausted before compilation")
    kernel_o = compiler / _OBJECT_FILE
    try:
        proc = _run([str(toolchain.clang), f"--target={toolchain.clang_target}",
                     f"-march={toolchain.march}", f"-mabi={toolchain.mabi}",
                     f"-mcmodel={toolchain.code_model}", *toolchain.clang_flags, "-c",
                     str(llvm_ir), "-o", str(kernel_o)], cwd=workdir, env=env,
                    timeout=remaining(), stdout=workdir / "compile.out",
                    stderr=workdir / "compile.err")
    except subprocess.TimeoutExpired:
        return finish(stage="compile", reason="clang timed out")
    if proc.returncode != 0 or not kernel_o.is_file():
        return finish(status=STATUS_FAILED, stage="compile",
                      reason=f"clang exited {proc.returncode} on the translated artifact",
                      log_tail=_tail(workdir / "compile.err"))

    # --- link (the payload's own recipe) -----------------------------------------------------
    if remaining() <= 0:
        return finish(stage="link", reason="budget exhausted before linking")
    try:
        proc = _run([str(staged_link), *toolchain.link_script_args], cwd=workdir, env=env,
                    timeout=remaining(), stdout=workdir / "link.out", stderr=workdir / "link.err")
    except subprocess.TimeoutExpired:
        return finish(stage="link", reason="link script timed out")
    elfs = sorted(path for path in workdir.glob("*.elf") if path.is_file())
    if proc.returncode == 2:
        # The shipped link scripts exit 2 from their own preflight (usage / a required file
        # missing): the environment, not the candidate, is what is incomplete.
        return finish(stage="link", reason="link script preflight refused (exit 2): "
                      "a required toolchain or payload file is absent",
                      log_tail=_tail(workdir / "link.err"))
    if proc.returncode != 0 or len(elfs) != 1:
        return finish(status=STATUS_FAILED, stage="link",
                      reason=(f"link script exited {proc.returncode} and produced {len(elfs)} ELF(s); "
                              "an emission that does not link cannot be accepted"),
                      log_tail=_tail(workdir / "link.err"))
    elf = elfs[0]
    base["elf_sha256"] = _sha256_file(elf)

    # --- simulate ----------------------------------------------------------------------------
    if remaining() <= 0:
        return finish(stage="simulate", reason="budget exhausted before simulation")
    transcript = workdir / "simulate.log"
    command = [str(toolchain.simulator), f"--isa={toolchain.simulator_isa}"]
    if toolchain.simulator_extension:
        command.append(f"--extension={toolchain.simulator_extension}")
    command.extend(toolchain.simulator_args)
    command.append(str(elf))
    try:
        proc = _run(command, cwd=workdir, env=env, timeout=remaining(), stdout=transcript,
                    stderr=workdir / "simulate.err")
    except subprocess.TimeoutExpired:
        partial = parse_simulator_transcript(transcript.read_text(encoding="utf-8", errors="replace")
                                             if transcript.exists() else "")
        return finish(stage="simulate", reason=f"simulator exceeded the {timeout:g}s budget",
                      fields=partial.fields, transcript_path=str(transcript),
                      log_tail=_tail(workdir / "simulate.err"))
    finally:
        if not keep_elf and elf.exists():
            elf.unlink()
    text = transcript.read_text(encoding="utf-8", errors="replace")
    parsed = parse_simulator_transcript(text)
    status, reason, comparisons = evaluate_transcript(parsed, gate_spec)
    if status == STATUS_NOT_RUN and proc.returncode != 0:
        reason += f" (simulator exited {proc.returncode})"
    return finish(status=status, stage="evaluate", reason=reason, fields=parsed.fields,
                  comparisons=comparisons, verdict=parsed.verdict,
                  verdict_line=parsed.verdict_line, simulation_executed=True,
                  simulator_returncode=proc.returncode, transcript_path=str(transcript),
                  transcript_sha256=_sha256_file(transcript),
                  duplicate_fields=parsed.duplicate_fields,
                  log_tail=_tail(workdir / "simulate.err") if proc.returncode != 0 else None)


def gate_result_for_selection(result: Mapping[str, Any] | None) -> tuple[bool, str | None]:
    """``(excluded, reason)`` -- the one question a candidate ranking asks of a gate record.

    Only a ``failed`` gate excludes. ``not_run`` and an absent record leave the candidate eligible,
    because neither is evidence about its numerics; callers surface them separately.
    """
    if not isinstance(result, Mapping):
        return False, None
    if result.get("status") == STATUS_FAILED:
        return True, str(result.get("reason") or "functional gate failed")
    return False, None
