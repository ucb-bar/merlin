"""Load scalar and RVV host implementation packages independently of compiler drivers.

The existing manifest/knobs schema and schedule payload remain unchanged. Provider metadata
records ownership, not qualification. Explicitly selected providers own schedule feature
realization. Existing unselected callers retain the legacy ``mining.from_strategy`` route
until their provider selection and source accounting migrate together.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.common.record_formats import load_yaml, validate_or_raise
from merlin.compiler.backend.providers import Provider, read_provider

# ---- host --------------------------------------------------------------------------------------

_CFLAGS_ALLOW_PREFIXES = ("-march", "-mabi", "-mcmodel", "-O", "-f")
# The compiler's supported schedule datatypes, shared by optional proposal validation.
DTYPE_STRATEGIES = {"fp32", "int8_w8a8", "bf16_f32acc", "fp16_f32acc"}
# Public compile modes, distinct from manifest strategies and numerical admission.
HOST_PRECISIONS = ("fp32", "int8", "fp16", "fp8")


@dataclass
class HostPackage:
    """An isolated scalar or RVV host code-generation package."""

    name: str  # target identity declared by the package
    run_id: str
    directory: Path
    schedule_text: str  # contents of schedule.mlir -> lower_to_llvm_ir(transform_schedule=)
    cflags: list[str]  # RVV-specific cflags -> build_app(cflags_override=cflags+_CFLAGS_COMMON)
    dtype_strategy: str  # fp32 | int8_w8a8 | bf16_f32acc | fp16_f32acc
    op_match: list[dict[str, Any]] = field(default_factory=list)
    lowering_patterns: list[str] = field(default_factory=list)
    lmul_policy: str = "m1"
    expected_instructions: list[str] = field(default_factory=list)
    # Default-off compiler features this (impr_) fork enables (PASS/HEURISTIC/PATTERN class).
    # Empty for the baseline hand_v0 -> byte-identical lowering. See llvmlower.impr_features.
    compiler_features: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    knobs: dict[str, Any] = field(default_factory=dict)
    backend: str = "rvv"
    codegen_provider: Any = field(default=None, repr=False, compare=False)

    @property
    def provider(self) -> Provider | None:
        """Read-only package role; neither a compiler qualification nor a trust grant."""
        return read_provider(self.directory)

    @property
    def is_int8(self) -> bool:
        return self.dtype_strategy == "int8_w8a8"

    @property
    def is_half_f32acc(self) -> bool:
        """True for the 16-bit-float datapaths (``bf16_f32acc`` / ``fp16_f32acc``).

        DECLARATIVE, not a gate. Unlike ``is_int8`` -- which switches ``_prepare_model_mlir``
        onto a different pass set (``apply_quant``) -- the 16-bit-float rewrite
        (``passes_xdsl.lower_bf16_matmul_f32acc``) runs UNCONDITIONALLY on every model, because
        f16/bf16 matmuls need f32 accumulation for correctness regardless of what any package
        declares. Gating it on this knob would silently break every fp16 model whose package
        happens to say ``fp32``.

        So this property exists for provenance, result records, and ``expected_instructions``
        verification (e16 vtype + ``vfwmacc``) -- it tells you what datapath a package is
        MEANT to exercise, and the runner checks the emitted code actually matches.
        """
        return self.dtype_strategy in ("bf16_f32acc", "fp16_f32acc")


def _check_cflags(cflags: list[str]) -> list[str]:
    """Integrity: a package may only carry compiler flags from an allowlist — the data-not-code
    analogue of the targetgen harness-import ban (a package must not smuggle arbitrary behavior)."""
    bad = [f for f in cflags if not f.startswith(_CFLAGS_ALLOW_PREFIXES)]
    return [f"cflag {f!r} outside allowlist {_CFLAGS_ALLOW_PREFIXES}" for f in bad]


def load_host_package(package_dir: str | Path, *, codegen_provider: Any = None) -> HostPackage:
    """Load a scalar or RVV package, retaining the existing serialized package schema.

    Validates the manifest against the ``rvv_package_manifest`` schema and the cflags against the
    allowlist. Raises ``ValueError`` on a schema/integrity problem (this is K0 of the runner).
    ``codegen_provider`` is an explicitly selected trusted compiler interface,
    retained for shape adaptation; it is not a source receipt or qualification.
    """
    d = Path(package_dir)
    manifest = load_yaml(d / "manifest.yaml")
    validate_or_raise(manifest, "rvv_package_manifest")

    knobs = load_yaml(d / "knobs.yaml")
    backend = knobs.get("backend", "rvv")
    if backend not in {"rvv", "scalar"}:
        raise ValueError(f"host package {d.name} has unknown backend {backend!r}")
    if backend == "scalar":
        if knobs.get("schedule_file") or knobs.get("op_match") or knobs.get("compiler_features"):
            raise ValueError("scalar host package cannot select an RVV schedule, op matcher, or compiler features")
        schedule_text = ""
    else:
        sched_path = d / knobs.get("schedule_file", "schedule.mlir")
        schedule_text = sched_path.read_text(encoding="utf-8")

    cflags = list(knobs.get("cflags", []))
    problems = _check_cflags(cflags)
    if problems:
        raise ValueError(f"rvv package {d.name} integrity failed: {'; '.join(problems)}")
    if backend == "scalar":
        marches = [flag.removeprefix("-march=") for flag in cflags if flag.startswith("-march=")]
        required = split_isa(marches[0])[1] if len(marches) == 1 else set()
        if len(marches) != 1 or any(
            extension == "v" or extension.startswith(("zve", "zvl", "zv")) for extension in required
        ):
            raise ValueError("scalar host package requires exactly one non-vector -march flag")

    return HostPackage(
        name=manifest["target"],
        run_id=manifest.get("run_id", d.name),
        directory=d,
        schedule_text=schedule_text,
        cflags=cflags,
        dtype_strategy=knobs.get("dtype_strategy", "fp32"),
        op_match=list(knobs.get("op_match", [])),
        lowering_patterns=list(knobs.get("lowering_patterns", [])),
        lmul_policy=knobs.get("lmul_policy", "m1"),
        expected_instructions=list(knobs.get("expected_instructions", [])),
        compiler_features=resolve_features(knobs, manifest, codegen_provider=codegen_provider),
        manifest=manifest,
        knobs=knobs,
        backend=backend,
        codegen_provider=codegen_provider,
    )


def resolve_features(knobs: dict, manifest: dict, shapes: Any = (), *, codegen_provider: Any = None) -> list[str]:
    """compiler_features from knobs/manifest, PLUS the v3 micro-kernel tuning point a ``microkernel``
    knob block {MR, NR, KC} resolves to. That knob is how the beam tunes the register block
    continuously via CODE GENERATION (no hand ukernel) — see from_strategy.microkernel_feature.

    ``shapes`` (a sequence of ``kernels.microkernel.ContractionShape``, normally observed from the
    workload by ``kernels.shapes.contraction_shapes``) opts into the SHAPE-AWARE resolution: the
    block in the knob block is then read as an upper BOUND and the target picks the largest blocking
    that masks no parallel dim of any contraction it must cover. Empty (the default) keeps the
    resolution byte-identical to the shape-blind one."""
    feats = list(knobs.get("compiler_features", manifest.get("compiler_features", [])))
    mk = knobs.get("microkernel")
    if mk:
        if codegen_provider is not None:
            resolver = getattr(codegen_provider, "microkernel_features", None)
            if getattr(codegen_provider, "API_VERSION", None) != 1 or not callable(resolver):
                raise ValueError("selected host codegen provider lacks the versioned microkernel policy")
            resolved = resolver(mk, shapes=shapes)
            if not isinstance(resolved, (list, tuple)) or any(
                not isinstance(name, str) or not name for name in resolved
            ):
                raise ValueError("selected host codegen provider returned invalid microkernel features")
        else:
            # Preserve legacy packages and deliberately registered custom targets
            # until their callers select a provider. New OOT callers pass one.
            from merlin.analysis.mining.from_strategy import microkernel_features

            resolved = microkernel_features(mk, target=str(manifest.get("target", "rvv")), shapes=shapes)
        for name in resolved:
            if name not in feats:
                feats.append(name)
    return feats


def default_run(generated_root: str | Path, target: str = "rvv") -> Path:
    """The most-recent run_id directory for ``target`` under the generated-targets root."""
    base = Path(generated_root) / target
    runs = sorted([p for p in base.iterdir() if p.is_dir()]) if base.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no generated runs for {target!r} under {base}")
    return runs[-1]


# ---- isa ---------------------------------------------------------------------------------------
# Conservative RISC-V ISA decomposition for host implementation admission.
#
# This preserves the existing package/ELF admission grammar; it is not a complete ISA
# specification parser. Hardware facts and package qualification remain caller-owned.


def split_isa(value: str) -> tuple[str, set[str]]:
    """Expand the standard base letters for a conservative package/host ISA check."""
    head, *tail = value.lower().split("_")
    xlen = head[:4]
    if xlen not in {"rv32", "rv64"}:
        raise ValueError(f"unrecognized RISC-V ISA {value!r}")
    position = 4
    while position < len(head) and "a" <= head[position] <= "y":
        position += 1
    if position == 4:
        raise ValueError(f"unrecognized RISC-V ISA {value!r}")
    base = set(head[4:position])
    extensions = set(tail)
    if position < len(head):
        suffix = head[position:]
        if (
            not suffix.startswith("z")
            or len(suffix) == 1
            or not all("a" <= letter <= "z" or "0" <= letter <= "9" for letter in suffix[1:])
        ):
            raise ValueError(f"unrecognized RISC-V ISA {value!r}")
        extensions.add(suffix)
    if "g" in base:
        base.remove("g")
        base.update("imafd")
        extensions.update({"zicsr", "zifencei"})
    return xlen, base | extensions


def _dts_name_char(letter: str) -> bool:
    return letter.isalnum() or letter in "#,._+?@-/\\"


def _skip_dts_trivia(source: str, position: int) -> int:
    """Skip whitespace and both DTS comment forms; an open comment consumes EOF."""
    while position < len(source):
        if source[position].isspace():
            position += 1
        elif source.startswith("//", position):
            end = source.find("\n", position + 2)
            position = len(source) if end < 0 else end + 1
        elif source.startswith("/*", position):
            end = source.find("*/", position + 2)
            position = len(source) if end < 0 else end + 2
        else:
            break
    return position


def _dts_quoted_value(source: str, position: int) -> tuple[str | None, int]:
    """Return raw quoted bytes and the next position, skipping escaped quotes."""
    start = position + 1
    position = start
    while position < len(source):
        if source[position] == "\\":
            position += 2
        elif source[position] == '"':
            return source[start:position], position + 1
        else:
            position += 1
    return None, len(source)


def dts_string_values(source: str, property_name: str) -> list[str]:
    """Read complete DTS property tokens with one quoted value, outside comments/strings.

    The selected DTS is separately bound by its board-owned digest. This is a
    lexical reader for the host ISA/CPU declarations, not a full DTS parser.
    """
    if not property_name:
        raise ValueError("DTS property name must not be empty")
    values: list[str] = []
    position = 0
    while position < len(source):
        position = _skip_dts_trivia(source, position)
        if position >= len(source):
            break
        if source[position] == '"':
            _, position = _dts_quoted_value(source, position)
            continue
        if not _dts_name_char(source[position]):
            position += 1
            continue
        start = position
        while position < len(source) and _dts_name_char(source[position]):
            position += 1
        if source[start:position] != property_name:
            continue
        assignment = _skip_dts_trivia(source, position)
        if assignment >= len(source) or source[assignment] != "=":
            continue
        value_start = _skip_dts_trivia(source, assignment + 1)
        if value_start >= len(source) or source[value_start] != '"':
            continue
        value, position = _dts_quoted_value(source, value_start)
        if value:
            values.append(value)
    return values


def require_host_isa(cflags: list[str], host_isa: str) -> None:
    """Refuse a codegen package whose required instructions exceed the selected host ISA."""
    marches = [flag.removeprefix("-march=") for flag in cflags if flag.startswith("-march=")]
    if len(marches) != 1:
        raise ValueError("host package must declare exactly one -march for ISA compatibility")
    package_xlen, required = split_isa(marches[0])
    host_xlen, available = split_isa(host_isa)
    missing = sorted(required - available)
    if package_xlen != host_xlen or missing:
        raise ValueError(
            f"host package -march={marches[0]} is incompatible with selected host ISA {host_isa}: "
            f"missing {missing or ['matching XLEN']}"
        )


def require_host_isa_dts(cflags: list[str], dts: str | Path, *, expected_sha256: str | None = None) -> list[str]:
    """Check every CPU ISA in one DTS, optionally bound to a board-owned digest."""
    source = Path(dts)
    if not source.is_file():
        raise ValueError(f"selected host device tree is unavailable: {source}")
    data = source.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("selected host DTS bytes differ from the board's pinned SHA256")
    isas = dts_string_values(data.decode("utf-8"), "riscv,isa")
    if not isas:
        raise ValueError(f"selected host device tree has no riscv,isa declaration: {source}")
    for isa in isas:
        require_host_isa(cflags, isa)
    return isas


def implementation_functions():
    """Actual host ISA/selected-DTS parser and admission bindings for source custody."""
    return (
        implementation_functions,
        split_isa,
        _dts_name_char,
        _skip_dts_trivia,
        _dts_quoted_value,
        dts_string_values,
        require_host_isa,
        require_host_isa_dts,
    )
