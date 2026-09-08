"""Runner-owned compile + execute of a *package-produced* lowered LLVM/RoCC MLIR.

The contract splits responsibility: the package emits ``lowered.llvm.mlir`` (a module defining a
kernel function under the entry symbol its contract declares, with the kernel ABI in
``mlir_oot_backend_contract.yaml``); the runner owns the harness (which embeds the deterministic leaf
tensors by name + output buffers and prints ``OUT/METRIC/DONE``), the link, and the oracle
invocation. This path is uniform for Python and C++ packages — the only difference is who produced
the MLIR.

**This module names no target and imports none.** ``target`` is a required argument throughout, and
everything target-specific is resolved through it: the harness ABI from the target's contract
(:mod:`.harness_abi`), and the harness renderer, build recipe and oracle from its backend via
:mod:`merlin.runtime.backends.base`. What remains here is orchestration — lower, render, link, run.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping


def llvm_mlir_to_object(lowered_mlir_text: str, workdir: Path, *, target: str | None = None,
                        _build_service=None) -> Path:
    """Lower package-emitted llvm-dialect MLIR to an rv64 object (.o) for ``target``'s own ISA.

    THE MARCH IS THE TARGET'S, NOT A DEFAULT. This object and the runner-owned harness are linked into
    one ELF and executed on one core, so they have to be built for the same instruction set. The
    default here is the vector-capable ``rv64gcv``, while a systolic target's harness recipe declares
    ``rv64gc`` -- and for years that disagreement was invisible, because every kernel on such a target
    was inline-asm accelerator instructions with nothing for the auto-vectorizer to take. The first
    kernel that gives it something (a scalar float program placed on the host lane) was compiled with
    ``vsetivli``/``vle32.v``/``vfadd.vv`` for a core with no vector unit, trapped on the first one, and
    was reported as the submission's kernel faulting at runtime.

    ``target=None`` keeps the previous default, for callers with no target in hand.
    """
    from merlin.llvmlower import codegen
    workdir.mkdir(parents=True, exist_ok=True)
    extra: tuple[str, ...] = ()
    if _build_service is not None:
        from .build_service import BuildOnlyService
        if type(_build_service) is not BuildOnlyService:
            raise ValueError("build-only override requires an exact host service")
        _build_service.verify(target)
        extra = (_build_service.recipe.march(),)
        # This opt-in service consumes finished target LLVM, not tensors or
        # partially lowered programs. Do not invoke an unrelated model importer
        # and its Python environment merely to translate an LLVM module.
        from xdsl.context import Context
        from xdsl.dialects import builtin, llvm
        from xdsl.parser import Parser
        from merlin.llvmlower import toolchain
        context = Context()
        context.load_dialect(builtin.Builtin)
        context.load_dialect(llvm.LLVM)
        module = Parser(context, lowered_mlir_text).parse_module()
        module.verify()
        if any(op.name != "builtin.module" and not op.name.startswith("llvm.") for op in module.walk()):
            raise ValueError("build-only translation requires a complete LLVM/Builtin module")
        source = workdir / "kernel.llvm.mlir"
        source.write_text(lowered_mlir_text, encoding="utf-8")
        translated = subprocess.run([str(toolchain.mlir_translate()), "--mlir-to-llvmir", str(source),
                                     "-o", str(workdir / "kernel.ll")], capture_output=True, text=True)
        if translated.returncode:
            raise _build_service.recipe.error_cls("LLVM translation failed:\n" + translated.stderr[-2000:])
        _build_service.verify(target)
    else:
        from merlin.llvmlower.pipeline import lower_to_llvm_ir
        ll = lower_to_llvm_ir(lowered_mlir_text, workdir=workdir)
        (workdir / "kernel.ll").write_text(ll, encoding="utf-8")
        if target is not None:
            from merlin.runtime.backends import base as _backends
            extra = (_backends.harness_build_recipe(target).march(),)
    return Path(codegen.compile_ll(workdir / "kernel.ll", workdir / "kernel.o", "riscv",
                                   extra_flags=extra))


def _recorded_operands(cb: dict[str, Any]) -> dict[str, list] | None:
    """The operands an independent-float or WHOLE-PROGRAM buffer must run on, or ``None``.

    A capsule graded under a float policy cannot have had its answer recomputed on the integer engine,
    so its golden is the INDEPENDENT one — computed off-device, on the operands the runner attached to
    the buffer as ``canonical_inputs``. The device is only answerable against that golden if it runs
    on those same operands; nothing consumed them on this path, so the harness materialized each leaf
    from its NAME and the device computed the right function of the wrong inputs — a guaranteed
    mismatch, reported as a functional failure of the submission.

    THE FLOAT CONDITION remains load-bearing for ordinary buffers. A per-op integer buffer is graded
    against a golden recomputed from deterministic name-materialized fill, and may carry unrelated
    historical operands (``GS0_matmul_spec`` is the guard). A WHOLE-PROGRAM buffer is different: the
    runner has already replaced/re-keyed ``canonical_inputs`` with the exact semantic stimulus because
    its compiler may rename leaves positionally. Discarding that table rematerializes from ``arg0``
    instead of ``A0`` and guarantees a false mismatch. The explicit ABI discriminator separates that
    exception without changing legacy per-op integer behavior.
    """
    from merlin.runtime.backends import base as _backends
    from merlin.runtime.commandbuffer import declared_output_dtypes
    recorded = cb.get("canonical_inputs") or {}
    tensors = cb.get("tensors") or {}
    if not recorded:
        return None
    dtypes = declared_output_dtypes(cb)
    outputs = [n for n, s in tensors.items() if (s or {}).get("role") == "output"]
    whole_program = ((cb.get("kernel_abi") or {}).get("kind") == "whole_program")
    if (not whole_program
            and (not outputs
                 or not all(_backends.float_format_of(dtypes.get(n, "")) for n in outputs))):
        return None
    return {name: spec["values"] for name, spec in recorded.items()
            if isinstance(spec, dict) and spec.get("values") is not None and name in tensors} or None


def _explicit_prepack_inputs(inputs, authorizations) -> None:
    """Never satisfy a host immutable-payload grant from candidate recorded operands."""
    if authorizations is None:
        return
    if not isinstance(authorizations, Mapping) or not isinstance(inputs, Mapping):
        raise ValueError("host prepack authorization requires explicit logical inputs")
    from merlin.runtime.prepack_authority import HostPrepackAuthorization
    if any(type(grant) is not HostPrepackAuthorization for grant in authorizations.values()):
        raise ValueError("prepack requires an exact host authorization object")
    if set(authorizations) - set(inputs):
        raise ValueError("authorized prepack input is missing from explicit logical inputs")


def _strict_warm_profile(profile, cb=None):
    """Validate the explicit final-profile capability without naming a target."""
    if profile is None:
        return None
    from merlin.perf.warm_profile_harness import require_strict_final_warm_profile
    validated = require_strict_final_warm_profile(profile)
    if cb is not None and (cb.get("kernel_abi") or {}).get("kind") != "whole_program":
        raise ValueError(
            "strict final warm profiling requires an explicit whole-program kernel ABI")
    return validated


def _accepts_keyword(callable_object, name: str) -> bool:
    """Whether a renderer explicitly accepts ``name`` or a generic keyword set."""
    import inspect
    parameters = inspect.signature(callable_object).parameters
    return (name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()))


def link_elf(cb: dict[str, Any], obj: Path, workdir: Path, *, target: str,
             inputs: dict | None = None, prepack_authorizations=None, _compact_caller=None,
             _build_service=None, warm_profile=None) -> Path:
    """Build the runner-owned harness from ``cb`` and link it with the package object -> ELF.

    Orchestration only: the harness TEXT comes from ``target``'s declared harness ABI and the BUILD
    from its declared recipe, both resolved through the backend registry. This module names no target
    and imports no target's module — ``target`` is a required argument precisely so no default can
    reintroduce one.

    ``prepack_authorizations`` is a trusted-host-only capability. It is not read
    from the command buffer, and requires explicitly supplied immutable operands.
    """
    warm_profile = _strict_warm_profile(warm_profile, cb)
    if _build_service is not None:
        from .build_service import BuildOnlyService
        if (type(_build_service) is not BuildOnlyService or _compact_caller is not None
                or prepack_authorizations is not None):
            raise ValueError("build-only service cannot mix caller authority paths")
        _build_service.verify(target)
        recipe = _build_service.recipe
        _render = _build_service.render
    else:
        from merlin.runtime.backends import base as _backends
        recipe = _backends.harness_build_recipe(target)
        _render = _backends.harness_renderer(target)
    # ``inputs`` INJECTS the caller's real operands into the device harness. A renderer written before
    # this parameter existed still works and still materializes from names -- but silently doing that
    # while the reference and simulator use injected data produces a guaranteed three-way mismatch that
    # reads as a functional failure of the TARGET, so an injecting caller is told instead.
    compact_object_sha = None
    if _compact_caller is not None:
        # Only compile_lowered_to_elf's trusted preparation path supplies this
        # object. No serialized candidate ABI facts or fallback inputs enter it.
        if inputs is not None or prepack_authorizations is not None:
            raise ValueError("prepared compact caller cannot be combined with other input sources")
        import hashlib
        compact_object_sha = hashlib.sha256(Path(obj).read_bytes()).hexdigest()
        kwargs = {"target": target, "compact_caller": _compact_caller}
        if warm_profile is not None:
            if not _accepts_keyword(_render, "warm_profile"):
                raise NotImplementedError(
                    "backend compact harness cannot consume a strict warm profile")
            kwargs["warm_profile"] = warm_profile
        harness = _render(cb, **kwargs)
    else:
        _explicit_prepack_inputs(inputs, prepack_authorizations)
    if _compact_caller is None and prepack_authorizations is None and _build_service is None:
        inputs = inputs or _recorded_operands(cb) or None
    if _compact_caller is not None:
        pass
    elif inputs or prepack_authorizations is not None:
        if not _accepts_keyword(_render, "inputs"):
            raise NotImplementedError(
                f"backend for target {target!r} declares a render_harness that cannot take `inputs`, so "
                f"the device would compute on name-materialized operands while the reference and the "
                f"simulator use the injected ones. Add an `inputs` parameter to its render_harness.")
        if prepack_authorizations is not None:
            if not _accepts_keyword(_render, "prepack_authorizations"):
                raise NotImplementedError("backend harness cannot consume host prepack authorization")
            kwargs = {"target": target, "inputs": inputs,
                      "prepack_authorizations": prepack_authorizations}
        else:
            kwargs = {"target": target, "inputs": inputs}
        if warm_profile is not None:
            if not _accepts_keyword(_render, "warm_profile"):
                raise NotImplementedError("backend harness cannot consume a strict warm profile")
            kwargs["warm_profile"] = warm_profile
        harness = _render(cb, **kwargs)
    else:
        kwargs = {"target": target}
        if warm_profile is not None:
            if not _accepts_keyword(_render, "warm_profile"):
                raise NotImplementedError("backend harness cannot consume a strict warm profile")
            kwargs["warm_profile"] = warm_profile
        harness = _render(cb, **kwargs)
    (workdir / "harness.c").write_text(harness, encoding="utf-8")
    # Linker load address DERIVED from the RTL memory map (platform DRAM base), reusing the curated
    # script's proven section layout but replacing its BAKED origin — so the base is a HW fact, not a
    # hardcoded literal in a vendored file.
    from ..runtime_build import derived_link_script
    link_ld = derived_link_script(recipe.load_address, recipe.link_script, Path(workdir))
    from ..elf_lanes import PACKAGE_ELF_NAME
    elf = workdir / PACKAGE_ELF_NAME
    # REPRODUCIBLE BUILD, in two phases. A single compile+link invocation lets the driver name its
    # intermediate objects `ccXXXXXX.o`, and those random names are recorded in the ELF as STT_FILE
    # symbols -- so two builds of byte-identical sources differ (measured: 6 bytes) while producing
    # identical cycles. That defeats content-addressed reuse of a measurement for no reason. Naming
    # each object explicitly makes the artifact a function of its inputs again.
    # ORDER IS PRESERVED EXACTLY. The single-step command linked
    # `harness.c, <kernel obj>, *support_sources`; object order decides placement within a section,
    # so reordering could move code and change cycles. This build changes how each object is NAMED,
    # nothing about which objects are linked or in what order.
    objects: list[Path] = []
    for source in [workdir / "harness.c", obj, *recipe.support_sources]:
        source = Path(source)
        # Assembly counts: the driver assembles a .S through the same temp-named intermediate that
        # a .c goes through, so leaving crt.S to the link step reintroduced the very STT_FILE symbol
        # this two-phase build exists to remove.
        if source.suffix not in (".c", ".S", ".s"):
            objects.append(source)
            continue
        unit = workdir / f"{source.stem}.o"
        step = subprocess.run(recipe.compile_command(source=source, output=unit),
                              capture_output=True, text=True)
        if step.returncode != 0:
            raise recipe.error_cls(f"compile of {source.name} failed:\n{step.stderr[-2000:]}")
        objects.append(unit)
    cmd = recipe.link_command(objects=objects, output=elf, link_script=link_ld)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise recipe.error_cls(f"link failed:\n{proc.stderr[-2000:]}")
    if _compact_caller is not None:
        verify = getattr(_backends.get_backend(target), "verify_compact_caller_link", None)
        if verify is None:
            raise NotImplementedError("target cannot verify linked compact caller allocations")
        verify(cb, _compact_caller, object_path=obj, elf_path=elf, workdir=workdir,
               expected_object_sha256=compact_object_sha)
    if _build_service is not None:
        _build_service.verify(target)
    return elf


def compile_lowered_to_elf(cb: dict[str, Any], lowered_mlir_text: str,
                           workdir: str | Path | None = None, *, target: str,
                           inputs: dict | None = None, prepack_authorizations=None,
                           compact_contract=None, logical_payloads=None,
                           compact_storage_limit_bytes: int = 64 * 1024,
                           _build_service=None, warm_profile=None) -> Path:
    """Full package-lowered-MLIR -> rv64 ELF (object + runner harness + link).

    The result is a pure function of its inputs, so an unchanged capsule is not recompiled: see
    :mod:`merlin.targetgen.build_cache` for the key, and for why reusing a BUILD carries none of the
    risk of reusing a verdict. A restored build reproduces the whole generated directory, not only the
    executable, because the agent reads what is in it. Every failure to establish a key -- an
    unresolvable recipe, a toolchain that will not answer, an unreadable source -- falls through to an
    ordinary build, which is also what ``MERLIN_ELF_BUILD_CACHE=0`` does.

    Measured before this existed: the screen tier spent 4.06 s building per capsule against 0.153 s
    simulating, and re-paid it on every capsule of every grade.

    The optional compact route requires both ``compact_contract`` and exact
    ``logical_payloads`` bytes, forbids legacy/fallback ``inputs``, and bypasses
    this cache entirely. Its target-owned preparation derives ABI facts and
    validates storage/prepack authority; the linker checks actual arena symbols.
    ``compact_storage_limit_bytes`` bounds host format setup, not hardware
    capacity. A successful build is not a numerical or execution verdict.

    ``warm_profile`` is an explicit, cache-free final measurement build.  It
    must be the strict shared contract (one warm invocation, one measured
    invocation, cycles only); the target renderer supplies launch/completion
    hooks and keeps result readback outside the cycle window.
    """
    warm_profile = _strict_warm_profile(warm_profile, cb)
    if _build_service is not None:
        from .build_service import BuildOnlyService
        if (type(_build_service) is not BuildOnlyService or inputs is None
                or prepack_authorizations is not None or compact_contract is not None
                or logical_payloads is not None):
            raise ValueError("build-only service requires explicit inputs and no alternate caller authority")
        _build_service.verify(target)
        work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="oot_build_only_"))
        obj = llvm_mlir_to_object(lowered_mlir_text, work, target=target, _build_service=_build_service)
        kwargs = {"target": target, "inputs": inputs, "_build_service": _build_service}
        if warm_profile is not None:
            kwargs["warm_profile"] = warm_profile
        return link_elf(cb, obj, work, **kwargs)
    from merlin.runtime.backends import base as _backends
    from .. import build_cache as _bc
    from ..elf_lanes import PACKAGE_ELF_NAME
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="oot_compile_"))
    if compact_contract is not None or logical_payloads is not None:
        if compact_contract is None or logical_payloads is None or inputs is not None:
            raise ValueError("compact build requires explicit contract + logical bytes, with no other inputs")
        prepare = getattr(_backends.get_backend(target), "prepare_compact_caller", None)
        if prepare is None:
            raise NotImplementedError("target has no verified compact caller preparation")
        prepared = prepare(cb, compact_contract, logical_payloads, lowered_mlir_text=lowered_mlir_text,
            workdir=work, prepack_authorizations=prepack_authorizations,
            max_storage_bytes=compact_storage_limit_bytes)
        # Never reuse/publish a cached build: the ABI authority and exact logical
        # byte/prepack grants are not part of the legacy ELF cache key.
        obj = llvm_mlir_to_object(lowered_mlir_text, work, target=target)
        kwargs = {"target": target, "_compact_caller": prepared}
        if warm_profile is not None:
            kwargs["warm_profile"] = warm_profile
        return link_elf(cb, obj, work, **kwargs)
    _explicit_prepack_inputs(inputs, prepack_authorizations)
    if prepack_authorizations is not None:
        # Cached builds skip the renderer's exact payload/binding checks. Until the
        # authorization policy is itself part of cache admission, never reuse or
        # publish such a build. The absent-authorization path remains unchanged.
        obj = llvm_mlir_to_object(lowered_mlir_text, work, target=target)
        kwargs = {"target": target, "inputs": inputs,
                  "prepack_authorizations": prepack_authorizations}
        if warm_profile is not None:
            kwargs["warm_profile"] = warm_profile
        return link_elf(cb, obj, work, **kwargs)
    if warm_profile is not None:
        # The profile changes the runner-owned harness but is deliberately not
        # serialized into the command buffer.  Never let the legacy build key
        # reuse/publish a cold or differently instrumented ELF under this opt-in.
        obj = llvm_mlir_to_object(lowered_mlir_text, work, target=target)
        return link_elf(cb, obj, work, target=target, inputs=inputs,
                        warm_profile=warm_profile)
    # Coalesced ONCE. The harness embeds these operands, so a key computed from the caller's argument
    # while the build used the recorded ones would key two different executables the same way.
    inputs = inputs or _recorded_operands(cb) or None
    try:
        key = _bc.build_identity(target=target, lowered_mlir_text=lowered_mlir_text, cb=cb,
                                 inputs=inputs,
                                 recipe=_backends.harness_build_recipe(target))
    except Exception:                    # noqa: BLE001 -- an unkeyable build is an ordinary build
        key = None
    cached = _bc.reuse(work, key, PACKAGE_ELF_NAME)
    if cached is not None:
        return cached
    obj = llvm_mlir_to_object(lowered_mlir_text, work, target=target)
    elf = link_elf(cb, obj, work, target=target, inputs=inputs)
    _bc.store(key, work, PACKAGE_ELF_NAME)
    return elf


def simulator_provenance(backend, simulator: str) -> dict[str, Any] | None:
    """Identify the simulator BINARY an engine ran on: its path, its digest, and its recorded lineage.

    A result that claims a hardware verdict must record which hardware revision it came from. A cert
    record already pins the ELF, the RTL commits and the toolchain — and said nothing at all about the
    prebuilt simulator that produced the numbers, which for GSIM is an out-of-tree build whose bytes are
    the only thing tying the verdict to an elaboration. A stale or mis-provenanced emulator was therefore
    undetectable after the fact, which is the same failure mode the binary provenance stamp was added to
    catch on the ELF side.

    Derived from the backend, not from a table: it is asked for ``<engine>_status()`` (a sentence about
    how the engine resolved) and ``<engine>_path()`` (where its binary is), both DERIVED attribute names,
    so an engine or a backend that does not publish them contributes nothing rather than a fabricated
    entry. Never raises — provenance that cannot be established is recorded as absent.
    """
    from pathlib import Path as _Path
    rec: dict[str, Any] = {"engine": simulator}
    try:
        getter = getattr(backend, f"{simulator}_path", None)
        if callable(getter):
            binary = _Path(str(getter()))
            rec["binary"] = str(binary)
            if binary.is_file():
                from merlin.common import provenance as _prov
                rec["sha256"] = _prov.file_digest(binary)
        status = getattr(backend, f"{simulator}_status", None)
        if callable(status):
            _ok, _why = status()
            rec["resolution"] = str(_why)
    except Exception as exc:  # noqa: BLE001 — unestablished provenance is recorded, never invented
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec if len(rec) > 1 else None


def _counter_observations(console: str, *, target: str, simulator: str, cycles: int | None,
                          oracle: Any) -> "tuple[list[dict] | None, dict | None]":
    """``(timing_observations, timing_capability)`` a bracketed run earned, or ``(None, None)``.

    THE HOP THAT WAS MISSING. The bracket emitter, the console parser, the wire contract and every
    consumer of a per-unit activity vector already existed; nothing turned the readings into the block
    a tier record carries, so a target that counts overlap in HARDWARE still reported
    ``missing: ['at least one activity source']`` and no composition operator, headroom or eta could
    resolve from it.

    ONLY AN RTL ORACLE MAY CARRY ONE. A functional model runs the program correctly without modelling
    the engines, so its counter CSRs describe nothing: measured on one, a 52-cycle window returned
    per-engine busy totals in the THOUSANDS. Those are not imprecise numbers, they are numbers about a
    different machine, and a composition operator derived from them would be a fabrication carrying a
    measurement's provenance. An oracle that states nothing fails closed for the same reason.

    NOTHING IS SWALLOWED. The first version of this guard put an unimported name inside a broad
    ``except``, so EVERY call returned "no capability": the negative cases passed for the wrong reason
    and the positive case silently never fired. Each refusal below is a specific, reachable condition.
    """
    if not isinstance(oracle, Mapping) or oracle.get("derived_from_rtl") is not True:
        return None, None                      # a model's counters describe a different machine
    from merlin.perf import hw_counters, observations as _observations
    readings = hw_counters.parse_counter_output(console)
    if not readings:
        return None, None                      # unbracketed: byte-identical to before
    discovery = hw_counters.counters_for_target(target)
    if discovery.get("status") != "derived":
        return None, None                      # no counter set derived from this target's own header
    measured_schema = hw_counters.parse_counter_schema(console)
    if measured_schema is not None and measured_schema != discovery.get("header_sha256"):
        return None, None                      # the ELF was bracketed against a DIFFERENT schema
    # An ABSENT schema line is UNKNOWN, not a mismatch -- a real bracketed run need not emit one, and
    # refusing on its absence would refuse every such run. What actually binds the readings to this
    # header is the coverage check below: the reading set must contain every combination the header
    # derives, which a run bracketed against a different counter set cannot satisfy.
    header = Path(discovery["header"]).read_text(encoding="utf-8", errors="replace")
    occupancy = hw_counters.derive_occupancy_counters(header)
    required = set(occupancy.by_combination.values())
    if not required or not required <= set(readings):
        return None, None                      # a partial combination set is a lower bound, not a total
    # The KIND of each engine is the TARGET's declaration: a kind cannot be read off a counter name,
    # and a consumer refuses a unit that lacks one. Absent when the backend declares none.
    from merlin.runtime.backends import base as _backends
    _kinds_reader = getattr(_backends.get_backend(target), "counter_engine_kinds", None)
    kinds = _kinds_reader() if callable(_kinds_reader) else None
    block = hw_counters.observations_from_counters(
        readings, occupancy, total_cycles=cycles,
        source=f"hardware combination counters ({discovery['header']})", kind_of=kinds)
    validated = _observations.validate_block(block)
    if validated is None:
        return None, None
    # A refused block still travels as a capability record: "the producer emitted a block we could not
    # believe" is a fact about the instrument, and dropping it hides the instrument rather than the bug.
    return ([dict(o) for o in validated.observations] or None), validated.to_dict()


def run_on_oracle(cb: dict[str, Any], lowered_mlir_text: str, *, simulator: str, target: str,
                  workdir: str | Path | None = None, timeout: int = 600,
                  inputs: dict | None = None) -> dict[str, Any]:
    """Compile the package's lowered MLIR + run on ``simulator``; return outputs/metrics/console.

    ``timing`` splits the work: ``build_s`` (ELF compile/link) and ``sim_active_s`` (the simulator
    subprocess) are *active* time; ``oracle_wait_s`` is queue/FPGA-slot wait (0 for local sims like
    spike/verilator — only VCS/FireSim adapters that route through a queue set it).
    """
    import time
    from merlin.runtime.backends import base as _backends
    backend = _backends.get_backend(target)
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="oot_run_"))
    _t0 = time.perf_counter()
    elf = compile_lowered_to_elf(cb, lowered_mlir_text, work, target=target, inputs=inputs)
    _t1 = time.perf_counter()
    console = backend.run_elf(elf, simulator=simulator, timeout=timeout)
    _t2 = time.perf_counter()
    outputs, raw = backend.parse_output(console)
    # DECODE A FLOAT RESULT THAT CAME BACK AS ITS CONTAINER WORD. `parse_output` yields whatever the
    # console carried; a target whose harness has integer-only formatting prints a float destination
    # buffer's stored PATTERN, so an f32 result arrives as its 32-bit word and a bf16 result as its
    # 16-bit one. That is a lossless hand-back, not a broken one -- but only once the pattern is read
    # back as the value, which needs the dtype the buffer was DECLARED in. Read here, from the command
    # buffer itself (the same declaration the harness sized the buffer from), so the writer and the
    # reader cannot disagree; keyed on that dtype and on nothing target-specific. A no-op for an
    # integer-declared output and for a backend that already prints decimals, so every existing
    # readback is byte-identical.
    from merlin.runtime.commandbuffer import declared_output_dtypes
    outputs = _backends.decode_float_readback(outputs, declared_output_dtypes(cb))
    # WHICH BUILD of the simulator answered — recorded beside the oracle's declared kind, not inferred
    # afterwards. The tier record identifies the ELF, the RTL pins and the tools, and identified the one
    # remaining input to the verdict not at all: the prebuilt simulator binary. Derived, never assumed:
    # the backend is asked where its ``<engine>_path()`` is and the bytes there are digested. A backend
    # that does not expose one contributes nothing rather than a guess.
    _oracle = dict(backend.ORACLE[simulator])
    _prov = simulator_provenance(backend, simulator)
    if _prov:
        _oracle["provenance"] = _prov
    result = {"outputs": outputs, "raw_metrics": raw, "cycles": raw.get("cycles", 0),
              "oracle": _oracle, "elf": str(elf), "console": console,
              "timing": {"build_s": round(_t1 - _t0, 3), "sim_active_s": round(_t2 - _t1, 3),
                         "oracle_wait_s": 0.0}}
    # Counter markers are a target-independent wire protocol.  The event names/codes remain the
    # target's own: this boundary merely preserves readings the runner already paid to collect.  If
    # they exactly cover a structurally derived joint-occupancy block, compute eta; otherwise retain
    # the raw named readings without guessing what they mean.
    from merlin.perf import hw_counters
    readings = hw_counters.parse_counter_output(console)
    if readings:
        discovery = hw_counters.counters_for_target(target)
        measured_schema = hw_counters.parse_counter_schema(console)
        report: dict[str, Any] = {"status": "measured", "readings": readings,
                                  "discovery": discovery,
                                  "measured_header_sha256": measured_schema}
        if (discovery.get("status") == "derived"
                and measured_schema == discovery.get("header_sha256")):
            header = Path(discovery["header"]).read_text(encoding="utf-8", errors="replace")
            occupancy = hw_counters.derive_occupancy_counters(header)
            required = set(occupancy.by_combination.values())
            if required and required <= set(readings):
                report["occupancy"] = occupancy.to_dict()
                partition_reader = getattr(backend, "counter_partition_inputs", None)
                partition = partition_reader() if callable(partition_reader) else {
                    "status": "unknown",
                    "why": "the target backend exposes no CIRCT counter-partition artifact",
                }
                if partition.get("status") == "available":
                    report["overlap"] = hw_counters.eta_from_counters(
                        readings, occupancy, hw_text=partition["hw_text"],
                        codes=hw_counters.event_codes(header), module=partition["module"],
                        counter_module=partition["counter_module"],
                        measurement_cycles=raw.get("cycles"), source=partition["source"])
                else:
                    report["overlap"] = {
                        "state": "unknown", "eta": None,
                        "why": partition.get("why", "CIRCT counter-partition proof is unavailable"),
                    }
        elif discovery.get("status") == "derived":
            report["status"] = "unknown"
            report["overlap"] = {
                "state": "unknown", "eta": None,
                "why": "the measured ELF counter-schema digest does not match current discovery",
            }
        result["counters"] = report
    _obs, _cap = _counter_observations(console, target=target, simulator=simulator,
                                       cycles=raw.get("cycles"), oracle=_oracle)
    if _cap is not None:
        result["timing_observations"] = _obs
        result["timing_capability"] = _cap
    return result
