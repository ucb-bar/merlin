"""Verify an author's own before/after builds from their emitted artifacts; never grade or advise.

The bank arm's ``--reference-analysis verify`` reads only owned ``submission/``, ``selfcheck_out/``,
``artifacts/`` and ``variants/`` paths: the ordinary selfcheck outputs (``lowered.llvm.mlir``,
``command_buffer.json``, optional ``package_kernel.elf`` or one author-built ``*.o``) and, when the
author kept them, selfcheck responses. Proved-stream facts come from the ELF, else that object, else
the build's own ``lowered.llvm.mlir`` compiled host-side with ``-mno-relax``. Emitted LLVM is decoded
again by the host's selected ISA decoder (RTL facts plus the contract's operand protocol); instruction
roles come from the same selected roster as ``decoded_effects``. Nothing here edits a compiler, ranks
a setting or extrapolates beyond the listed builds. Each verdict is separate:

* ``numerically_correct``: the selfcheck verdict row of that capsule, when a response is present;
* ``intervention_applied``: ``applied`` only when the emitted stream differs from the before build
  and the declared mechanism's derived effect differs; an identical stream, or an unchanged declared
  effect, is ``optimization_did_not_apply_emitted_fallback``; otherwise ``unknown``. With no declared
  mechanism, a changed stream is ``applied`` only when a stream mechanism measured on both builds' proved
  streams differs (those are listed, none attributed); else ``unknown`` names each one not measured;
* ``traffic_reduced``: proved per-call executions (static sites x proved trip counts) of the
  movement-in roles per accumulate-role execution; any unproved site keeps it ``unknown``.

Storage validity reports the schedule-dialect verifier over emitted schedule records and
lexical local-address window overlaps from ``derive_effects``; neither is dynamic proof.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path, PurePosixPath

SCHEMA = "merlin.candidate_verification.v1"
OWNED = ("submission", "selfcheck_out", "artifacts", "variants")
APPLIED, FALLBACK, UNKNOWN = "applied", "optimization_did_not_apply_emitted_fallback", "unknown"
LOWERED, COMMANDS, ELF = "lowered.llvm.mlir", "command_buffer.json", "package_kernel.elf"
#: An author's own relocatable object of the build (``*.o``, e.g. a ``-mno-relax`` rebuild), and the key
#: recording several candidates when the choice would be ambiguous.
OBJECT, OBJECTS_AMBIGUOUS = "relocatable_object", "relocatable_objects_ambiguous"
_COMPILE_TIMEOUT_S = 300
MOVEMENT_IN = ("operand_load", "weight_load")
#: Movement-in roles that ``decoded_effects`` classifies as memory transfers; the remaining movement-in
#: roles (array register preloads) are counted separately so a preload never reads as data traffic.
DATA_MOVEMENT = ("operand_load",)
COMPUTE = "accumulate"
_MAX_CASES = 128
_MAX_REPLY = 48 << 10
_NOT_ESTABLISHED = ("dynamic_issue_order", "hardware_execution", "performance", "semantic_equivalence",
                    "behaviour_on_unlisted_shapes")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def mechanisms():
    """Accepted mechanism names: exactly the fact names ``derive_effects`` reports."""
    from merlin.kernels.decoded_effects import derive_effects

    return tuple(derive_effects({}, {})["facts"])


# ----------------------------------------------------------------------------- owned inputs
def owned_path(workspace, value):
    """Resolve one workspace-relative owned path; refuse anything else before reading bytes."""
    if not isinstance(value, str) or not value or len(value) > 1024 or "\\" in value:
        raise ValueError("verify_requires_owned_path")
    relative = PurePosixPath(value.rstrip("/"))
    parts = relative.parts
    if (relative.is_absolute() or not parts or any(part in (".", "..") for part in parts)
            or parts[0] not in OWNED or (parts[0] == "variants" and len(parts) < 2)):
        raise ValueError("verify_requires_owned_path")
    workspace = Path(workspace).resolve()
    path = workspace.joinpath(*parts)
    chain = [path, *path.parents]
    if any(p.is_symlink() for p in chain[:len(parts)]) or not path.resolve().is_relative_to(workspace):
        raise ValueError("verify_requires_owned_path")
    if not path.exists():
        raise ValueError("verify_input_missing")
    return path, relative.as_posix()


def _read(path, label):
    from . import reference_analysis as A

    raw, pin = A._read(path, label)
    return raw, pin


def case_dirs(workspace, value):
    """``{capsule: (path, label)}`` for one case directory or a directory of case directories."""
    path, label = owned_path(workspace, value)
    if path.is_file():
        return {path.parent.name: (path, label)}
    if (path / LOWERED).is_file() or (path / COMMANDS).is_file():
        return {path.name: (path, label)}
    cases = {}
    for child in sorted(path.iterdir()):
        if child.is_symlink() or not child.is_dir():
            continue
        if (child / LOWERED).is_file() or (child / COMMANDS).is_file():
            cases[child.name] = (child, label + "/" + child.name)
        if len(cases) > _MAX_CASES:
            raise ValueError("verify_case_bound_exceeded")
    if not cases:
        raise ValueError("verify_no_selfcheck_case_artifacts")
    return cases


def _files(path, label):
    """The build's named artifacts, plus the author's own relocatable object when exactly one is present."""
    if path.is_file():
        name = (LOWERED if path.suffix == ".mlir" else COMMANDS if path.suffix == ".json"
                else OBJECT if path.suffix == ".o" else ELF)
        return {name: (path, label)}
    found = {name: (path / name, label + "/" + name) for name in (LOWERED, COMMANDS, ELF)
             if (path / name).is_file() and not (path / name).is_symlink()}
    objects = sorted(child.name for child in path.iterdir()
                     if child.suffix == ".o" and child.is_file() and not child.is_symlink())
    if len(objects) == 1:
        found[OBJECT] = (path / objects[0], label + "/" + objects[0])
    elif objects:
        found[OBJECTS_AMBIGUOUS] = objects[:8]
    return found


# ----------------------------------------------------------------------------- one build
def decode_lowered(text, decoder):
    """The host decoder's trace of emitted LLVM; ``None`` when no selected decoder exists."""
    if not decoder or decoder.get("semantics") is None:
        return None
    from merlin.targetgen.rocc.decode import _parse_module, decode_module

    module = _parse_module(text)
    if module is None:
        raise ValueError("verify_emitted_ir_unparseable")
    return decode_module(module, target=decoder["target"], semantics=decoder["semantics"])


def traffic(rows, roles):
    """Per-role static sites and proved per-call executions; unknown when any site is unproved."""
    static, dynamic, unproved, unrostered = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        names = tuple((roles or {}).get(row.get("class")) or ())
        if not names:
            unrostered[str(row.get("class"))] += 1
            continue
        count = row.get("executions")
        for role in names:
            static[role] += 1
            if type(count) is int and count >= 0:
                dynamic[role] += count
            else:
                unproved[role] += 1
    per_role = {role: dict(static_sites=static[role],
                           dynamic=dynamic[role] if not unproved[role] else None,
                           dynamic_status="proved" if not unproved[role] else "unknown",
                           unproved_sites=unproved[role])
                for role in sorted(static)}
    return dict(per_role=per_role, unrostered_sites=dict(unrostered),
                count_basis="static IR sites x counted-loop executions per kernel call (None: not proved)",
                ratios=dict(static=_ratio(per_role, "static_sites"), dynamic=_ratio(per_role, "dynamic")))


def _ratio(per_role, key):
    loads = [per_role.get(role, {}).get(key, 0) for role in MOVEMENT_IN]
    compute = per_role.get(COMPUTE, {}).get(key)
    if any(value is None for value in loads) or compute is None:
        return dict(loads_per_compute=None, status="unknown", reason="unproved_execution_count")
    if not compute:
        return dict(loads_per_compute=None, status="unknown", reason="no_accumulate_role_sites")
    by_role = {role: _round(Fraction(value, compute)) for role, value in zip(MOVEMENT_IN, loads)}
    moved = sum(value for role, value in zip(MOVEMENT_IN, loads) if role in DATA_MOVEMENT)
    preloads = sum(value for role, value in zip(MOVEMENT_IN, loads) if role not in DATA_MOVEMENT)
    return dict(loads_per_compute=_round(Fraction(sum(loads), compute)), by_role=by_role,
                data_movement_loads_per_compute=_round(Fraction(moved, compute)),
                preloads_per_compute=_round(Fraction(preloads, compute)),
                data_movement_roles=[r for r in MOVEMENT_IN if r in DATA_MOVEMENT],
                preload_roles=[r for r in MOVEMENT_IN if r not in DATA_MOVEMENT],
                loads=sum(loads), compute=compute, status="observed")


def _round(fraction):
    return round(float(fraction), 6)


def effects(rows, decoder, structure, evidence_id):
    """``derive_effects`` over the decoded IR sites in IR order (lexical, not issue order)."""
    if structure is None:
        return None
    from merlin.kernels.decoded_effects import derive_effects

    commands = [dict(index=i, pc=i, instruction=row["class"], fields=dict(row.get("decoded") or {}),
                     status="decoded" if row.get("class") != "UNKNOWN" else "unknown",
                     operands=dict(rs1=row.get("rs1") or {}, rs2=row.get("rs2") or {}))
                for i, row in enumerate(rows) if (decoder.get("structure") or {}).get("instruction_roles", {}).get(
                    row.get("class")) is not None]
    analysis = dict(mechanism_projection=dict(instruction_classes=dict(Counter(c["instruction"] for c in commands))))
    return derive_effects(analysis, structure, commands=commands, evidence_id=evidence_id)


def receipts(command_buffer):
    """Emitted compiler-lowering records: route selections, refusals, schedule IR (compiler-reported)."""
    out = []
    lowering = ((command_buffer or {}).get("params") or {}).get("compiler_lowering") or {}
    for family, records in sorted(lowering.items()):
        for record in records if isinstance(records, list) else [records]:
            if not isinstance(record, dict):
                continue
            row = dict(family=family, schema=record.get("schema"))
            for key, value in sorted(record.items()):
                if key.startswith("requested_") and isinstance(value, str):
                    suffix = key[len("requested_"):]
                    row.setdefault("routes", []).append(dict(axis=suffix, requested=value,
                                                             selected=record.get("selected_" + suffix)))
                elif key.endswith("refusal") or key.endswith("refusals"):
                    row.setdefault("refusals", []).append(dict(field=key, value=value))
            schedule = record.get("schedule_ir")
            if isinstance(schedule, dict) and isinstance(schedule.get("text"), str):
                row["schedule_ir"] = dict(sha256=schedule.get("sha256"), text=schedule["text"])
            out.append(row)
    return out


def verify_schedule(text, declared_sha256):
    """The schedule dialect's own parser/verifier over one emitted schedule record."""
    actual = hashlib.sha256(text.encode()).hexdigest()
    if declared_sha256 and declared_sha256 != actual:
        return dict(verifier="schedule_dialect.verify", status="fail", reason="schedule_ir_digest_mismatch")
    try:
        from xdsl.utils.exceptions import ParseError, VerifyException

        from merlin.compiler.schedule.schedule_ir import parse_dense_schedule_plan
    except ImportError:
        return dict(verifier="schedule_dialect.verify", status="unknown", reason="verifier_unavailable")
    try:
        parse_dense_schedule_plan(text).verify()
    except (ParseError, VerifyException) as error:
        return dict(verifier="schedule_dialect.verify", status="fail", sha256=actual, reason=str(error)[:300])
    except (ValueError, TypeError, KeyError) as error:
        return dict(verifier="schedule_dialect.verify", status="unknown", sha256=actual, reason=str(error)[:300])
    return dict(verifier="schedule_dialect.verify", status="pass", sha256=actual)


def window_overlaps(effect_facts):
    """Lexical overlaps of operand-load windows of different classes in one declared flag set.

    Two movement-in classes filling overlapping local rows can clobber a live operand; a load
    feeding a compute or readout window is ordinary dataflow and is not reported."""
    fact = (effect_facts or {}).get("local_address_windows") or {}
    if fact.get("status") not in ("observed", "partial"):
        return dict(status="unknown", reason=fact.get("reason") or "local_address_windows_unknown", overlaps=[])
    windows = [w for w in fact["value"]["windows"] if type(w.get("min")) is int and MOVEMENT_IN[0] in w["roles"]]
    overlaps = []
    for i, left in enumerate(windows):
        for right in windows[i + 1:]:
            if left["flags"] != right["flags"] or left["instruction"] == right["instruction"]:
                continue
            ends = [w.get("max_end") or w["max"] + 1 for w in (left, right)]
            if left["min"] < ends[1] and right["min"] < ends[0]:
                overlaps.append(dict(flags=left["flags"], first=dict(instruction=left["instruction"],
                    field=left["field"], roles=left["roles"], range=[left["min"], ends[0]]),
                    second=dict(instruction=right["instruction"], field=right["field"], roles=right["roles"],
                                range=[right["min"], ends[1]])))
    unresolved = fact["value"].get("unresolved_sites") or {}
    status = "overlap_observed" if overlaps else ("no_overlap_observed" if not unresolved else "unknown")
    return dict(status=status, overlaps=overlaps[:16], unresolved_address_sites=unresolved,
                scope=fact.get("scope"), basis="lexical IR windows; sequential reuse is not excluded")


def analyze_build(files, *, decoder=None, structure=None, evidence_id="build", symbol=None, build=None):
    """One build record from its pinned emitted artifacts (all optional; missing stays unknown).

    The proved command stream comes from the first available form of the author's own build: the linked
    ``package_kernel.elf``, else one owned relocatable object in the build directory, else the build's
    own ``lowered.llvm.mlir`` compiled host-side with the selected object recipe and ``-mno-relax``
    (``build``: :func:`object_recipe`). ``symbol`` is the entry the stream is proved from."""
    pins, record = [], dict(evidence_id=evidence_id)
    command_buffer = None
    if COMMANDS in files:
        raw, pin = _read(*files[COMMANDS])
        pins.append(pin)
        try:
            command_buffer = json.loads(raw)
        except ValueError:
            record["command_buffer_status"] = "unparseable"
    for name in (ELF, OBJECT):
        if name in files:
            raw, pin = _read(*files[name])
            pins.append(pin)
            record["elf" if name == ELF else "object"] = pin
    trace, lowered = None, None
    if LOWERED in files:
        raw, pin = _read(*files[LOWERED])
        pins.append(pin)
        record["emitted_sha256"] = pin["sha256"]
        lowered = raw
        trace = decode_lowered(raw.decode("utf-8", "replace"), decoder)
    stream, reason, source = _build_stream(files, lowered, decoder, symbol, build)
    record["stream_source"] = source
    return build_record(trace, decoder=decoder, structure=structure, command_buffer=command_buffer,
                        evidence_id=evidence_id, inputs=pins, extra=record, stream_commands=stream,
                        stream_reason=reason)


def _build_stream(files, lowered, decoder, symbol, build):
    """``(decoded commands, reason, source)`` of the build's first available binary form."""
    from merlin.kernels.mechanism_applicability import NO_STREAM

    if not symbol:
        return None, NO_STREAM + ": no kernel entry symbol (candidate.symbol, or the contract's declared entry)", \
            dict(kind=None, symbol=None)
    for name, kind in ((ELF, "linked_elf"), (OBJECT, "owned_relocatable_object")):
        if name in files:
            stream, reason = _elf_stream(files[name], decoder, symbol)
            return stream, reason, dict(kind=kind, path=files[name][1], symbol=symbol)
    if lowered is None:
        detail = ("several relocatable objects; name one with --after <dir>/<file>.o"
                  if OBJECTS_AMBIGUOUS in files else "no package_kernel.elf, relocatable object or lowered.llvm.mlir")
        return None, f"{NO_STREAM}: {detail} in this build", dict(kind=None, symbol=symbol)
    return _compiled_stream(lowered, decoder, symbol, build)


def object_recipe(document, target=None):
    """Entry symbol and ``-march``/``-mabi`` of the selected contract's kernel build recipe.

    The entry is the contract's declared harness entry; the flags come from the toolchain block that
    declares the kernel stack frame (the grader's object recipe). Missing values stay ``None``."""
    def blocks(node):
        if isinstance(node, dict):
            if isinstance(node.get("kernel_stack_frame"), dict):
                yield node
            for value in node.values():
                yield from blocks(value)
        elif isinstance(node, list):
            for value in node:
                yield from blocks(value)

    document = document or {}
    block = next(blocks(document), {})
    flags = [str(f) for f in block.get("cflags") or () if str(f).startswith(("-march=", "-mabi=", "-mcmodel="))]
    entry = ((document.get("harness_abi") or {}).get("entry_symbol")
             or (block.get("kernel_stack_frame") or {}).get("entry_symbol"))
    return dict(entry_symbol=entry if isinstance(entry, str) and entry else None, flags=flags, target=target,
                basis="selected contract harness_abi.entry_symbol and kernel toolchain cflags")


def _compiled_stream(lowered, decoder, symbol, build):
    """Compile the build's own emitted LLVM dialect host-side (-mno-relax) and step ``symbol`` in it."""
    import subprocess
    import tempfile

    from merlin.kernels.mechanism_applicability import NO_STREAM

    source = dict(kind="host_compiled_lowered_mlir", symbol=symbol, relaxation="-mno-relax")
    flags = list((build or {}).get("flags") or ())
    if not any(f.startswith("-march=") for f in flags) or not any(f.startswith("-mabi=") for f in flags):
        return None, NO_STREAM + ": the selected contract declares no kernel -march/-mabi recipe", source
    try:
        from merlin.llvmlower import codegen, toolchain

        clang = Path(toolchain.clang())
        translate = Path(toolchain.mlir_translate())
        if not translate.is_file():
            translate = clang.with_name("mlir-translate")  # the selected clang's own installation
        selected = tuple(f.partition("=")[0] + "=" for f in flags)
        base = [f for f in codegen.RISCV_FLAGS if not f.startswith(selected)]
        with tempfile.TemporaryDirectory(prefix="cca-verify-build-") as work:
            work = Path(work)
            (work / "kernel.llvm.mlir").write_bytes(lowered)
            argv = [[str(translate), "--mlir-to-llvmir", "kernel.llvm.mlir", "-o", "kernel.ll"],
                    [str(clang), *base, *flags, "-mno-relax", "-c", "kernel.ll", "-o", "kernel.o"]]
            for command in argv:
                done = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=_COMPILE_TIMEOUT_S)
                if done.returncode:
                    return None, (f"{NO_STREAM}: host compile of lowered.llvm.mlir failed ({Path(command[0]).name} "
                                  f"rc {done.returncode}): {done.stderr[-300:]}"), dict(source, argv=command)
            source.update(argv=[[Path(c[0]).name, *c[1:]] for c in argv],
                          object_sha256=hashlib.sha256((work / "kernel.o").read_bytes()).hexdigest())
            stream, reason = _elf_stream((work / "kernel.o", "host_compiled/kernel.o"), decoder, symbol)
            return stream, reason, source
    except (OSError, subprocess.SubprocessError, ImportError) as error:
        return None, f"{NO_STREAM}: host compile unavailable: {type(error).__name__}: {str(error)[:200]}", source


def _elf_stream(entry, decoder, symbol):
    """Decoded commands of a proved bounded execution of ``symbol`` in an ELF or relocatable object."""
    from merlin.kernels.mechanism_applicability import NO_STREAM, command_stream

    if entry is None or not symbol:
        return None, NO_STREAM + (": no package_kernel.elf in this build" if entry is None
                                  else ": no kernel symbol named (candidate.symbol)")
    from merlin.analysis.artifact_reader import PinnedELF
    from merlin.analysis.disassembly import tokenize_text

    from . import reference_analysis as A

    raw, _ = A._read(*entry, allow_elf=True)
    if not isinstance(raw, PinnedELF):
        return None, NO_STREAM + f": {Path(str(entry[1])).name} is not an ELF"
    try:
        if decoder.get("host_isa") is not None:
            raw.declare_isa(decoder["host_isa"])
        rows = tuple(tokenize_text(raw.function_disassembly(symbol)["text"]))
        return command_stream(rows, decoder)
    except (ArithmeticError, LookupError, TypeError, ValueError) as error:
        return None, f"{NO_STREAM}: {type(error).__name__}: {str(error)[:200]}"
    finally:
        raw.close()


def build_record(trace, *, decoder=None, structure=None, command_buffer=None, evidence_id="build", inputs=(),
                 extra=None, stream_commands=None, stream_reason=None):
    """Pure core over an (already decoded) trace and command buffer; used by tests and real inputs.

    ``stream_commands``: the decoded commands of a proved dynamic execution, when one exists; width,
    buffering and overlap facts come only from it and otherwise stay unknown."""
    record = dict(extra or {}, evidence_id=evidence_id, inputs=list(inputs))
    record["stream_facts"] = stream_facts(stream_commands, structure, evidence_id, stream_reason)
    # Identity of the issued commands (class and decoded fields in issue order; code addresses excluded).
    record["proved_stream_sha256"] = (None if stream_commands is None else
                                      _digest([(e.get("class"), e.get("fields")) for e in stream_commands]))
    roles = ((decoder or {}).get("structure") or {}).get("instruction_roles")
    if trace is None:
        record.update(stream=dict(status="unknown", reason="emitted_ir_or_selected_decoder_unavailable"),
                      traffic=None, effects=None)
    else:
        rows = trace["instructions"]
        stream = [(row.get("class"), row.get("decoded"), row.get("executions")) for row in rows]
        record["stream"] = dict(status="decoded", sites=len(rows), sha256=_digest(stream),
                                classes=dict(Counter(row.get("class") for row in rows)))
        record["traffic"] = traffic(rows, roles) if roles else None
        derived = effects(rows, decoder or {}, structure, evidence_id) if roles else None
        record["effects"] = None if derived is None else derived["facts"]
        loops = ((trace.get("summary") or {}).get("counted_loops") or {})
        record["loop_proof"] = dict(basis=loops.get("basis"),
                                    unavailable=[u for f in loops.get("functions") or () for u in f.get("unavailable")
                                                 or ()][:8])
    found = receipts(command_buffer)
    verifiers = [dict(family=r["family"], **verify_schedule(r["schedule_ir"]["text"], r["schedule_ir"]["sha256"]))
                 for r in found if "schedule_ir" in r]
    for r in found:
        r.pop("schedule_ir", None)
    record["receipts"] = found
    record["storage"] = dict(
        library_verifiers=verifiers or [dict(verifier="schedule_dialect.verify", status="not_run",
                                             reason="no emitted schedule record in the command buffer")],
        local_address_windows=window_overlaps(record.get("effects")))
    if command_buffer is not None:
        record["declared_tensors"] = {name: dict(shape=t.get("shape"), dtype=t.get("dtype"), role=t.get("role"))
                                      for name, t in (command_buffer.get("tensors") or {}).items()
                                      if isinstance(t, dict)}
    return record


# ----------------------------------------------------------------------------- comparison
def _storage_status(record):
    verifiers = record["storage"]["library_verifiers"]
    windows = record["storage"]["local_address_windows"]["status"]
    if any(v["status"] == "fail" for v in verifiers):
        return "verifier_failed"
    if windows == "overlap_observed":
        return "operand_window_overlap_observed"
    if all(v["status"] == "pass" for v in verifiers) and windows == "no_overlap_observed":
        return "no_conflict_observed"
    return "unknown"


def compare(before, after, *, mechanism=None, verdicts=None):
    """Separate verdicts for one before/after pair of owned builds."""
    delta = {}
    if before.get("traffic") and after.get("traffic"):
        for role in sorted(set(before["traffic"]["per_role"]) | set(after["traffic"]["per_role"])):
            b, a = before["traffic"]["per_role"].get(role, {}), after["traffic"]["per_role"].get(role, {})
            d = dict(static_before=b.get("static_sites", 0), static_after=a.get("static_sites", 0),
                     dynamic_before=b.get("dynamic", 0) if b else 0, dynamic_after=a.get("dynamic", 0) if a else 0)
            d["static_delta"] = d["static_after"] - d["static_before"]
            known = None not in (d["dynamic_before"], d["dynamic_after"])
            d["dynamic_delta"] = d["dynamic_after"] - d["dynamic_before"] if known else None
            delta[role] = d
    ratios = {}
    for basis in ("static", "dynamic"):
        rb = ((before.get("traffic") or {}).get("ratios") or {}).get(basis) or {}
        ra = ((after.get("traffic") or {}).get("ratios") or {}).get(basis) or {}
        ratios[basis] = dict(before=rb.get("loads_per_compute"), after=ra.get("loads_per_compute"),
                             **{key: dict(before=rb.get(key), after=ra.get(key))
                                for key in ("data_movement_loads_per_compute", "preloads_per_compute")})
    dyn = ratios["dynamic"]
    if None in (dyn["before"], dyn["after"]):
        reduced = dict(status="unknown", reason="dynamic loads per compute not proved on both builds",
                       static_before=ratios["static"]["before"], static_after=ratios["static"]["after"])
    else:
        reduced = dict(status="reduced" if dyn["after"] < dyn["before"] else "not_reduced",
                       before=dyn["before"], after=dyn["after"], basis="proved dynamic loads per compute")
    applied = _applied(before, after, mechanism)
    storage = dict(before=_storage_status(before), after=_storage_status(after),
                   before_detail=before["storage"], after_detail=after["storage"])
    changes = stream_changes(before, after)
    return dict(
        numerically_correct=verdicts or dict(status="unknown", reason="no selfcheck verdict row supplied"),
        intervention_applied=applied, traffic_reduced=reduced, storage_valid=storage,
        traffic_delta=dict(per_role=delta, loads_per_compute=ratios),
        stream_changes=changes, regressions=regressions(before, after, changes),
        route_receipts=dict(before=_routes(before), after=_routes(after)),
        independence=("numerical correctness, intervention application and traffic are separate; "
                      "a correct fallback build is not evidence of optimization transfer"))


def stream_facts(commands, structure, evidence_id, reason=None):
    """Width, buffering and overlap facts of one build's proved command stream (else explicit unknown)."""
    from merlin.kernels.mechanism_applicability import NO_STREAM, STREAM_FACTS, STREAM_SCOPE

    unknown = {name: dict(status="unknown", reason=reason or NO_STREAM, scope=STREAM_SCOPE) for name in STREAM_FACTS}
    if commands is None or structure is None:
        return unknown
    from merlin.kernels.mechanism_applicability import EXTRA_FACTS, stream_effects

    derived = stream_effects({}, structure, commands, evidence_id=evidence_id)
    extra = {name: derived[name] for name, *_ in EXTRA_FACTS if name in derived}  # present only when derived
    return {**{name: derived.get(name) or unknown[name] for name in STREAM_FACTS}, **extra}


def stream_changes(before, after):
    """Per-role width, buffering and overlap of ``after`` relative to ``before`` (factual directions)."""
    from merlin.kernels.mechanism_applicability import _stream_rows

    def record(build):
        return dict(derived_effects=dict(facts=build.get("stream_facts") or {}))

    rows = _stream_rows(record(before), record(after), before.get("evidence_id", "before"),
                        after.get("evidence_id", "after"))
    return [{k: v for k, v in row.items() if k not in ("evidence", "basis")} for row in rows]


def _flag(rows, reason, change="increases"):
    if not rows:
        return dict(status="unknown", reason=reason)
    hit = [dict(role=r.get("role"), before=r.get("reference"), after=r.get("candidate")) for r in rows
           if r["direction"] == "candidate_differs" and r.get("needed_change") == change]
    known = [r for r in rows if r["direction"] != "unknown"]
    if not known:
        return dict(status="unknown", reason=rows[0].get("reason") or reason)
    return dict(status=bool(hit), roles=hit, unknown_roles=[r.get("role") for r in rows if r["direction"] == "unknown"])


#: Flags over the further stream facts: (flag, compared effects, change that would restore ``before``).
_EXTRA_FLAGS = (
    ("issue_rate_regressed", ("host_issue_rate",), "increases"),
    ("reconfiguration_increased", ("reconfiguration", "reconfiguration_in_loop"), "reduces"),
    ("capacity_underused", ("local_capacity_use",), "increases"),
    ("bank_conflict_introduced", ("same_bank_halves",), "reduces"),
    ("contiguity_lost", ("dram_contiguity",), "increases"),
    ("epilogue_moved_off_accelerator", ("epilogue_placement",), "increases"),
)


def regressions(before, after, changes=None):
    """Facts, not advice: traffic per compute rose, overlap fell, or a role's transfers narrowed."""
    changes = stream_changes(before, after) if changes is None else changes
    ratio = [((b.get("traffic") or {}).get("ratios") or {}).get("dynamic") or {} for b in (before, after)]
    traffic = {}
    for key in ("data_movement_loads_per_compute", "preloads_per_compute"):
        b, a = (r.get(key) for r in ratio)
        traffic[key] = (dict(status="unknown", reason="dynamic loads per compute not proved on both builds")
                        if None in (b, a) else dict(status=a > b, before=b, after=a))
    statuses = [v["status"] for v in traffic.values()]
    traffic_flag = dict(status="unknown" if "unknown" in statuses and True not in statuses else True in statuses,
                        **traffic)
    width = [r for r in changes if r["effect"] == "role_transfer_width"]
    overlap = [r for r in changes if r["effect"] == "issue_overlap"]
    return dict(traffic_regressed=traffic_flag,
                overlap_regressed=_flag(overlap, "issue_overlap unknown"),
                width_narrowed=_flag(width, "role_transfer_width unknown"),
                output_group_narrowed=_flag([r for r in changes if r["effect"] == "output_group_extent"],
                                            "output_group_extent unknown"),
                prefetch_coarsened=_flag([r for r in changes if r["effect"] == "prefetch_interleave"],
                                         "prefetch_interleave unknown", change="reduces"),
                prefetch_distance_shortened=_flag([r for r in changes if r["effect"] == "prefetch_distance"],
                                                  "prefetch_distance unknown"),
                compute_ahead_reduced=_flag([r for r in changes if r["effect"] == "prefetch_compute_ahead"],
                                            "prefetch_compute_ahead unknown"),
                **{flag: _flag([r for r in changes if r["effect"] in effects], effects[0] + " not reported",
                               change=change)
                   for flag, effects, change in _EXTRA_FLAGS},
                basis="after relative to the compared build; a regression flag states a fact, not a cause")


def _routes(record):
    return [dict(family=r["family"], schema=r.get("schema"), routes=r.get("routes", []),
                 refusals=r.get("refusals", [])) for r in record.get("receipts", [])
            if r.get("routes") or r.get("refusals")][:16]


def _applied(before, after, mechanism):
    if mechanism and mechanism in _stream_fact_names():
        return _stream_applied(before, after, mechanism)
    sb, sa = before.get("stream") or {}, after.get("stream") or {}
    if sb.get("status") != "decoded" or sa.get("status") != "decoded":
        return dict(status=UNKNOWN, reason="emitted stream not decoded on both builds")
    if sb["sha256"] == sa["sha256"]:
        return dict(status=FALLBACK, reason="emitted_stream_identical_to_before",
                    evidence=dict(stream_sha256=sa["sha256"]))
    if not mechanism:
        return _undeclared_applied(before, after, dict(before=sb["sha256"], after=sa["sha256"]))
    fb = (before.get("effects") or {}).get(mechanism) or {}
    fa = (after.get("effects") or {}).get(mechanism) or {}
    evidence = dict(mechanism=mechanism, basis="effects", before=_brief(fb), after=_brief(fa))
    if fb.get("status") not in ("observed", "partial") or fa.get("status") not in ("observed", "partial"):
        return dict(status=UNKNOWN, reason="declared mechanism effect unknown on a build", evidence=evidence)
    if fb.get("value") == fa.get("value"):
        return dict(status=FALLBACK, reason="declared_mechanism_effect_unchanged", evidence=evidence)
    return dict(status=APPLIED, reason="emitted stream and declared mechanism effect both changed",
                evidence=evidence)


def _stream_applied(before, after, mechanism):
    """A proved-stream mechanism: judged on the builds' proved command streams and that stream fact.

    The static IR site list can be identical while loop structure, and so the issued stream, differs;
    identity is therefore read from the proved streams themselves."""
    pb, pa = before.get("proved_stream_sha256"), after.get("proved_stream_sha256")
    fb = (before.get("stream_facts") or {}).get(mechanism) or {}
    fa = (after.get("stream_facts") or {}).get(mechanism) or {}
    evidence = dict(mechanism=mechanism, basis="stream_facts", before=_brief(fb), after=_brief(fa),
                    proved_stream_sha256=dict(before=pb, after=pa))
    if before.get("emitted_sha256") and before.get("emitted_sha256") == after.get("emitted_sha256"):
        return dict(status=FALLBACK, reason="emitted_code_identical_to_before", evidence=evidence)
    if pb is None or pa is None:
        return dict(status=UNKNOWN, reason="proved command stream not established on both builds", evidence=evidence)
    if pb == pa:
        return dict(status=FALLBACK, reason="proved_command_stream_identical_to_before", evidence=evidence)
    if fb.get("status") not in ("observed", "partial") or fa.get("status") not in ("observed", "partial"):
        return dict(status=UNKNOWN, reason="declared mechanism effect unknown on a build", evidence=evidence)
    if fb.get("value") == fa.get("value"):
        return dict(status=FALLBACK, reason="declared_mechanism_effect_unchanged", evidence=evidence)
    return dict(status=APPLIED, reason="proved command stream and declared mechanism effect both changed",
                evidence=evidence)


#: Stream-comparison directions in which the after build's measure differs from the before build's.
_DIFFERS = ("candidate_differs", "candidate_already_exceeds")


def _undeclared_applied(before, after, emitted):
    """No declared mechanism: judged on every stream mechanism measured on both builds' proved streams.

    The measures are :func:`stream_changes` (before as reference, after as candidate), so a mechanism
    counts as changed only when its own projection differs, not when an incidental count does.
    ``applied`` needs a changed emitted stream and at least one mechanism whose measure differs; no single
    mechanism is attributed. An identical proved stream is a fallback. Otherwise ``unknown`` names each
    mechanism not measured on both builds and why."""
    pb, pa = before.get("proved_stream_sha256"), after.get("proved_stream_sha256")
    evidence = dict(basis="stream mechanisms measured on both builds' proved command streams",
                    emitted_stream_sha256=emitted, proved_stream_sha256=dict(before=pb, after=pa))
    if pb is not None and pb == pa:
        return dict(status=FALLBACK, reason="proved_command_stream_identical_to_before", evidence=evidence)
    rows = {}
    for row in stream_changes(before, after):
        rows.setdefault(row["effect"], []).append(row)
    changed = sorted(e for e, rs in rows.items() if any(r["direction"] in _DIFFERS for r in rs))
    unproved = {e: (next((r.get("reason") for r in rs if r.get("reason")), None) or "not measured")
                for e, rs in rows.items() if e not in changed and any(r["direction"] == "unknown" for r in rs)}
    unchanged = sorted(set(rows) - set(changed) - set(unproved))
    evidence.update(changed_mechanisms=changed, unchanged_mechanisms=unchanged, unmeasured_mechanisms=unproved)
    if changed:
        return dict(status=APPLIED, reason="emitted stream changed and these stream mechanisms differ on the "
                    "proved streams: " + ", ".join(changed) + " (no mechanism declared; none attributed)",
                    evidence=evidence)
    if pb is None or pa is None:
        side = " and ".join(s for s, v in (("before", pb), ("after", pa)) if v is None)
        return dict(status=UNKNOWN, reason=f"emitted stream changed; no mechanism declared and no proved "
                    f"command stream on {side}", evidence=evidence)
    return dict(status=UNKNOWN, reason="emitted stream changed; no mechanism declared and no stream mechanism "
                "measured on both builds differs" + (f"; not measured: {', '.join(sorted(unproved))}"
                                                     if unproved else ""), evidence=evidence)


def _stream_fact_names():
    from merlin.kernels.mechanism_applicability import EXTRA_FACTS, STREAM_FACTS

    return STREAM_FACTS + tuple(name for name, *_ in EXTRA_FACTS)


def _brief(fact):
    if not fact:
        return None
    value = json.dumps(fact.get("value"), sort_keys=True, default=str)
    return dict(status=fact.get("status"), reason=fact.get("reason"),
                value=fact.get("value") if len(value) <= 2048 else dict(sha256=_digest(fact.get("value"))))


# ----------------------------------------------------------------------------- verdict rows
def verdict_rows(workspace, paths):
    """``{capsule: row}`` from owned selfcheck response documents (``per_capsule`` rows)."""
    rows = {}
    for value in paths or ():
        path, label = owned_path(workspace, value)
        raw, pin = _read(path, label)
        document = json.loads(raw)
        for row in document.get("per_capsule") or ():
            if isinstance(row, dict) and isinstance(row.get("capsule"), str):
                rows[row["capsule"]] = dict(source=pin, submission_sha256=document.get("submission_sha256"), row=row)
    return rows


def numerical(row):
    if row is None:
        return dict(status="unknown", reason="no selfcheck verdict row for this capsule")
    value = row["row"]
    executed = value.get("numerical_execution_passed")
    status = ("correct" if executed is True and value.get("numeric_consistency_passed") is not False
              else "incorrect" if executed is False and value.get("pass") is False else "unknown")
    return dict(status=status, certified=value.get("pass"), numerical_execution_passed=executed,
                numeric_consistency_passed=value.get("numeric_consistency_passed"),
                source=value.get("capsule"), record=row["source"], submission_sha256=row["submission_sha256"],
                binding="author-supplied selfcheck response; the build-to-response binding is not proved")


# ----------------------------------------------------------------------------- shapes and capacity
def declared_shape(capsules_root, capsule, record):
    """Shape conditions from the public capsule declaration, else the build's emitted tensors."""
    if capsules_root is not None:
        path = Path(capsules_root) / capsule / "capsule.yaml"
        if path.is_file() and not path.is_symlink():
            import yaml

            document = yaml.safe_load(path.read_text()) or {}
            tensors = {row.get("name"): dict(shape=row.get("shape"), dtype=row.get("dtype"), role=row.get("role"))
                       for row in document.get("inputs") or () if isinstance(row, dict)}
            return dict(basis="public_capsule_declaration", operation=(document.get("operation") or {}).get("op"),
                        tensors=tensors)
    if record.get("declared_tensors"):
        return dict(basis="emitted_command_buffer_tensors", operation=None, tensors=record["declared_tensors"])
    return dict(basis="unknown", operation=None, tensors={})


def public_capsules_root(descriptor):
    """The selected descriptor's public capsule set (declared shapes); None when it cannot be derived."""
    if descriptor is None:
        return None
    try:
        from merlin.targetgen.target_experiment import load_target_experiment
        from merlin_experiments.corpus.admission import public_capsules_for

        return public_capsules_for(load_target_experiment(descriptor))
    except Exception:  # noqa: BLE001 -- declared shapes then stay unknown, never guessed
        return None


def capacities(facts):
    """Local memory capacities as RTL-fact data (name, banks, rows, bytes); empty when absent."""
    if facts is None:
        return []
    document = json.loads(Path(facts).read_bytes())
    spaces = (document.get("facts") or document).get("memory_address_spaces") or []
    return [dict(name=s.get("name"), banks=s.get("banks"), depth=s.get("depth"), bytes=s.get("bytes"))
            for s in spaces if isinstance(s, dict) and type(s.get("bytes")) is int]


def _element_bytes(dtype):
    if not isinstance(dtype, str):
        return None
    digits = "".join(c for c in dtype if c.isdigit())
    return int(digits) // 8 if digits and int(digits) % 8 == 0 else None


def capacity_conditions(shape, spaces):
    rows = []
    for name, tensor in sorted(shape["tensors"].items()):
        size, width = 1, _element_bytes(tensor.get("dtype"))
        for extent in tensor.get("shape") or ():
            size *= extent if type(extent) is int else 0
        footprint = size * width if width and size else None
        rows.append(dict(tensor=name, role=tensor.get("role"), bytes=footprint,
                         exceeds=[s["name"] for s in spaces if footprint is not None and footprint > s["bytes"]]))
    return dict(tensors=rows, spaces=spaces,
                interpretation="whole-tensor footprint vs each RTL local memory; placement is not inferred")


# ----------------------------------------------------------------------------- operation
def verify(workspace, value, *, decoder=None, structure=None, facts=None, capsules_root=None, build=None):
    """The full verify report for one validated request value.

    ``build``: :func:`object_recipe` of the selected contract (declared entry symbol and object flags);
    ``candidate.symbol`` overrides the declared entry."""
    mechanism = value.get("mechanism") or None
    if mechanism is not None and mechanism not in mechanisms():
        raise ValueError("verify_mechanism_not_a_derived_effect")
    before_cases = case_dirs(workspace, value["path"])
    after_cases = case_dirs(workspace, value["candidate"]["path"])
    baseline_cases = case_dirs(workspace, value["baseline"]) if value.get("baseline") else {}
    symbol = value["candidate"].get("symbol") or (build or {}).get("entry_symbol") or None
    symbol_source = ("candidate.symbol" if value["candidate"].get("symbol")
                     else "selected contract declared entry" if symbol else None)
    build_recipe = build
    sweep = {}
    for item in value.get("sweep") or ():
        for capsule, entry in case_dirs(workspace, item).items():
            sweep.setdefault(capsule, entry)
    verdicts = verdict_rows(workspace, value.get("verdicts"))
    spaces = capacities(facts)
    cache = {}

    def build(entry):
        if entry[1] not in cache:
            cache[entry[1]] = analyze_build(_files(*entry), decoder=decoder, structure=structure,
                                            evidence_id=entry[1], symbol=symbol, build=build_recipe)
        return cache[entry[1]]

    pairs = sorted(set(before_cases) & set(after_cases))
    if len(before_cases) == 1 and len(after_cases) == 1 and not pairs:
        pairs = [(next(iter(before_cases)), next(iter(after_cases)))]
    else:
        pairs = [(name, name) for name in pairs]
    if not pairs:
        raise ValueError("verify_before_after_share_no_capsule")
    results = []
    for left, right in pairs:
        b, a = build(before_cases[left]), build(after_cases[right])
        row = compare(b, a, mechanism=mechanism, verdicts=numerical(verdicts.get(right)))
        base = _baseline_entry(baseline_cases, right)
        if base is not None:
            row["against_baseline"] = _against(build(base), a, base[1])
        results.append(dict(capsule=right, before=before_cases[left][1], after=after_cases[right][1],
                            before_inputs=b["inputs"], after_inputs=a["inputs"],
                            fell_back_to_seed_or_before=_identical(a, b, build(base) if base else None), **row))
    report = dict(schema=SCHEMA, kind="candidate_verification", mechanism=mechanism,
                  accepted_mechanisms=list(mechanisms()), pairs=results,
                  entry_symbol=dict(symbol=symbol, source=symbol_source),
                  decoder_inputs=list((decoder or {}).get("decoder_inputs") or []),
                  not_established=list(_NOT_ESTABLISHED), advisory_only=True, qualification_issued=False)
    if sweep:
        report["sweep"] = sweep_table(sweep, before_cases, build, mechanism=mechanism, verdicts=verdicts,
                                      spaces=spaces, capsules_root=capsules_root, baseline_cases=baseline_cases)
    if baseline_cases:
        report["baseline"] = value["baseline"]
    return report


def _baseline_entry(baseline_cases, capsule):
    if capsule in baseline_cases:
        return baseline_cases[capsule]
    return next(iter(baseline_cases.values())) if len(baseline_cases) == 1 else None


def _against(baseline, after, label):
    """Regression flags of ``after`` against the author-supplied seed build of the same capsule."""
    changes = stream_changes(baseline, after)
    return dict(baseline=label, regressions=regressions(baseline, after, changes), stream_changes=changes)


def _identical(after, before, baseline):
    """``fell_back_to_seed_or_before``: the emitted kernel is byte-identical to the before or seed build."""
    digest = after.get("emitted_sha256")
    if digest is None:
        return dict(status="unknown", reason="no emitted lowered.llvm.mlir in this build")
    same = [name for name, other in (("before", before), ("seed_baseline", baseline))
            if other is not None and other.get("emitted_sha256") == digest]
    compared = [name for name, other in (("before", before), ("seed_baseline", baseline))
                if other is not None and other.get("emitted_sha256")]
    if not compared:
        return dict(status="unknown", reason="no before or seed build with emitted code to compare")
    return dict(status=bool(same), identical_to=same, compared=compared, emitted_sha256=digest,
                basis="identical emitted code (lowered.llvm.mlir bytes)")


def sweep_table(sweep, before_cases, build, *, mechanism, verdicts, spaces, capsules_root, baseline_cases=None):
    """Per-shape rows for the frozen after-compiler builds; baselines pair by capsule name."""
    rows = []
    for capsule, entry in sorted(sweep.items()):
        after = build(entry)
        shape = declared_shape(capsules_root, capsule, after)
        row = dict(capsule=capsule, build=entry[1], shape=shape, capacity=capacity_conditions(shape, spaces),
                   numerically_correct=numerical(verdicts.get(capsule)), route_receipts=_routes(after),
                   storage_valid=_storage_status(after))
        if capsule in before_cases:
            pair = compare(build(before_cases[capsule]), after, mechanism=mechanism)
            row.update(baseline=before_cases[capsule][1], intervention_applied=pair["intervention_applied"],
                       traffic_reduced=pair["traffic_reduced"], traffic_delta=pair["traffic_delta"])
        else:
            row.update(baseline=None, traffic=after.get("traffic"),
                       intervention_applied=dict(status=UNKNOWN, reason="no before build of this capsule supplied"),
                       traffic_reduced=dict(status="unknown", reason="no before build of this capsule supplied"))
        base = (baseline_cases or {}).get(capsule)
        seed = build(base) if base is not None else None
        if seed is not None:
            row["against_baseline"] = _against(seed, after, base[1])
        row["fell_back_to_seed_or_before"] = _identical(
            after, build(before_cases[capsule]) if capsule in before_cases else None, seed)
        refusals = [dict(family=r["family"], **f) for r in row["route_receipts"] for f in r.get("refusals", [])]
        if refusals:
            row["blockers"] = dict(legality=refusals)
            row["blocker_source"] = "compiler-emitted route refusal records; capacity from RTL facts"
        over = [t for t in row["capacity"]["tensors"] if t["exceeds"]]
        if over:
            row.setdefault("blockers", {})["capacity"] = over
        rows.append(row)
    region = {APPLIED: [], FALLBACK: [], UNKNOWN: []}
    for row in rows:
        region[row["intervention_applied"]["status"]].append(dict(
            capsule=row["capsule"], operation=row["shape"]["operation"],
            shapes={k: v["shape"] for k, v in row["shape"]["tensors"].items()},
            correct=row["numerically_correct"]["status"], traffic=row["traffic_reduced"]["status"],
            blockers=sorted((row.get("blockers") or {}).keys())))
    return dict(rows=rows, applicability_region=dict(
        applied=region[APPLIED], fallback=region[FALLBACK], unknown=region[UNKNOWN],
        counts={k: len(v) for k, v in region.items()},
        identical_emitted_code=_region_by_dimensions(rows),
        frozen_after_builds=sorted({r["build"] for r in rows}),
        extrapolation="none: only the listed capsules' emitted builds are reported"))


def _dimensions(row):
    """Declared tensor extents of one sweep row as named axes ``<tensor>[i]``; nothing is inferred."""
    out = {}
    for name, tensor in sorted(row["shape"]["tensors"].items()):
        for axis, extent in enumerate(tensor.get("shape") or ()):
            if type(extent) is int:
                out[f"{name}[{axis}]"] = extent
    return out


def _region_by_dimensions(rows):
    """Listed shapes split by identical emitted code, with each declared axis's observed values per side.

    Only the listed capsules are summarized; no boundary between them is inferred."""
    sides = dict(fell_back=[], emitted_new_code=[], unknown=[])
    for row in rows:
        flag = row["fell_back_to_seed_or_before"]["status"]
        side = "fell_back" if flag is True else "emitted_new_code" if flag is False else "unknown"
        sides[side].append(dict(capsule=row["capsule"], dimensions=_dimensions(row),
                                identical_to=row["fell_back_to_seed_or_before"].get("identical_to", [])))
    axes = {}
    for side, items in sides.items():
        for item in items:
            for axis, extent in item["dimensions"].items():
                axes.setdefault(axis, {}).setdefault(side, set()).add(extent)
    return dict(**sides, counts={k: len(v) for k, v in sides.items()},
                observed_values_by_axis={axis: {side: sorted(values) for side, values in by.items()}
                                         for axis, by in sorted(axes.items())},
                extrapolation="none: values are the listed capsules' declared extents, not a derived boundary")


# ----------------------------------------------------------------------------- broker edge
def validate(value):
    """Shape-check one verify request before any bytes are read."""
    allowed = {"operation", "path", "candidate", "mechanism", "sweep", "verdicts", "format", "offset", "limit",
               "symbol", "entry", "member", "baseline"}
    if not isinstance(value, dict) or set(value) - allowed or value.get("operation") != "verify":
        raise ValueError("invalid_analysis_request")
    if value.get("entry") or value.get("symbol") or value.get("member"):
        raise ValueError("verify_requires_owned_path")
    other = value.get("candidate")
    if not isinstance(other, dict) or set(other) - {"path", "format", "symbol", "offset"}:
        raise ValueError("invalid_analysis_request")
    if not isinstance(other.get("symbol", ""), str) or len(other.get("symbol", "")) > 512:
        raise ValueError("invalid_analysis_request")
    for key in ("sweep", "verdicts"):
        items = value.get(key, [])
        if not isinstance(items, list) or len(items) > 32 or any(not isinstance(i, str) for i in items):
            raise ValueError("invalid_analysis_request")
    if not isinstance(value.get("mechanism", ""), str) or len(value.get("mechanism", "")) > 128:
        raise ValueError("invalid_analysis_request")
    if "baseline" in value and (not isinstance(value["baseline"], str) or not value["baseline"]):
        raise ValueError("invalid_analysis_request")


def _selected(contract, facts, target, descriptor):
    from . import reference_analysis as A

    decoder = A._operand_decoder(contract, facts, target, descriptor=descriptor, include_structure=True)
    if not decoder or decoder.get("semantics") is None:
        return decoder, None
    from merlin.kernels.decoded_effects import selected_structure

    return decoder, selected_structure(decoder, _contract_document(contract, descriptor))


def _contract_document(contract, descriptor):
    """The selected target contract, resolved as the operand decoder resolves it (``{}`` when none)."""
    import yaml

    if descriptor is not None:
        from merlin.targetgen.target_experiment import load_target_experiment, selected_experiment_contract

        contract = selected_experiment_contract(load_target_experiment(descriptor), environment=os.environ)
    return (yaml.safe_load(Path(contract).read_bytes()) or {}) if contract is not None else {}


def _raised(flags):
    return sorted(k for k, v in flags.items() if isinstance(v, dict) and v.get("status") is True)


def _summary(report):
    """A bounded reply: verdicts and compact deltas; complete records stay in the detail."""
    def pair(row):
        evidence = row["intervention_applied"].get("evidence") or {}
        effect = {side: (evidence.get(side) or {}).get("status") for side in ("before", "after")
                  if "mechanism" in evidence}
        return dict(capsule=row["capsule"], numerically_correct=row["numerically_correct"]["status"],
                    **(dict(mechanism_effect_status=effect) if effect else {}),
                    intervention_applied={k: row["intervention_applied"].get(k) for k in ("status", "reason")},
                    traffic_reduced=row["traffic_reduced"],
                    storage_valid={k: row["storage_valid"][k] for k in ("before", "after")},
                    loads_per_compute=row["traffic_delta"]["loads_per_compute"],
                    dynamic_delta={role: d["dynamic_delta"] for role, d in row["traffic_delta"]["per_role"].items()},
                    fell_back_to_seed_or_before=row["fell_back_to_seed_or_before"]["status"],
                    regressions={k: v["status"] for k, v in row["regressions"].items() if isinstance(v, dict)},
                    **({"regressions_vs_baseline": {k: v["status"] for k, v in
                                                    row["against_baseline"]["regressions"].items()
                                                    if isinstance(v, dict)}} if "against_baseline" in row else {}))
    out = dict(kind=report["kind"], mechanism=report["mechanism"], pairs=[pair(r) for r in report["pairs"][:8]],
               n_pairs=len(report["pairs"]), not_established=report["not_established"])
    if "sweep" in report:
        region = report["sweep"]["applicability_region"]
        out["sweep"] = dict(counts=region["counts"], extrapolation=region["extrapolation"],
                            identical_emitted_code={k: region["identical_emitted_code"][k] for k in
                                                    ("counts", "observed_values_by_axis")}, rows=[
            dict(capsule=r["capsule"], correct=r["numerically_correct"]["status"],
                 applied=r["intervention_applied"]["status"], traffic=r["traffic_reduced"]["status"],
                 why=r["intervention_applied"].get("reason"),
                 fell_back_to_seed_or_before=r["fell_back_to_seed_or_before"]["status"],
                 **({"regressed_vs_baseline": _raised(r["against_baseline"]["regressions"])}
                    if "against_baseline" in r else {}),
                 blockers=sorted((r.get("blockers") or {}).keys()))
            for r in report["sweep"]["rows"][:48]])
    return out


def bank_decoding(workspace, bundle_manifest, repo, pin):
    """``(contract, facts)`` of the selected CCA bank's own pinned decoder, or ``None`` for older banks.

    Applicability decodes the candidate with the bank's decoder; verify uses the same one so both
    endpoints describe an author's build under one decoder identity."""
    from . import reference_bank

    if pin is None or bundle_manifest is None or repo is None:
        return None
    root, bank = reference_bank.open_bank(Path(workspace), bundle_manifest, repo, pin)
    decoder = reference_bank._bank_decoder(root, bank)
    return None if decoder is None else (decoder["contract"], decoder["facts"])


def handle(value, *, base, workspace, facts=None, contract=None, target=None, descriptor=None,
           compiler_sha256=None, capsules_root=None, bank=None):
    """Serve one verify request: a bounded reply plus a hash-pinned detail file in the workspace.

    ``bank``: ``(contract, facts)`` of the selected bank's pinned decoder (:func:`bank_decoding`); it
    replaces the descriptor's selection for decoding only."""
    from merlin.verification.lean_contract import document_digest

    validate(value)
    workspace = Path(workspace).resolve()
    if bank is not None:
        contract, facts, descriptor = bank[0], bank[1], None
    decoder, structure = _selected(contract, facts, target, descriptor)
    build = object_recipe(_contract_document(contract, descriptor), target)
    report = verify(workspace, value, decoder=decoder, structure=structure, facts=facts,
                    capsules_root=capsules_root, build=build)
    report["decoder_selection"] = "cca_bank_decoder" if bank is not None else "selected_contract"
    identity = document_digest(report)
    full = dict(base, status="analyzed", report=report, report_identity=identity,
                current_compiler_sha256=compiler_sha256)
    from .result_details import compact_row

    detail = compact_row(dict(name="candidate_verification", **full), workspace=workspace)["detail"]
    reply = dict(base, status="analyzed", report=_summary(report), report_identity=identity, detail=detail,
                 report_in_detail=True, current_compiler_sha256=compiler_sha256,
                 next_action="Read the hash-pinned detail for every role, effect, receipt and sweep row.")
    if len(json.dumps(reply, indent=2).encode()) > _MAX_REPLY:
        reply["report"] = dict(kind=report["kind"], projection="read full detail; summary exceeded reply bound")
    return reply


GAPS = {
    "verify_requires_owned_path": "verify reads only your own submission/, selfcheck_out/, artifacts/ or "
        "variants/<name>/ paths: a selfcheck_out/<capsule> case, a selfcheck_out/requests/<id> directory, "
        "or a lowered.llvm.mlir/command_buffer.json file.",
    "verify_input_missing": "The named owned path does not exist.",
    "verify_no_selfcheck_case_artifacts": "No lowered.llvm.mlir or command_buffer.json case directories found.",
    "verify_before_after_share_no_capsule": "Pair one case with one case, or two request directories that "
        "share capsule names.",
    "verify_mechanism_not_a_derived_effect": "Use a derived-effect name listed in accepted_mechanisms.",
    "verify_emitted_ir_unparseable": "The emitted lowered.llvm.mlir does not parse; rebuild it with selfcheck.",
    "verify_case_bound_exceeded": "At most 128 cases per directory.",
    "verify_requires_cca_bank_arm": "verify is served by the CCA bank arm's analysis service only.",
}


def broker_response(request, *, workspace, facts, compiler_sha256, rid, started_ns, bundle_manifest=None,
                    repo=None, contract=None, target=None, descriptor=None, capsules_root=None):
    """The broker-decorated verify reply; failures are explicit gaps, never a pass."""
    import time

    from ..brokers.selfcheck import _decorate_response
    from . import reference_bank

    base = dict(schema="merlin.reference_analysis.v1", all_pass=None, numeric_assessment="not_run",
                official=False, qualification_issued=False, semantic_equivalence="unknown", performance="unknown")
    try:
        pin = reference_bank.selected_pin()
        if pin is None:
            raise ValueError("verify_requires_cca_bank_arm")
        validate(request.get("reference_analysis"))
        body = handle(request.get("reference_analysis"), base=base, workspace=workspace, facts=facts,
                      contract=contract, target=target, descriptor=descriptor, compiler_sha256=compiler_sha256,
                      capsules_root=capsules_root, bank=bank_decoding(workspace, bundle_manifest, repo, pin))
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as error:
        code = str(error) if str(error) in GAPS else ("analysis_dependency_unavailable"
                                                      if isinstance(error, ImportError)
                                                      else "analysis_input_unavailable_invalid_or_unsupported")
        body = dict(base, status="unknown", reason=code,
                    next_action=GAPS.get(code, "Check owned paths, the selected decoder inputs and bounds."),
                    gap=dict(kind="evidence_or_adapter_gap", code=code, retryable=False,
                             detail=str(error)[:300]))
    document = _decorate_response(json.dumps(body), request, rid, compiler_sha256, started_ns=started_ns,
                                  completed_ns=time.time_ns())
    document["official"] = False
    return document
