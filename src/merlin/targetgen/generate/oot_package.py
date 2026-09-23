"""The ONE shape a generated out-of-tree target package has — emitted by both generators.

There are two places in this repo that materialise a target package, and before this module they
emitted different things: :func:`merlin.targetgen.capability_manifests.write_oot_target` (the zero-env
path ``target_registry.resolve`` finds, and what ``merlin-onboard`` runs) wrote contracts and nothing
else, while :func:`merlin.targetgen.pipeline.build` wrote an MLIR/Zephyr/xDSL repo whose runtime layer
was an ``adapter.py`` holding a class nothing registers. Either way the result RESOLVED and could never
EXECUTE: no ``plugin`` block was ever written, so ``get_backend(<generated target>)`` raised ``KeyError``
for every target this repo has ever generated.

So the package shape lives HERE, once, and both generators call it. A convergence with two
implementations is not a convergence; it is the same hole with a second entrance.

WHAT A GENERATED PACKAGE CLAIMS, AND WHAT IT REFUSES TO CLAIM
------------------------------------------------------------
``backend``: PROVIDED, in the sense the registry means — the package ships ``backend/__init__.py``,
which self-registers via ``runtime.backends.base.register`` and exposes the module-level functions of
the ``Backend`` protocol, so ``get_backend(<target>)`` returns a real module instead of raising. What
it does NOT do is execute: every execution entry declines by name. A generated package has no toolchain
and no device, and the one shortcut available — running the buffer through Merlin's own reference
runtime and returning those numbers as the target's — would attribute a result to the wrong machine.
It registers only when the target's class can be DERIVED from its declared compute engines; when
``target_class_for`` returns ``None`` the module RAISES at import with that as its stated reason, which
``_load_oot_backend`` catches and records in ``load_failures()``. An undeclared-silicon target therefore
becomes a NAMED ABSENCE rather than a target quietly filed as a CPU.

``compiler``: ABSENT, deliberately, and the manifest says so. Emitting a working out-of-tree compiler
is phase 1's entire job; a generator that could do it would make the thesis unfalsifiable. What is
emitted is the WORK ORDER — a manifest naming the four commands phase 1 must fill in, and a tool that,
if invoked anyway, writes a well-formed command buffer carrying ``declined.reason`` and ``declined.op``.
The two alternatives are both dishonest: exiting 0 with an empty ``commands`` list is ``SILENT_NO_WORK``
(244 such buffers were measured across ``out/runs`` on 2026-09-07 — each one read downstream as a
program that ran and produced nothing), and a non-zero exit is indistinguishable from a broken tool.

NO OPCODE IS EVER A LITERAL HERE
--------------------------------
The retired ``generate/runtime_adapter.py`` baked ``RES_PACK``/``MATMUL_RESIDENT``/``COMMIT``/``EVICT``
into the adapter source, again into ``semantics.py``, again into ``command_encoding.yaml``, plus a
``DEFAULT_REQUANT_SHIFT = 4``. Those are facts about one family of machine written into every machine's
package. Here the map is DERIVED from the target's own manifest ``encoding`` block (which is itself
derived from RTL facts by ``capability_manifests.derive_manifest``) and written to
``backend/command_encoding.yaml``; the backend reads that file at RUN TIME with **no fallback literal**.
When nothing can be derived the file carries ``opcodes: {}`` and a ``derivation: UNKNOWN(<reason>)``
string, and the backend declines instead of guessing.
"""

from __future__ import annotations

from typing import Any

#: Where the backend module lives inside a generated package, and what ``plugin.backend`` names.
BACKEND_DIR = "backend"
#: The derived opcode map, read at run time by the generated backend.
ENCODING_FILE = "command_encoding.yaml"
#: The experiment-ABI work order phase 1 fills in.
COMPILER_MANIFEST = "manifest.yaml"
#: The declining tool the work order points at.
COMPILER_TOOL = "tools/compile.py"

#: The four commands the frozen experiment ABI requires (``mlir_oot_backend_contract.yaml``). Listed
#: so the emitted work order names every one of them; the runner validates against the schema.
ABI_COMMANDS: tuple[str, ...] = (
    "parse",
    "lower_interface_to_target",
    "emit_command_buffer",
    "lower_target_to_llvm",
)

#: Token that marks a value nobody could derive. Fail closed, and say which fact was missing.
UNKNOWN = "UNKNOWN"


def plugin_block() -> dict[str, str]:
    """The ``plugin`` block a generated contract carries. One key, pointing at what is actually shipped."""
    return {"backend": BACKEND_DIR}


# --------------------------------------------------------------------------- derived opcode map
def command_encoding(target: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """``backend/command_encoding.yaml`` for ``target``, derived from its own manifest.

    The source of truth is the manifest's ``encoding.semantic_class`` — the instruction-identity map
    ``derive_manifest`` extracts from the target's RTL facts. A target whose facts ground no such map
    gets ``opcodes: {}`` and an ``UNKNOWN(...)`` derivation naming why, never a borrowed default.
    """
    encoding = manifest.get("encoding")
    semantic = (encoding or {}).get("semantic_class") if isinstance(encoding, dict) else None
    if isinstance(semantic, dict) and semantic:
        # Keys are the target's own instruction codes; values its mnemonics. Both are parsed as DATA
        # from the facts-derived manifest -- nothing here knows what any particular code means.
        opcodes = {str(code): str(mnemonic) for code, mnemonic in sorted(semantic.items(), key=lambda kv: str(kv[0]))}
        derivation = (
            f"contracts/target_contract.yaml:encoding.semantic_class ({len(opcodes)} entries, "
            f"itself derived from this target's RTL facts)"
        )
    else:
        opcodes = {}
        derivation = (
            f"{UNKNOWN}(the derived manifest for {target!r} carries no encoding.semantic_class: no RTL "
            f"fact source grounded this target's instruction identities. The backend DECLINES rather "
            f"than substituting another target's opcode names.)"
        )
    return {
        "target": target,
        "opcodes": opcodes,
        "derivation": derivation,
        "notes": (
            "Command-buffer opcode -> this target's encoded instruction identity. Read at RUN TIME by "
            "backend/__init__.py, which carries no fallback literal: an empty map makes the backend "
            "unavailable and its reason quotable."
        ),
    }


# --------------------------------------------------------------------------- the backend module
#: ``@TARGET@`` is substituted by :func:`backend_module_source`. Plain substitution rather than
#: ``str.format`` so the generated source can contain ordinary braces without doubling every one of
#: them -- the doubling is exactly how the retired adapter template became unreadable.
_BACKEND_PY = '''"""Generated runtime backend for `@TARGET@` — the package's own, self-registering.

GENERATED by `merlin.targetgen.generate.oot_package`. Do not hand-edit: regenerate the package.

This module is loaded by `merlin.runtime.backends.base._load_oot_backend` because this package's
`contracts/target_contract.yaml` declares `plugin.backend: backend`. Importing it runs the
`register(...)` below, after which `base.get_backend("@TARGET@")` returns THIS module.

It exposes the module-level functions of the `Backend` protocol (`available`,
`compile_command_buffer`, `run_elf`, `parse_output`, `run_command_buffer`) — the shape every other
backend in the repo has. It is NOT a class with a bespoke `lower/encode/run_simulator` surface; that
shape was the reason the generated runtime layer could never be reached through the registry.

WHAT IT IS, AND WHAT IT REFUSES TO BE. This is a REGISTRATION and a DECLARATION, not an execution
engine. Every execution entry DECLINES, by name, saying what is missing. That is deliberate twice over:

  * a generated package has no toolchain, no device and no simulator of its own, so there is nothing
    to run — building that is the out-of-tree compiler's job, and `manifest.yaml` beside this file is
    the work order naming the four commands phase 1 must fill in;
  * the tempting shortcut — running the command buffer through Merlin's OWN reference runtime and
    returning the result — would report numbers computed by the reference oracle under this target's
    name. A result attributed to the wrong device is worse than no result, because it gets cited. The
    retired `runtime_adapter.py` took that shortcut, which is why it is on the sandbox's
    oracle-callable deny list to this day.

The opcode map comes from `command_encoding.yaml` beside this file, read at run time with no fallback
literal: when that file carries an `UNKNOWN(...)` derivation, nothing grounded this target's
instruction identities and the decline says which fact was missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from merlin.runtime.backends.base import (
    BackendInfo,
    BackendKind,
    parse_console,
    register,
    target_class_for,
)

TARGET = "@TARGET@"

_ENCODING_PATH = Path(__file__).resolve().parent / "command_encoding.yaml"


def _encoding() -> dict:
    """The derived opcode map + its provenance. Never a literal; absent is an answer."""
    if not _ENCODING_PATH.is_file():
        return {"opcodes": {}, "derivation": f"UNKNOWN(no {_ENCODING_PATH.name} in this package)"}
    doc = yaml.safe_load(_ENCODING_PATH.read_text(encoding="utf-8")) or {}
    return doc if isinstance(doc, dict) else {"opcodes": {}, "derivation": "UNKNOWN(not a mapping)"}


def opcodes() -> dict:
    """Command-buffer opcode -> this target's encoded identity, as DERIVED for this package."""
    got = _encoding().get("opcodes")
    return dict(got) if isinstance(got, dict) else {}


def derivation() -> str:
    """Where the opcode map came from, or `UNKNOWN(<reason>)` when nothing grounded it."""
    return str(_encoding().get("derivation") or "UNKNOWN(no derivation recorded)")


def why_unavailable() -> str:
    """Why this package cannot execute — the sentence every decline below carries.

    Both halves are named, because they are fixed by different work: the compiler is phase 1's job,
    and the encoding is a question for the target's RTL facts.
    """
    parts = [
        f"{TARGET}: this generated package registers a backend but ships no execution path — no "
        f"toolchain, no device, no simulator of its own. Writing one is the out-of-tree compiler's "
        f"job; manifest.yaml beside this file is the work order naming the four commands it must "
        f"implement."
    ]
    if not opcodes():
        parts.append(f"Its command encoding is also underived: {derivation()}")
    return " ".join(parts)


def _decline(op: str | None = None) -> dict:
    """The declined answer: a stated refusal, never an empty program that reads as zeros."""
    out = {"reason": why_unavailable()}
    if op:
        out["op"] = op
    return out


def available() -> bool:
    """False, and `why_unavailable()` says why.

    Deliberately NOT keyed on whether the opcode map was derived. A derived encoding says what this
    target's instructions ARE; it does not give this package anything to run them on, and returning
    True on that basis would advertise an execution path that does not exist.
    """
    return False


def compile_command_buffer(cb: dict[str, Any], workdir: Any, **kw: Any):
    """DECLINES: there is no toolchain here, so there is no ELF.

    Raising is correct rather than returning something: every caller of this function goes on to run
    what it returns, and a stand-in would be run as though it were a program.
    """
    raise NotImplementedError(why_unavailable())


def run_elf(elf: Any, **kw: Any) -> str:
    """DECLINES, for the same reason `compile_command_buffer` does."""
    raise NotImplementedError(why_unavailable())


def parse_output(text: str) -> tuple[dict, dict]:
    """The shared OUT/METRIC/DONE console protocol — the one shared parser, not a copy.

    Implemented rather than declined because it is the one part of the protocol that is real here: it
    parses whatever console text a caller already has, and it does not claim to have produced it.
    """
    return parse_console(text)


def run_command_buffer(cb: dict[str, Any], **kw: Any) -> dict:
    """DECLINES, and says what is missing.

    A decline is a legitimate, reportable answer (see `oot_runner.BackendDeclined`): it is not a pass,
    and it is not wrong arithmetic. What it must never be is an empty result a grader reads as a
    program that ran and produced zeros — nor a number that Merlin's own reference oracle computed,
    handed back under this target's name.
    """
    return {"outputs": {}, "metrics": {}, "declined": _decline(op=str(cb.get("target") or TARGET))}


# --- registration ----------------------------------------------------------------------------------
# The class is DERIVED from the engines this target's contract declares. `target_class_for` returns
# None when it declares none, and None is a real answer: nobody has said what silicon this is. Guessing
# CPU there is how a result gets attributed to the wrong kind of device, so this RAISES instead — which
# `base._load_oot_backend` catches and records in `load_failures()`, making the target a named absence
# rather than a silent misfiling.
_CLASS = target_class_for(TARGET)
if _CLASS is None:
    raise RuntimeError(
        f"{TARGET}: no target class could be derived — its contract declares no compute engines "
        f"(compute_units), so nothing has said whether this is a CPU, a GPU or an NPU. Refusing to "
        f"register a backend under a guessed class; declare compute_units in the target contract."
    )

register(BackendInfo(TARGET, _CLASS, BackendKind.KERNEL, __name__))
'''


def backend_module_source(target: str) -> str:
    """The generated ``backend/__init__.py`` for ``target``."""
    return _BACKEND_PY.replace("@TARGET@", target)


# --------------------------------------------------------------------------- the compiler work order
_COMPILER_TOOL_PY = '''#!/usr/bin/env python3
"""Generated compiler entrypoint for `@TARGET@` — a WORK ORDER that declines honestly.

GENERATED by `merlin.targetgen.generate.oot_package`. This is NOT a compiler. Emitting one is phase 1's
entire job; a generator that could emit it would make the thesis it is testing unfalsifiable.

Why it exists at all, rather than the manifest simply pointing at nothing: the package's declared
`compiler` capability is ABSENT, so `oot_runner` refuses before invoking anything. If something invokes
this tool DIRECTLY anyway, it must still answer in the contract's own vocabulary. The two things it
could do instead are both dishonest:

  * exit 0 having written a buffer with an empty `commands` list — `SILENT_NO_WORK`, which downstream
    reads as a program that ran and computed nothing (244 such buffers measured across out/runs on
    2026-09-07);
  * exit non-zero — indistinguishable from a tool that is simply broken.

So it writes a well-formed command buffer carrying `declined.reason` and `declined.op`, and exits 0.
The grader scores a decline as not-passed and can quote back exactly which operation was declined.
"""
from __future__ import annotations

import argparse
import json
import sys

TARGET = "@TARGET@"

REASON = (
    "this package's compiler has not been written: the generated manifest is a WORK ORDER naming the "
    "four experiment-ABI commands (" + ", ".join(@COMMANDS@) + ") that phase 1 must implement. "
    "No lowering exists, so no program is emitted for this input."
)


def _declined_buffer(op: str) -> dict:
    """A buffer valid under BOTH command-buffer schemas, carrying the decline.

    The experiment ABI's schema (merlin/contract/schemas) requires abi_version/target/commands; the
    runtime's own (merlin/schemas) additionally requires backend/resources/metrics_requested. A decline
    that only one of them accepts would be rejected as malformed by whichever reader got it second,
    and "your buffer is malformed" is not the message this is trying to send.
    """
    return {
        "abi_version": "0.1",
        "target": TARGET,
        "backend": "simulator",
        "tensors": {},
        "commands": [],
        "params": {},
        "resources": {},
        "metrics_requested": [],
        "declined": {"reason": REASON, "op": op},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="@TARGET@-compile", add_help=True)
    ap.add_argument("input", nargs="?", help="the interface MLIR this package was asked to lower")
    ap.add_argument("--emit-command-buffer", dest="out_json", default=None)
    ap.add_argument("--emit-target-artifact", action="store_true")
    args, _unrecognised = ap.parse_known_args(argv)

    buffer = _declined_buffer(op=args.input or "(no input named)")
    text = json.dumps(buffer, indent=2, sort_keys=True) + "\\n"
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    # Exit 0 DELIBERATELY: the decline IS the answer, and a non-zero exit would be read as a broken
    # tool rather than as a backend that stated it does not handle this input.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def compiler_tool_source(target: str) -> str:
    """The generated ``tools/compile.py`` — the declining entrypoint the work order names."""
    commands = "[" + ", ".join(repr(c) for c in ABI_COMMANDS) + "]"
    return _COMPILER_TOOL_PY.replace("@TARGET@", target).replace("@COMMANDS@", commands)


def compiler_manifest(target: str, *, package_id: str | None = None) -> dict[str, Any]:
    """The experiment-ABI ``manifest.yaml`` for a generated package: a schema-valid WORK ORDER.

    Schema-valid on purpose — it is the frozen v0.1 ABI (``mlir_oot_backend_contract.yaml``) and a
    work order that does not validate is not a work order, it is a second problem. What it does NOT
    claim is that any of the four commands works: ``package_capabilities.compiler.provided: false``
    says so in the one vocabulary every loader now shares, so ``oot_runner`` refuses BEFORE invoking
    anything and the refusal names this key rather than whichever field happened to be missing first.
    """
    tool = COMPILER_TOOL
    return {
        "artifact_type": "mlir_oot_target_backend",
        "target": target,
        "package_id": package_id or f"{target}_generated_v0",
        "language": "python",
        "authoring": {"mode": "deterministic_generated_from_spec", "generated_by_agent": False},
        "integrity_exempt": False,
        "entrypoints": {"tool": tool},
        "commands": {
            "parse": {"argv": ["python3", tool, "{input_mlir}"]},
            "lower_interface_to_target": {"argv": ["python3", tool, "{input_mlir}"]},
            "emit_command_buffer": {"argv": ["python3", tool, "--emit-command-buffer={output_json}", "{input_mlir}"]},
            "lower_target_to_llvm": {"argv": ["python3", tool, "--emit-target-artifact", "{input_mlir}"]},
        },
        # The shared capability vocabulary (merlin.targetgen.package). Explicit beats inferred, and
        # what is declared here is an ABSENCE with its cause -- the honest state of a fresh package.
        "package_capabilities": {
            "compiler": {
                "provided": False,
                "missing": [
                    f"{tool}: a real lowering for each of {list(ABI_COMMANDS)}",
                ],
            }
        },
        "notes": (
            "WORK ORDER, not a compiler. Generated by merlin.targetgen.generate.oot_package. Phase 1 "
            "replaces tools/compile.py with a real out-of-tree compiler and flips "
            "package_capabilities.compiler.provided to true; until then the tool emits a command "
            "buffer carrying declined.reason so a decline is never mistaken for zeros."
        ),
    }
