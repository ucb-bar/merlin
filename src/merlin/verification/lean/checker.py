"""Mathematical projections after actual-IR structural correspondence has passed.

Totality/effects, SSA argument identity and exact body cloning are checked by
the frozen structural validator. These lemmas model functional observations,
not timing-sensitive environment interactions or physical target state.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from merlin.compiler.transforms.structural_rewrites import ScalarRewriteWitness, UnrollWitness
from merlin.verification.lean.contract import (
    DOMAINS,
    EXTENDED_DOMAINS,
    SCHEMA,
    LeanToolchain,
    LeanTransformReceipt,
    digest,
    document_digest,
)
from merlin.verification.lean.host_checks import UnsupportedLeanIR, equivalence_query, export_scalar

# ---- lean_obligations --------------------------------------------------------------------------


def extended_obligations(change) -> tuple[list[str], list[str]]:
    statements, proofs = [], []
    if change.kind in {"scalar_redundancy_elimination", "scalar_loop_invariant_motion"}:
        rules = set()
        for witness in change.certificate.witnesses:
            if not isinstance(witness, ScalarRewriteWitness) or not 1 <= witness.width <= 256:
                raise ValueError("scalar rewrite witness lacks its admitted actual integer operation")
            rules.add(witness.rule)
        for rule in sorted(rules):
            if rule == "cse":
                statements.append(
                    "∀ (α β σ : Type) (producer : α → β) (observe : β → β → σ) (argument : α), "
                    "(let first := producer argument; let second := producer argument; observe first second) = "
                    "(let first := producer argument; observe first first)"
                )
                proofs.append(
                    "by intro α β σ producer observe argument; "
                    "exact Merlin.eliminateDuplicate producer observe argument"
                )
            elif rule == "dce":
                statements.append(
                    "∀ (α β σ : Type) (producer : α → β) (argument : α) (result : σ), "
                    "(let _unused := producer argument; result) = result"
                )
                proofs.append(
                    "by intro α β σ producer argument result; exact Merlin.eliminateUnused producer argument result"
                )
            elif rule == "licm":
                statements.append(
                    "∀ (α β σ : Type) (producer : α → β) (argument : α) (body : β → σ → Nat → σ) "
                    "(iterations : List Nat) (initial : σ), "
                    "iterations.foldl (fun state index => body (producer argument) state index) initial = "
                    "(let invariant := producer argument; iterations.foldl (body invariant) initial)"
                )
                proofs.append(
                    "by intro α β σ producer argument body iterations initial; "
                    "exact Merlin.hoistInvariant producer argument body iterations initial"
                )
            else:
                raise ValueError("scalar rewrite has no admitted mathematical rule")
        return statements, proofs
    if change.kind != "ordered_loop_unrolling":
        raise ValueError("unknown extended mathematical domain")
    for witness in change.certificate.witnesses:
        if not isinstance(witness, UnrollWitness):
            raise ValueError("unrolling requires its complete source geometry")
        groups, factor, rest, count = witness.full_groups, witness.factor, witness.remainder, witness.iterations
        bulk = groups * factor
        statements.extend(
            [
                f"(0 < {factor}) ∧ ({bulk} + {rest} = {count}) ∧ ({rest} < {factor})",
                f"List.range' {witness.lower} {bulk} {witness.step} ++ "
                f"List.range' ({witness.lower} + {witness.step} * {bulk}) {rest} {witness.step} = "
                f"List.range' {witness.lower} {count} {witness.step}",
                f"∀ index : Nat, {factor} * (index / {factor}) + index % {factor} = index",
                "∀ (σ : Type) (body : σ → Nat → σ) (initial : σ) (chunks : List (List Nat)), "
                "chunks.foldl (fun state chunk => chunk.foldl body state) initial = chunks.flatten.foldl body initial",
                f"{witness.upper} + {witness.step} * {factor} ≤ {2 ** (witness.index_bits - 1) - 1}",
            ]
        )
        proofs.extend(
            [
                "by decide",
                f"by exact Merlin.rangeSplit {witness.lower} {bulk} {rest} {witness.step}",
                f"by intro index; exact Merlin.tileIndex index {factor}",
                "by intros; exact Merlin.orderedSegments _ _ _",
                "by decide",
            ]
        )
    return statements, proofs

    # Preserve historical object lookup; implementation and completion state live here.


# ---- lean_sat_runtime --------------------------------------------------------------------------
# Original pinned SAT executable for generated bit-vector obligations.
#
# The solver supplies an LRAT certificate checked by the same selected Lean
# toolchain. It is not proof authority. No ambient PATH or default sibling binary
# may supply this runtime dependency, and candidates cannot select tactic options.

NATIVE_AXIOMS = frozenset({"Lean.ofReduceBool", "Lean.trustCompiler"})
NATIVE_ASSUMPTION = "originally admitted Lean native reflection, Lean compiler and LRAT certificate checker are trusted"


def _tokens(source):
    """Lex identifiers in generated Lean, excluding strings and nested comments."""
    tokens, word = [], []
    index, depth = 0, 0
    while index < len(source):
        pair = source[index : index + 2]
        if depth:
            if pair == "/-":
                depth += 1
                index += 2
            elif pair == "-/":
                depth -= 1
                index += 2
            else:
                index += 1
            continue
        if pair in {"/-", "--"} or source[index] == '"':
            if word:
                tokens.append("".join(word))
                word = []
            if pair == "/-":
                depth, index = 1, index + 2
            elif pair == "--":
                newline = source.find("\n", index)
                index = len(source) if newline < 0 else newline + 1
            else:
                index += 1
                while index < len(source):
                    if source[index] == "\\":
                        index += 2
                    elif source[index] == '"':
                        index += 1
                        break
                    else:
                        index += 1
            continue
        character = source[index]
        if character.isalnum() or character in "_.'":
            word.append(character)
        elif word:
            tokens.append("".join(word))
            word = []
        index += 1
    if word:
        tokens.append("".join(word))
    return tuple(tokens)


def validate_sat_solver(solver, files):
    """Require one original canonical executable already in the file inventory."""
    if solver is None:
        return
    if type(solver) is not tuple or len(solver) != 2:
        raise ValueError("Lean SAT selection requires an exact path/SHA256 pair")
    path, identity = solver
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path.resolve() != path
        or any(owner.is_symlink() for owner in (path, *path.parents))
        or type(identity) is not str
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
        or solver not in files
    ):
        raise ValueError("Lean SAT solver is outside its exact original file inventory")
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError("selected Lean SAT solver is unavailable or nonexecutable")


def prepared_sat_solver(executable, runtime_files, files, *, explicit=None):
    """Select only an explicitly supplied runtime file in Lean's reviewed layout."""
    selected = explicit if explicit is not None else executable.parent / "cadical"
    matches = [path for path in runtime_files if path == selected]
    if not matches:
        if explicit is not None:
            raise ValueError("explicit Lean SAT executable is outside its runtime inventory")
        return None
    if len(matches) != 1:
        raise ValueError("Lean SAT runtime selection is duplicated")
    solver = next(row for row in files if row[0] == selected)
    validate_sat_solver(solver, files)
    return solver


def proof_runtime_arguments(query, toolchain):
    """Fixed host options, checked before any execution or cached completion."""
    tokens = _tokens(query)
    if any(pair == ("set_option", "sat.solver") for pair in zip(tokens, tokens[1:])):
        raise ValueError("generated proof cannot override the original pinned SAT solver")
    if "bv_decide" not in tokens:
        return ()
    if toolchain.sat_solver is None:
        raise FileNotFoundError("generated bit-vector proof requires an explicitly selected pinned SAT solver")
    validate_sat_solver(toolchain.sat_solver, toolchain.files)
    if not toolchain.allow_native_reflection:
        raise FileNotFoundError("bit-vector proof requires original Lean native-reflection admission")
        # Lean's sat.solver is an exact executable string. A nonempty absolute value
        # prevents both sibling discovery and PATH fallback, also after relocation.
    return ("-Dsat.solver=" + str(toolchain.sat_solver[0]),)


def completion_assurance(query, toolchain, axioms, assumptions, allowed_axioms):
    """Qualify actual reported axioms; no extra axiom is globally admitted."""
    native = bool(set(axioms) & NATIVE_AXIOMS)
    if native:
        if not toolchain.allow_native_reflection or "bv_decide" not in _tokens(query):
            raise ValueError("Lean obligation depends on unadmitted native-reflection axioms")
        proof_runtime_arguments(query, toolchain)
    if not set(axioms) <= allowed_axioms | (NATIVE_AXIOMS if native else frozenset()):
        raise ValueError("Lean obligation depends on an unapproved axiom")
    if native:
        return "lean_native_reflection", tuple(dict.fromkeys((*assumptions, NATIVE_ASSUMPTION)))
    return "lean_kernel", assumptions


def runtime_stage(argv):
    return "lean_native_reflection" if any(value.startswith("-Dsat.solver=") for value in argv) else "lean_kernel"


def runtime_receipt(receipt, options):
    """Failed requested native checks also describe their actual runtime scope."""
    if not options:
        return receipt
    return replace(
        receipt,
        method="lean_native_reflection",
        assumptions=tuple(dict.fromkeys((*receipt.assumptions, NATIVE_ASSUMPTION))),
    )

    # Preserve historical object lookup; implementation and completion state live here.


# ---- lean_checker ------------------------------------------------------------------------------
# Run generated, artifact-bound obligations through the selected Lean kernel.
#
# This module installs nothing. It neither accepts saved proofs nor turns missing
# tools, tactic failures or opaque semantics into proof evidence. The exporter,
# structural correspondence checks and admitted primitive meanings remain in the
# trusted computing base; hardware/LLVM conformance is checked separately.

_ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
_MAX_QUERY = 1024 * 1024
_MAX_OUTPUT = 4 * 1024 * 1024
_COMPLETED: weakref.WeakValueDictionary[int, LeanTransformReceipt] = weakref.WeakValueDictionary()
_PROOF_CACHE: OrderedDict[str, tuple[str, ...]] = OrderedDict()
_CACHE_LIMIT = 128
_TRANSFORM_DOMAINS = {
    "shape_view_normalization": DOMAINS[0],
    "exact_integer_simplification": DOMAINS[0],
    "ordered_loop_segmentation": DOMAINS[1],
    "buffer_reuse": DOMAINS[2],
    "sequential_dependency_schedule": DOMAINS[3],
    "scalar_redundancy_elimination": EXTENDED_DOMAINS[4],
    "scalar_loop_invariant_motion": EXTENDED_DOMAINS[4],
    "ordered_loop_unrolling": EXTENDED_DOMAINS[5],
}


def accepted_receipt(receipt: LeanTransformReceipt, before: str, after: str) -> bool:
    """Only a fresh completion in this host process can be accepted by its broker.

    Candidate subprocesses and deserialized summaries do not share this ledger.
    Neither a status string nor a callback returning True creates completion.
    """
    return (
        _COMPLETED.get(id(receipt)) is receipt
        and receipt.verified
        and receipt.source_sha256 == digest(before)
        and receipt.target_sha256 == digest(after)
    )


def pipeline_semantic_identity(kind: str | None = None) -> str:
    root = Path(__file__).parents[2] / "compiler" / "transforms"
    selected = ["certificates.py", "normalization.py", "loops.py", "planning.py", "validation.py"]
    additional = b""
    if kind in {"scalar_redundancy_elimination", "scalar_loop_invariant_motion", "ordered_loop_unrolling"}:
        selected += ["pure_scalar.py", "scalar_witness.py", "unrolling.py"]
        additional = (Path(__file__).parent / "checker.py").read_bytes()
    return digest(semantic_identity().encode() + b"".join((root / name).read_bytes() for name in selected) + additional)


def accepted_pipeline_receipt(receipt, change, *, toolchain: LeanToolchain | None = None) -> bool:
    """Bind the completion to the entire change's fixed domain and semantic scope."""
    domain = _TRANSFORM_DOMAINS.get(change.kind)
    return (
        isinstance(receipt, LeanTransformReceipt)
        and accepted_receipt(receipt, change.before_ir, change.after_ir)
        and receipt.domain == domain
        and receipt.scope == tuple(change.certificate.scope)
        and receipt.semantics_sha256 == pipeline_semantic_identity(change.kind)
        and receipt.toolchain_sha256 is not None
        and (toolchain is None or receipt.toolchain_sha256 == toolchain.identity())
    )


def semantic_identity(registry: Mapping | None = None) -> str:
    root = Path(__file__).parent
    parts = b"".join(
        name.encode() + b"\0" + (root / name).read_bytes() + b"\0"
        for name in (
            "contract.py",
            "host_checks.py",
            "checker.py",
            "Merlin/Foundation.lean",
        )
    )
    return digest(parts + document_digest(dict(registry or {})).encode())


def _run(argv, toolchain, directory, *, deadline):
    """One bounded, owned process group; output stays in task scratch files."""
    from merlin.common.process import invocation_records as invocation_record

    directory.mkdir(parents=True, exist_ok=False)
    environment = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "LEAN_PATH",
            "LEAN_SRC_PATH",
            "LEAN_SYSROOT",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "LD_AUDIT",
            "DYLD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
        }
    }
    environment["LEAN_PATH"] = os.pathsep.join(str(p) for p in toolchain.library_roots)
    environment["TMPDIR"] = str(toolchain.work_root)
    toolchain.verify(deadline=deadline)
    inventory = directory / "observed-inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "merlin.lean_invocation_inputs.v1",
                "toolchain": toolchain.identity(),
                "files": [(str(path), sha) for path, sha in toolchain.files],
            },
            sort_keys=True,
        )
    )
    with tempfile.TemporaryFile(dir=directory) as out, tempfile.TemporaryFile(dir=directory) as err:
        with invocation_record.observe(
            directory / "invocation",
            stage=runtime_stage(argv),
            argv=argv,
            inputs=tuple(Path(v) for v in argv[-1:] if Path(v).is_file()),
            dependencies=(toolchain.executable, inventory),
            cwd=directory,
            env=environment,
        ) as record:
            if time.monotonic() >= deadline:
                raise TimeoutError("Lean exhausted its whole-check wall budget")
            process = subprocess.Popen(
                argv, stdout=out, stderr=err, cwd=directory, env=environment, start_new_session=True
            )
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Lean exceeded the selected wall budget")
                    if os.fstat(out.fileno()).st_size > _MAX_OUTPUT or os.fstat(err.fileno()).st_size > _MAX_OUTPUT:
                        raise TimeoutError("Lean exceeded the selected output budget")
                    time.sleep(0.02)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            out.seek(0)
            err.seek(0)
            stdout, stderr = out.read(_MAX_OUTPUT + 1), err.read(_MAX_OUTPUT + 1)
            if len(stdout) > _MAX_OUTPUT or len(stderr) > _MAX_OUTPUT:
                raise TimeoutError("Lean exceeded the selected output budget")
            result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
            record.complete(result)
    return result.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


def _check_generated(
    query: str, receipt: LeanTransformReceipt, toolchain: LeanToolchain | None, *, deadline: float | None = None
):
    if toolchain is None:
        return replace(receipt, status="unavailable", reason="no explicitly selected Lean toolchain")
    if len(query.encode()) > _MAX_QUERY:
        return replace(receipt, status="unknown", reason="proof query exceeds the admitted work budget")
    started = time.monotonic()
    deadline = min(deadline, started + toolchain.timeout_s) if deadline is not None else started + toolchain.timeout_s
    try:
        toolchain.verify(deadline=deadline)
        runtime_options = proof_runtime_arguments(query, toolchain)
        receipt = runtime_receipt(receipt, runtime_options)
        before_identity = toolchain.identity()
        cache_key = document_digest(
            {
                "query": digest(query),
                "toolchain": before_identity,
                "source": receipt.source_sha256,
                "target": receipt.target_sha256,
                "semantics": receipt.semantics_sha256,
                "domain": receipt.domain,
                "scope": receipt.scope,
                "assumptions": receipt.assumptions,
            }
        )
        if cache_key in _PROOF_CACHE:
            axioms = _PROOF_CACHE[cache_key]
            _PROOF_CACHE.move_to_end(cache_key)
            method, assumptions = completion_assurance(query, toolchain, axioms, receipt.assumptions, _ALLOWED_AXIOMS)
            completed = replace(
                receipt,
                method=method,
                assumptions=assumptions,
                status="verified",
                query_sha256=digest(query),
                toolchain_sha256=before_identity,
                axioms=axioms,
                elapsed_s=time.monotonic() - started,
                cache_hit=True,
            )
            _COMPLETED[id(completed)] = completed
            return completed
        with tempfile.TemporaryDirectory(prefix="lean-check-", dir=toolchain.work_root) as temporary:
            directory = Path(temporary)
            source = directory / "Check.lean"
            source.write_text(query)
            # Check the selected import files before running tactics. No arbitrary
            # candidate imports can reach this generated theorem file.
            rc, dependencies, errors = _run(
                [str(toolchain.executable), *runtime_options, "--deps", str(source)],
                toolchain,
                directory / "deps",
                deadline=deadline,
            )
            if rc:
                raise FileNotFoundError("selected Lean imports could not be resolved: " + errors[-1000:])
            pinned = {str(p.resolve()): h for p, h in toolchain.files}
            for line in dependencies.splitlines():
                path = Path(line.strip())
                if not path.is_file() or str(path.resolve()) not in pinned:
                    raise ValueError("Lean import is outside the selected dependency inventory")
            rc, stdout, stderr = _run(
                [str(toolchain.executable), *runtime_options, "--json", "-j1", f"-M{toolchain.memory_mb}", str(source)],
                toolchain,
                directory / "proof",
                deadline=deadline,
            )
            toolchain.verify(deadline=deadline)
            if digest(source.read_bytes()) != digest(query) or toolchain.identity() != before_identity:
                raise ValueError("Lean inputs changed during checking")
            messages = []
            for line in stdout.splitlines():
                if line.strip():
                    messages.append(json.loads(line))
            if rc or any(row.get("severity") in {"error", "warning"} for row in messages):
                from merlin.common.access import scrub_host_paths

                details = "\n".join(
                    row.get("data", "") for row in messages if row.get("severity") in {"error", "warning"}
                )
                reason = scrub_host_paths(
                    "Lean did not establish the generated obligation: " + (details or stderr)[-3000:]
                )
                return replace(
                    receipt,
                    status="unknown",
                    reason=reason,
                    query_sha256=digest(query),
                    toolchain_sha256=before_identity,
                    elapsed_s=time.monotonic() - started,
                )
            reports = [row.get("data", "") for row in messages if "'merlin_transform'" in row.get("data", "")]
            if len(reports) != 1:
                raise ValueError("Lean did not report the obligation's transitive axioms")
            report = reports[0]
            if "does not depend on any axioms" in report:
                axioms = ()
            else:
                opening, _, tail = report.partition("depends on axioms: [")
                if not opening or not tail.endswith("]"):
                    raise ValueError("unrecognized Lean axiom report")
                axioms = tuple(item.strip() for item in tail[:-1].split(",") if item.strip())
            method, assumptions = completion_assurance(query, toolchain, axioms, receipt.assumptions, _ALLOWED_AXIOMS)
            completed = replace(
                receipt,
                method=method,
                assumptions=assumptions,
                status="verified",
                query_sha256=digest(query),
                toolchain_sha256=before_identity,
                axioms=axioms,
                elapsed_s=time.monotonic() - started,
            )
            _COMPLETED[id(completed)] = completed
            _PROOF_CACHE[cache_key] = axioms
            if len(_PROOF_CACHE) > _CACHE_LIMIT:
                _PROOF_CACHE.popitem(last=False)
            return completed
    except FileNotFoundError as exc:
        status, reason = "unavailable", str(exc)
    except TimeoutError as exc:
        status, reason = "unknown", str(exc)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        status, reason = "error", str(exc)
    from merlin.common.access import scrub_host_paths

    return replace(
        receipt,
        status=status,
        reason=scrub_host_paths(reason),
        query_sha256=digest(query),
        elapsed_s=time.monotonic() - started,
    )


def verify_scalar_transform(
    before: str,
    after: str,
    *,
    entry: str,
    toolchain: LeanToolchain | None,
    registry: Mapping[str, dict] | None = None,
    tactic: str = "simp",
    deadline: float | None = None,
) -> LeanTransformReceipt:
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.common.access import scrub_host_paths

    receipt = LeanTransformReceipt(
        SCHEMA,
        "lean_kernel",
        DOMAINS[0],
        digest(before),
        digest(after),
        semantic_identity(registry),
        "unsupported",
        ("actual scalar integer SSA input/output equality for every argument bit pattern",),
        ("admitted primitive semantics; exporter and Lean kernel/toolchain are trusted",),
    )
    try:
        source = export_scalar(before, entry, registry)
        target = export_scalar(after, entry, registry)
        query = equivalence_query(source, target, tactic=tactic)
    except (UnsupportedLeanIR, ValueError, KeyError, TypeError, AttributeError, ParseError, VerifyException) as exc:
        return replace(receipt, reason=scrub_host_paths(str(exc)))
    return _check_generated(query, receipt, toolchain, deadline=deadline)


def _conjunction(statements: list[str], proofs: list[str]) -> tuple[str, str]:
    if not statements:
        raise UnsupportedLeanIR("no modeled obligation was obtained from the actual change")
    if len(statements) == 1:
        return statements[0], proofs[0]
    tail, proof = _conjunction(statements[1:], proofs[1:])
    return f"({statements[0]}) ∧ ({tail})", f"And.intro ({proofs[0]}) ({proof})"


def verify_pipeline_change(
    change, *, toolchain: LeanToolchain | None, deadline: float | None = None
) -> LeanTransformReceipt:
    """The common broker's host-owned callback on an actual changed IR pair.

    Structural projection is explicit trusted code. Mathematical certificates
    cover local integer equality, ordered loop indexing, or disjoint lifetimes;
    they are not a universal proof of packing, asynchronous DMA or machine code.
    """
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.common.access import scrub_host_paths
    from merlin.compiler.transforms.certificates import BufferReuseWitness, IntegerWitness, LoopWitness, ShapeWitness
    from merlin.compiler.transforms.pipeline import validate_change

    model = pipeline_semantic_identity(change.kind)
    domain = _TRANSFORM_DOMAINS.get(change.kind)
    receipt = LeanTransformReceipt(
        SCHEMA,
        "lean_kernel",
        domain if domain is not None else "unsupported",
        digest(change.before_ir),
        digest(change.after_ir),
        model,
        "unsupported",
        tuple(change.certificate.scope),
        (
            "actual IR structural correspondence checker is trusted",
            "standard operation semantics and valid descriptors",
            "target instruction/RTL, allocation failure and final ABI conformance are separate",
        ),
    )
    if domain is None:
        return replace(receipt, reason=f"transform kind {change.kind!r} has no formal domain")
    try:
        validate_change(change)
        statements, proofs = [], []
        if change.kind in {"scalar_redundancy_elimination", "scalar_loop_invariant_motion", "ordered_loop_unrolling"}:
            statements, proofs = extended_obligations(change)
        if change.kind == "sequential_dependency_schedule":
            # General opaque effects have no Lean state interpretation yet.
            # Prove supported scalar programs directly; retain the gap elsewhere.
            from xdsl.dialects.func import FuncOp

            from merlin.verification.lean.host_checks import parse_ir

            functions = tuple(op for op in parse_ir(change.before_ir).body.block.ops if isinstance(op, FuncOp))
            if len(functions) != 1:
                raise UnsupportedLeanIR("scheduling proof currently requires one modeled scalar function")
            scalar = verify_scalar_transform(
                change.before_ir,
                change.after_ir,
                entry=functions[0].sym_name.data,
                toolchain=toolchain,
                deadline=deadline,
            )
            if not scalar.verified:
                return replace(receipt, status=scalar.status, reason=scalar.reason)
            completed = replace(
                scalar, domain=domain, scope=receipt.scope, assumptions=receipt.assumptions, semantics_sha256=model
            )
            _COMPLETED[id(completed)] = completed
            return completed
        for witness in change.certificate.witnesses:
            if statements and change.kind in {
                "scalar_redundancy_elimination",
                "scalar_loop_invariant_motion",
                "ordered_loop_unrolling",
            }:
                break
            if isinstance(witness, IntegerWitness):
                width = witness.width
                operation = {"arith.addi": "Add", "arith.subi": "Sub", "arith.muli": "Mul"}[witness.operation]
                if witness.rule == "constant_ring":
                    statements.append(
                        f"Merlin.bv{operation} {width} (BitVec.ofInt {width} ({witness.lhs})) "
                        f"(BitVec.ofInt {width} ({witness.rhs})) = BitVec.ofInt {width} ({witness.result})"
                    )
                    proofs.append("by decide")
                else:
                    left = "x" if witness.retained_operand == 0 else f"BitVec.ofInt {width} ({witness.lhs})"
                    right = "x" if witness.retained_operand == 1 else f"BitVec.ofInt {width} ({witness.rhs})"
                    statements.append(f"∀ x : BitVec {width}, Merlin.bv{operation} {width} ({left}) ({right}) = x")
                    proofs.append("by intro x; simp")
            elif isinstance(witness, ShapeWitness):
                if witness.rule == "identity_memref_cast":
                    statements.append("∀ (α : Type) (value : α), id value = value")
                    proofs.append("by intros; rfl")
                else:
                    actual = witness.shape[witness.axis]
                    statements.append(f"({actual} : Nat) = {actual}")
                    proofs.append("by rfl")
            elif isinstance(witness, LoopWitness):
                count = (witness.upper - witness.lower + witness.step - 1) // witness.step
                tile = witness.tile_iterations
                statements.extend(
                    [
                        f"(0 < {tile}) ∧ (0 < {witness.step}) ∧ "
                        f"({witness.upper} + {witness.step} * {tile} ≤ {2 ** (witness.index_bits - 1) - 1})",
                        f"∀ index : Nat, {tile} * (index / {tile}) + index % {tile} = index",
                        f"∀ index : Nat, index < {count} → index % {tile} < "
                        f"min {tile} ({count} - {tile} * (index / {tile}))",
                        "∀ (σ : Type) (body : σ → Nat → σ) (initial : σ) (chunks : List (List Nat)), "
                        "chunks.foldl (fun state chunk => chunk.foldl body state) initial = "
                        "chunks.flatten.foldl body initial",
                    ]
                )
                proofs.extend(
                    [
                        "by decide",
                        f"by intro index; exact Merlin.tileIndex index {tile}",
                        f"by intro index inside; exact Merlin.tileTail index {count} {tile} (by decide) inside",
                        "by intros; exact Merlin.orderedSegments _ _ _",
                    ]
                )
            elif isinstance(witness, BufferReuseWitness):
                for i, left in enumerate(witness.intervals):
                    for right in witness.intervals[i + 1 :]:
                        if left.slot != right.slot:
                            continue
                        a, b, c, d = (
                            left.allocation_index,
                            left.deallocation_index,
                            right.allocation_index,
                            right.deallocation_index,
                        )
                        statements.append(f"∀ time : Nat, ¬ ({a} ≤ time ∧ time ≤ {b} ∧ {c} ≤ time ∧ time ≤ {d})")
                        proofs.append(f"by intro time; exact Merlin.disjointLifetime {a} {b} {c} {d} time (by decide)")
            else:
                raise UnsupportedLeanIR("shared witness has no formal model")
        statement, proof = _conjunction(statements, proofs)
        query = (
            "import Merlin.Foundation\nset_option warningAsError true\n"
            "set_option linter.unusedSimpArgs false\ntheorem merlin_transform :\n  "
            + statement
            + " :=\n  "
            + proof
            + "\n#print axioms merlin_transform\n"
        )
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, ParseError, VerifyException) as exc:
        return replace(receipt, status="error", reason=scrub_host_paths(str(exc)))
    return _check_generated(query, receipt, toolchain, deadline=deadline)

    # Preserve historical object lookup; implementation and completion state live here.
