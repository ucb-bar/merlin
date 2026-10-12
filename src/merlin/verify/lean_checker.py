"""Run generated, artifact-bound obligations through the selected Lean kernel.

This module installs nothing. It neither accepts saved proofs nor turns missing
tools, tactic failures or opaque semantics into proof evidence. The exporter,
structural correspondence checks and admitted primitive meanings remain in the
trusted computing base; hardware/LLVM conformance is checked separately.
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

from .lean_contract import DOMAINS, SCHEMA, LeanToolchain, LeanTransformReceipt, digest, document_digest
from .lean_export import UnsupportedLeanIR, equivalence_query, export_scalar

_ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
_MAX_QUERY = 1024 * 1024
_MAX_OUTPUT = 4 * 1024 * 1024
_COMPLETED: weakref.WeakValueDictionary[int, LeanTransformReceipt] = weakref.WeakValueDictionary()
_PROOF_CACHE: OrderedDict[str, tuple[str, ...]] = OrderedDict()
_CACHE_LIMIT = 128


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


def pipeline_semantic_identity() -> str:
    root = Path(__file__).parent.parent / "transforms"
    return digest(
        semantic_identity().encode()
        + b"".join(
            (root / name).read_bytes()
            for name in ("certificates.py", "normalization.py", "loops.py", "planning.py", "validation.py")
        )
    )


def accepted_pipeline_receipt(receipt, change, *, toolchain: LeanToolchain | None = None) -> bool:
    """Bind the completion to the entire change's fixed domain and semantic scope."""
    domain = {
        "shape_view_normalization": DOMAINS[0],
        "exact_integer_simplification": DOMAINS[0],
        "ordered_loop_segmentation": DOMAINS[1],
        "buffer_reuse": DOMAINS[2],
        "sequential_dependency_schedule": DOMAINS[3],
    }.get(change.kind)
    return (
        isinstance(receipt, LeanTransformReceipt)
        and accepted_receipt(receipt, change.before_ir, change.after_ir)
        and receipt.domain == domain
        and receipt.scope == tuple(change.certificate.scope)
        and receipt.semantics_sha256 == pipeline_semantic_identity()
        and receipt.toolchain_sha256 is not None
        and (toolchain is None or receipt.toolchain_sha256 == toolchain.identity())
    )


def semantic_identity(registry: Mapping | None = None) -> str:
    root = Path(__file__).parent
    parts = b"".join(
        name.encode() + b"\0" + (root / name).read_bytes() + b"\0"
        for name in ("lean_contract.py", "lean_export.py", "lean_checker.py", "lean/Merlin/Foundation.lean")
    )
    return digest(parts + document_digest(dict(registry or {})).encode())


def _run(argv, toolchain, directory, *, deadline):
    """One bounded, owned process group; output stays in task scratch files."""
    from merlin.common import invocation_record

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
    with tempfile.TemporaryFile(dir=directory) as out, tempfile.TemporaryFile(dir=directory) as err:
        with invocation_record.observe(
            directory / "invocation",
            stage="lean_kernel",
            argv=argv,
            inputs=tuple(Path(v) for v in argv[-1:] if Path(v).is_file()),
            dependencies=(toolchain.executable, *(p for p, _ in toolchain.files)),
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
            completed = replace(
                receipt,
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
                [str(toolchain.executable), "--deps", str(source)], toolchain, directory / "deps", deadline=deadline
            )
            if rc:
                raise FileNotFoundError("selected Lean imports could not be resolved: " + errors[-1000:])
            pinned = {str(p.resolve()): h for p, h in toolchain.files}
            for line in dependencies.splitlines():
                path = Path(line.strip())
                if not path.is_file() or str(path.resolve()) not in pinned:
                    raise ValueError("Lean import is outside the selected dependency inventory")
            rc, stdout, stderr = _run(
                [str(toolchain.executable), "--json", "-j1", f"-M{toolchain.memory_mb}", str(source)],
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
                from merlin.common.path_scrub import scrub_host_paths

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
            if not set(axioms) <= _ALLOWED_AXIOMS:
                raise ValueError("Lean obligation depends on an unapproved axiom")
            completed = replace(
                receipt,
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
    from merlin.common.path_scrub import scrub_host_paths

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

    from merlin.common.path_scrub import scrub_host_paths

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

    from merlin.common.path_scrub import scrub_host_paths
    from merlin.transforms.certificates import BufferReuseWitness, IntegerWitness, LoopWitness, ShapeWitness
    from merlin.transforms.validation import validate_change

    model = pipeline_semantic_identity()
    domain = {
        "shape_view_normalization": DOMAINS[0],
        "exact_integer_simplification": DOMAINS[0],
        "ordered_loop_segmentation": DOMAINS[1],
        "buffer_reuse": DOMAINS[2],
        "sequential_dependency_schedule": DOMAINS[3],
    }.get(change.kind, "unsupported")
    receipt = LeanTransformReceipt(
        SCHEMA,
        "lean_kernel",
        domain,
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
    try:
        validate_change(change)
        statements, proofs = [], []
        if change.kind == "sequential_dependency_schedule":
            # General opaque effects have no Lean state interpretation yet.
            # Prove supported scalar programs directly; retain the gap elsewhere.
            from xdsl.dialects.func import FuncOp

            from .lean_export import parse_ir

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
