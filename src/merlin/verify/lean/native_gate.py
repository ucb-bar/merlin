"""Replayed tensor-prefix proof for a held ordinary native Module.

A verified result covers the actual source/preparation/preprocessing/native tensor
chain. Bufferization and the remaining native suffix remain trusted lowering.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from merlin.llvmlower import proof_stage_transport as transport
from merlin.llvmlower.toolchain import m2m_python

from . import check, native_plan, plans, preparation, proof_worker, stages
from .handoff import Options, VerificationError, VerifiedPreprocessing, _inventory


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Session:
    """Invocation-owned prefix witnesses; all retained evidence is independently replayed."""

    options: Options
    source: str
    verified: VerifiedPreprocessing
    preprocessing_plan: plans.Plan
    captured_preparation: preparation.CapturedPreparation | None = None

    def prepare(self, work: Path, actual_input: str, *, timeout: int = 7200, **selection) -> native_plan.NativePlan:
        self.proof_deadline = time.monotonic() + timeout
        if actual_input != self.verified.text:
            raise VerificationError("native input differs from the checked preprocessing endpoint")
        if (selection["index_bits"], Path(self.options.compiler_python).absolute()) != (
            self.options.index_bits,
            Path(m2m_python()).absolute(),
        ):
            raise VerificationError("native consumer interpreter or index domain differs from proof options")
        self.plan = native_plan.ordinary(**selection)
        self.root = Path(work) / "lean-native-prefix" / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        self.owner = self.root / "native"
        self.owner.mkdir()
        self._status("unmeasured", "native tensor-prefix proof has not completed")
        (self.root / "expected-native-plan.json").write_bytes(self.plan.canonical_bytes())
        (self.root / "expected-boundaries.json").write_bytes(self.plan.boundaries.canonical_bytes())
        self.consumer = transport.identity(self.options.compiler_python, timeout=self._remaining())
        if self.consumer["compiler_python"] != transport.fingerprint(self.options.compiler_python):
            raise VerificationError("measured native interpreter differs from proof invocation")
        self.binding = {
            "function": self.options.function,
            "index_bits": self.options.index_bits,
            "data_layout": self.plan.data_layout,
            "input_sha256": _sha(actual_input.encode()),
            "consumer": self.consumer,
        }
        self.binding_bytes = transport.canonical(self.binding)
        (self.root / "binding.json").write_bytes(self.binding_bytes)
        self.run = uuid.uuid4().hex
        self.sources = {}
        from merlin.llvmlower import fusion_guard, pipeline

        for module in (
            native_plan,
            proof_worker,
            transport,
            pipeline,
            fusion_guard,
            check,
            plans,
            preparation,
            stages,
            __import__(Options.__module__, fromlist=["Options"]),
            __import__(__name__, fromlist=["Session"]),
        ):
            path = Path(module.__file__)
            self.sources[str(path)] = transport.fingerprint(path)
        self.sources[str(check.MLIR_EXPORT)] = transport.fingerprint(check.MLIR_EXPORT)
        self.prefix_records, self.prefix_frozen = self._prefix_inventory()
        return self.plan

    def _status(self, verdict: str, reason: str) -> None:
        check._atomic_json(
            self.root / "status.json",
            {
                "schema": "merlin.lean.native_gate_status.v1",
                "verdict": verdict,
                "reason": reason,
                "scope": "tensor prefix before bufferization; native suffix remains trusted",
            },
        )

    def _prefix_inventory(self):
        if plans.preprocessing() != self.preprocessing_plan:
            raise VerificationError("effective preprocessing roster changed before native consumption")
        root = self.verified.report.parent
        stage_root = root / "stages"
        published = stages.completed(stage_root)
        if not published:
            raise VerificationError("native composition requires a complete preprocessing prefix")
        run = json.loads((published[0] / "meta.json").read_bytes())["run"]
        published, frozen = _inventory(stages.Capture(stage_root, run, complete=True))
        plans.replay(self.preprocessing_plan, root / "expected-plan.json", stage_root)
        if (published[0] / "before.mlir").read_bytes() != self.source.encode():
            raise VerificationError("preprocessing prefix does not start at the actual supplied source")
        if (published[-1] / "after.mlir").read_bytes() != self.verified.text.encode():
            raise VerificationError("preprocessing prefix does not end at the actual native input")
        if self.captured_preparation is not None:
            earlier, earlier_frozen = preparation.validate(
                self.captured_preparation, self.options, self.source.encode()
            )
            published = earlier + published
            frozen = {**earlier_frozen, **frozen}
        return published, frozen

    def bind_runner(self, source: str, *, timeout: int) -> str:
        return transport.bind_runner(
            source,
            directory=self.owner,
            run=self.run,
            plan=self.plan,
            binding=self.binding,
            max_bytes=64 * 1024 * 1024,
            timeout=min(timeout, self._remaining()),
            deadline=self.proof_deadline,
        )

    def freeze_command(self, command) -> None:
        if Path(command[0]).absolute() != Path(self.options.compiler_python).absolute():
            raise VerificationError("actual native command uses another compiler interpreter")
        if (
            len(command) != 22
            or command[4] != self.plan.pipeline()
            or command[18] != self.plan.data_layout
            or any(value != "0" for value in (*command[5:18], *command[19:]))
        ):
            raise VerificationError("actual native command differs from ordinary authored pipeline/configuration")
        self.command = tuple(str(x) for x in command)
        for index in (1, 2):
            self.sources[str(Path(command[index]))] = transport.fingerprint(Path(command[index]))
        if _sha(Path(command[2]).read_bytes()) != self.binding["input_sha256"]:
            raise VerificationError("actual native source file differs from the verified input")
        self.sources_bytes = transport.canonical(self.sources)
        self.command_bytes = transport.canonical(self.command)
        (self.root / "sources.json").write_bytes(self.sources_bytes)
        (self.root / "command.json").write_bytes(self.command_bytes)
        self._recheck_selection()

    def _remaining(self) -> float:
        remaining = self.proof_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("native tensor invocation exceeded its explicit deadline")
        return remaining

    def _recheck_selection(self) -> None:
        self._remaining()
        current = native_plan.ordinary(
            features=frozenset(), index_bits=self.plan.index_bits, data_layout=self.plan.data_layout
        )
        if current != self.plan or Path(m2m_python()).absolute() != Path(self.options.compiler_python).absolute():
            raise VerificationError("native compiler selection or authored phase plan changed")
        from merlin.llvmlower.impr_features import normalize
        from merlin.llvmlower.optional_passes import selected_features

        if normalize(selected_features(frozenset())):
            raise VerificationError("native effective features changed while proof was pending")
        if (self.root / "expected-native-plan.json").read_bytes() != self.plan.canonical_bytes():
            raise VerificationError("native expected plan manifest changed")
        if (self.root / "binding.json").read_bytes() != self.binding_bytes:
            raise VerificationError("native proof binding changed")
        current_binding = {
            "function": self.options.function,
            "index_bits": self.options.index_bits,
            "data_layout": self.plan.data_layout,
            "input_sha256": _sha(self.verified.text.encode()),
            "consumer": self.consumer,
        }
        if self.binding != current_binding or transport.canonical(self.binding) != self.binding_bytes:
            raise VerificationError("native proof options changed")
        for path, expected in self.sources.items():
            if transport.fingerprint(Path(path)) != expected:
                raise VerificationError("native producer source changed while proof was pending")
        if transport.identity(self.options.compiler_python, timeout=self._remaining()) != self.consumer:
            raise VerificationError("native compiler bindings changed while proof was pending")
        if hasattr(self, "command_bytes") and (
            (self.root / "sources.json").read_bytes() != self.sources_bytes
            or (self.root / "command.json").read_bytes() != self.command_bytes
        ):
            raise VerificationError("native source/command manifests changed")

    def _verify(self, request: dict, owner: Path) -> transport.HoldAcknowledgment:
        self._recheck_selection()
        if self._prefix_inventory() != (self.prefix_records, self.prefix_frozen):
            raise VerificationError("source/preparation/preprocessing evidence changed before native proof")
        local, frozen = _inventory(stages.Capture(owner / "stages", self.run, complete=True))
        plans.replay(self.plan.boundaries, self.root / "expected-boundaries.json", owner / "stages")
        if (local[0] / "before.mlir").read_bytes() != self.verified.text.encode():
            raise VerificationError("native parse does not consume exact verified preprocessing bytes")
        held = (owner / "current.mlir").read_bytes()
        if (local[-1] / "after.mlir").read_bytes() != held or _sha(held) != request["held_sha256"]:
            raise VerificationError("native proof does not reach the exact held module")
        for stage in local:
            receipt = check.check_typed_stage(
                stage, m2m_python=self.options.compiler_python, index_bits=self.options.index_bits
            )
            if receipt["verdict"] != "verified":
                raise VerificationError(f"{stage.name}: {receipt.get('reason', receipt['verdict'])}")
        original = self.prefix_records[0] / "before.mlir"
        for checker, name in ((check.check_support, "support"), (check.check_definedness, "definedness")):
            receipt = checker(
                original,
                self.options.function,
                self.root / name,
                m2m_python=self.options.compiler_python,
                index_bits=self.options.index_bits,
            )
            if receipt["verdict"] != "verified":
                raise VerificationError(f"original source {name}: {receipt.get('reason', receipt['verdict'])}")
        chain_path = self.root / "chain.json"
        chain = check.compose_chain(
            self.prefix_records + local,
            self.root / "support" / "receipt.json",
            self.options.function,
            chain_path,
            handoff=owner / "stages" / "handoff.json",
            m2m_python=self.options.compiler_python,
            definedness=self.root / "definedness" / "receipt.json",
        )
        if chain["verdict"] != "verified":
            raise VerificationError(chain.get("reason", "concatenated native tensor chain did not verify"))
        self._recheck_selection()
        if self._prefix_inventory() != (self.prefix_records, self.prefix_frozen):
            raise VerificationError("source/preparation/preprocessing evidence changed during native proof")
        if _inventory(stages.Capture(owner / "stages", self.run, complete=True)) != (local, frozen):
            raise VerificationError("native tensor snapshots changed during proof replay")
        plans.replay(self.plan.boundaries, self.root / "expected-boundaries.json", owner / "stages")
        if (owner / "current.mlir").read_bytes() != held:
            raise VerificationError("held native snapshot changed during proof replay")
        self.local_records, self.local_frozen = local, frozen
        self._remaining()
        report = self.root / "verification.json"
        check._atomic_json(
            report,
            {
                "schema": "merlin.lean.native_gate.v1",
                "verdict": "verified",
                "scope": "actual tensor chain through ordinary native prefix, before bufferization",
                "trusted_suffix": "bufferization and remaining native lowering (T4); no native execution proof",
                "coverage_granularity": self.plan.boundaries.granularity,
                "function": self.options.function,
                "index_bits": self.options.index_bits,
                "source_sha256": _sha(original.read_bytes()),
                "held_sha256": _sha(held),
                "plan_sha256": self.plan.sha256,
                "preprocessing_plan_sha256": self.preprocessing_plan.sha256,
                "preparation_plan_sha256": self.captured_preparation.plan.sha256 if self.captured_preparation else None,
                "boundary_count": len(self.prefix_records) + len(local),
                "consumer": self.consumer,
                "binding_sha256": _sha(self.binding_bytes),
                "sources_sha256": _sha(self.sources_bytes),
                "command_sha256": _sha(self.command_bytes),
                "chain": str(chain_path),
                "chain_sha256": _sha(chain_path.read_bytes()),
                "domain": chain["domain"],
                "assumptions": chain["assumptions"],
            },
        )
        self.proof_frozen = {str(p): _sha(p.read_bytes()) for p in (report, chain_path)}
        return transport.HoldAcknowledgment(_sha((owner / "request.json").read_bytes()), report, chain_path)

    def _verify_bounded(self, request: dict, owner: Path) -> transport.HoldAcknowledgment:
        # Only explicit immutable evidence/options cross the spawned boundary;
        # preprocessor statistics may contain unrelated host objects and are excluded.
        snapshot = Session(
            self.options,
            self.source,
            VerifiedPreprocessing(self.verified.text, {}, self.verified.report),
            self.preprocessing_plan,
            self.captured_preparation,
        )
        for key in (
            "plan",
            "root",
            "owner",
            "consumer",
            "binding",
            "binding_bytes",
            "run",
            "sources",
            "prefix_records",
            "prefix_frozen",
            "command",
            "sources_bytes",
            "command_bytes",
            "proof_deadline",
        ):
            setattr(snapshot, key, getattr(self, key))
        result = proof_worker.run(
            _prove, snapshot, request, owner, timeout=self.proof_deadline - time.monotonic(), logs=owner
        )
        if set(result) != {"request_sha256", "local_records", "local_frozen", "proof_frozen"}:
            raise VerificationError("native proof worker returned unexpected evidence fields")
        self._recheck_selection()
        if self._prefix_inventory() != (self.prefix_records, self.prefix_frozen):
            raise VerificationError("original evidence changed while the proof worker was running")
        actual = _inventory(stages.Capture(owner / "stages", self.run, complete=True))
        if actual != (result["local_records"], result["local_frozen"]):
            raise VerificationError("proof worker inventory differs from actual native snapshots")
        plans.replay(self.plan.boundaries, self.root / "expected-boundaries.json", owner / "stages")
        proof_paths = (self.root / "verification.json", self.root / "chain.json")
        expected_proofs = {str(path): _sha(path.read_bytes()) for path in proof_paths}
        if result["proof_frozen"] != expected_proofs:
            raise VerificationError("proof worker report/chain changed or name unexpected evidence")
        request_sha = _sha((owner / "request.json").read_bytes())
        if result["request_sha256"] != request_sha:
            raise VerificationError("proof worker acknowledged another pending request")
        self.local_records, self.local_frozen = actual
        self.proof_frozen = expected_proofs
        self._remaining()
        return transport.HoldAcknowledgment(request_sha, *proof_paths)

    def run_command(self, command, *, timeout: int):
        if tuple(str(x) for x in command) != self.command:
            raise VerificationError("native command changed after producer binding")
        try:
            self._recheck_selection()
            result = transport.run_command(
                command,
                directory=self.owner,
                callback=self._verify_bounded,
                run=self.run,
                plan=self.plan,
                binding=self.binding,
                max_bytes=64 * 1024 * 1024,
                timeout=min(timeout, self._remaining()),
            )
            if result.returncode == 0:
                self._recheck_selection()
                if self._prefix_inventory() != (self.prefix_records, self.prefix_frozen):
                    raise VerificationError("original prefix evidence changed during native consumption")
                if _inventory(stages.Capture(self.owner / "stages", self.run, complete=True)) != (
                    self.local_records,
                    self.local_frozen,
                ):
                    raise VerificationError("native snapshots changed during native consumption")
                plans.replay(self.plan.boundaries, self.root / "expected-boundaries.json", self.owner / "stages")
                if any(_sha(Path(p).read_bytes()) != digest for p, digest in self.proof_frozen.items()):
                    raise VerificationError("acknowledged tensor proof changed during native consumption")
                self._remaining()
                self._status("verified", "native suffix resumed only after exact tensor-prefix acknowledgment")
            else:
                self._status("unmeasured", f"native child failed with exit {result.returncode}")
            return result
        except BaseException as exc:
            self._status("unmeasured", f"{type(exc).__name__}: {exc}")
            raise


def _prove(session: Session, request: dict, owner: Path) -> dict:
    proof = session._verify(request, owner)
    return {
        "request_sha256": proof.request_sha256,
        "local_records": session.local_records,
        "local_frozen": session.local_frozen,
        "proof_frozen": session.proof_frozen,
    }
