"""Capsule evaluation: trace."""

from __future__ import annotations

import json

from merlin.development.targetgen import trace_check as TCK
from merlin.development.targetgen.capsule_common import _cat
from merlin.development.targetgen.contract import schemas
from merlin.development.targetgen.elf_lanes import ACCELERATOR_LANE as _ACCELERATOR_LANE
from merlin.development.targetgen.package_runtime import CertFailure
from merlin.development.targetgen.rocc import decode as RD
from merlin.evaluation.capsule.cpu import _grade_cpu_step
from merlin.evaluation.capsule.instructions import enforce_executed_stream
from merlin.evaluation.capsule.lanes import accelerator_lane_violated


def check_trace(capsule, paths, cfg, contract, eff_target, required, no_oracle, evidence):
    evidence.parsed_lowered_module = None

    # --- trace gate (optional; only for targets whose ISA has a decoder+checker plugin) --------
    # A command-ISA target (systolic, trace_gate="rocc_insn") decodes the emitted .insn stream to a
    # RoCC trace and checks instruction coverage. A SIMT/other target (trace_gate=None) has no analog
    # and skips it — the L0/L1 math floor + the oracle tiers carry the verdict either way.
    if cfg.trace_gate == "rocc_insn":
        # PARSED ONCE, then shared. This is the first of up to three full xDSL parses of the same
        # bytes in one grade; on a 29 MB artifact each costs about 44 s plus 5 s to verify, so the
        # plan evidence below is handed THIS module instead of parsing the text again. The decoder
        # keeps its own text path for an artifact that is not a parseable module.
        evidence.parsed_lowered_module = _grade_cpu_step(RD._parse_module, evidence.lowered_text)
        trace = (
            _grade_cpu_step(
                RD.decode_module,
                evidence.parsed_lowered_module,
                target=eff_target,
                source=str(paths.generated / cfg.fourth_output_name),
            )
            if evidence.parsed_lowered_module is not None
            else _grade_cpu_step(
                RD.decode_text,
                evidence.lowered_text,
                source=str(paths.generated / cfg.fourth_output_name),
                target=eff_target,
            )
        )
        evidence.decoded_trace = trace  # hand to the advisory localizer on an oracle divergence
        # Validating OUR OWN decoder's output must never crash the capsule before the oracle runs: a
        # schema hiccup (e.g. the manifest maps a funct to a class the enum lacks) is our tooling's
        # limitation, not the backend's defect — record it and continue, not a pre-oracle RUNNER_CRASH.
        try:
            schemas.validate(trace, "instruction_trace", contract=contract)
        except schemas.ContractViolation as _sv:
            import sys as _sys

            _sys.stderr.write(f"WARNING: instruction_trace self-validation failed (advisory): {_sv}\n")
        (paths.generated / "instruction_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        # Advisory silicon-liveness screen (HW-agnostic): would this emitted stream stall/fault on real
        # silicon? It uses only the target's OWN derived facts (scratchpad/accumulator capacity, address
        # map, ISA legality) + this decoded trace, never a target literal, and — like trace_check — is
        # advisory: it writes liveness_report.json for the author and never gates or crashes the capsule.
        try:
            from merlin.development.targetgen.dram_facts import dram_window_for
            from merlin.verification.liveness import Program as _LProg
            from merlin.verification.liveness import assess as _lassess

            _lname = capsule.get("id") or capsule.get("name") or "kernel"
            # THE DRAM WINDOW IS A DERIVABLE FACT AND WAS NEVER HANDED OVER. The screen's address-map
            # check bounds a movement transaction against [base, base+size); nothing supplied `size`,
            # so EVERY report — including targets whose green card ships `start ~ end` — recorded
            # `dram-window-unknown` and the upper bound went unchecked. The window comes from the
            # target's OWN memory map (never a board default: fabricating a size would manufacture
            # false unmapped-address faults), and stays UNKNOWN with its reason named when the target
            # ships no map.
            _dwin = dram_window_for(eff_target)  # (base, size_bytes | None, provenance_why)
            _lrep = _lassess(
                _LProg(
                    name=_lname,
                    trace=trace,
                    address_model="pointer_args",
                    dram_bytes=_dwin[1],
                    dram_window_why=_dwin[2],
                ),
                eff_target,
            )
            (paths.generated / "liveness_report.json").write_text(
                json.dumps(_lrep.to_dict(), indent=2), encoding="utf-8"
            )
        except Exception as _le:  # noqa: BLE001 — advisory: our screener's limit is not the backend's defect
            import sys as _sys

            _sys.stderr.write(f"WARNING: liveness screen failed (advisory): {_le}\n")
        # Structural observations supplement numerical execution. Applicable trace
        # requirements gate qualification; numerical passes alone are not certificates.
        # The RoCC/inline-asm bare-metal harness passes each operand buffer as a POINTER argument, so
        # its address model is pointer_args: DRAM addresses must derive from the kernel's args, not be
        # baked. (A fixed-preload oracle — e.g. the program oracle — would pass "fixed_preload"; the
        # model is the harness's, derivable per oracle, not a target literal.)
        evidence.trace_check = _grade_cpu_step(
            TCK.check,
            trace,
            capsule.get("expected", {}),
            cb=evidence.command_buffer,
            address_model="pointer_args",
            # A design with a hardware loop performs several declared
            # classes with one instruction; its contract says which.
            subsumes=TCK.subsumption_for(eff_target),
            # Whose derived facts a declared residency mode reads.
            target=eff_target,
            lowered_module=evidence.parsed_lowered_module,
        )
        # Prohibited instruction roles on the executed stream: a hit fails the case at L2 here.
        _stream_policy = enforce_executed_stream(capsule, trace, target=eff_target, tiers=evidence.tiers)
        if _stream_policy is not None:
            evidence.trace_check["instruction_policy"] = _stream_policy
        # The MODE-FREE movement diagnostic, kept OUT of `violations` on purpose. It is a
        # whole-program performance observation, not a conformance one: `violations` feeds
        # `trace_all_pass`, which gates the run's formal verdict, and a program that moves one tile
        # per transfer is slow, not wrong. Reported under its own key so the author can see the
        # declared transfer width going unused without a performance note being able to fail a
        # correct capsule. Silent unless the target declares a payload bound.
        _movement = TCK.movement_findings(trace, TCK.movement_bound_for(eff_target))
        if _movement:
            evidence.trace_check["movement"] = _movement
        # Anti-cheese floor (must drive the accelerator) is the ONE trace-derived gate — but ONLY where
        # the RTL oracle will NOT run to prove it. When a required oracle tier executes the emitted
        # stream on real hardware, THAT is the spelling-independent anti-cheese (a host-computed fake
        # produces wrong output on RTL); the decoder-shaped floor only recognizes the `.insn r` form, so
        # gating on it here would false-fail a conformant `.word`-encoded kernel before the oracle.
        if (no_oracle or not required) and not TCK.drives_accelerator(trace):
            raise CertFailure(
                "trace_check",
                _cat("PROTOCOL_VIOLATION"),
                "no accelerator instructions emitted — the kernel did not drive the "
                "target's ISA (compute must happen on the device; correctness cannot "
                "come from the host).",
            )

        # A FORBIDDEN ACCELERATOR LANE, judged on the one thing this path can prove. `lane_report`
        # needs a routing plan and an execution record, which only the whole-model path owns, so an
        # op or model-slice capsule forbidding the accelerator had no verdict at all (see
        # LANE_CONTRACT_NOT_EVALUATED).
        #
        # The asymmetry is deliberate and is the whole reason this is sound here. A DECODED
        # accelerator instruction proves work went there: that is a violation, and it is a fail. Its
        # ABSENCE proves nothing -- the decoder recognizes the `.insn r` form, so a `.word`-encoded
        # kernel that does drive the device decodes as silent. Reading that silence as "the host
        # carried it" would hand a forbidding capsule a free pass, which is exactly the shape of
        # vacuous evidence this contract exists to refuse. So: violated, or unmeasured. Never
        # satisfied.
        if accelerator_lane_violated(capsule, trace):
            # PROTOCOL_VIOLATION is the enum member this is: the emitted stream broke a contract the
            # capsule declared. `_cat` would silently resolve an unknown name to RUNNER_CRASH, which
            # would report a compiler defect as a harness crash, so the plane carries the lane
            # vocabulary and the category stays one the enum actually has.
            raise CertFailure(
                "lanes",
                _cat("PROTOCOL_VIOLATION"),
                f"capsule forbids lane {_ACCELERATOR_LANE!r} and the emitted stream "
                f"decodes accelerator instructions; this target's capability manifest "
                f"does not admit this capsule's family, so the compiler must leave it "
                f"on the host lane.",
            )
