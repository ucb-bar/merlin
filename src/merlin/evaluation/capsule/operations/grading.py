"""Capsule evaluation: inputs."""

from __future__ import annotations

import datetime as _dt
import json
import traceback as _traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.common.digest import readable_json
from merlin.development.targetgen.corpus.capsule_io import _cat, make_run_paths, run_entrypoints
from merlin.development.targetgen.grading import golden as CG
from merlin.development.targetgen.packaging.loading import BackendDeclined, CertFailure, InfraFailure, Package
from merlin.evaluation.capsule.evidence.certificates import _adapter_readback_policy, _resolve_oracle_adapters
from merlin.evaluation.capsule.evidence.source_custody import _capture_compiler_source
from merlin.evaluation.capsule.evidence.tiers import TierResult as TierResult
from merlin.evaluation.capsule.models.runtime import _grade_model_capsule, model_budget_seconds
from merlin.evaluation.capsule.operations import setup_timing as _host_timing
from merlin.evaluation.capsule.operations.numerics import (
    _absent_output_detail,
    _bind_positional_output_name,
    _grade_cpu_step,
    _merge_match_policy,
    _readout_epilogue_capabilities,
)
from merlin.evaluation.capsule.operations.numerics import _absent_outputs as _absent_outputs
from merlin.evaluation.capsule.operations.numerics import _unwritten_output_detail as _unwritten_output_detail
from merlin.evaluation.capsule.operations.oracle_ladder import run_oracle_ladder
from merlin.evaluation.capsule.operations.publication import check_trace, publish_operation
from merlin.runtime.engine import command_buffer as _CB
from merlin.runtime.engine.reference import reference_outputs
from merlin.runtime.engine.simulator import simulate
from merlin.target.registry.identity import _config_for_target

# ---- inputs ------------------------------------------------------------------------------------


def emit_program(capsule, package_dir, paths, cfg, contract, timeout, eff_target, pkg, evidence):
    evidence.compiler_source = _grade_cpu_step(_capture_compiler_source, package_dir, paths.generated)
    # shared front half: build + the 4 contract entrypoints (parse/target/cb/artifact), validated.
    with _host_timing.stage("emission"):
        pkg, evidence.command_buffer, evidence.lowered_text = run_entrypoints(
            pkg,
            package_dir,
            capsule,
            paths,
            contract=contract,
            timeout=timeout,
            fourth_output_name=cfg.fourth_output_name,
        )
    if capsule.get("owned_host_source") is not None:
        import hashlib

        source_receipt_path = paths.generated / "owned-source" / "receipt.json"
        source_receipt = json.loads(source_receipt_path.read_text())
        from merlin.compiler.build import owned_sources as owned_shared_source

        source_receipt_schema = (
            owned_shared_source.RECEIPT
            if owned_shared_source.selected(capsule)
            else "merlin.owned_spatial_source_compilation.v1"
        )
        if (
            source_receipt.get("schema") != source_receipt_schema
            or source_receipt.get("artifact_sha256") != hashlib.sha256(evidence.lowered_text.encode()).hexdigest()
        ):
            raise ValueError("owned source compilation product join changed")
        evidence.owned_source_compilation = {
            "owned_source_compilation": {
                "receipt": str(source_receipt_path),
                "sha256": hashlib.sha256(source_receipt_path.read_bytes()).hexdigest(),
                "source_sha256": source_receipt["source_sha256"],
                "artifact_sha256": source_receipt["artifact_sha256"],
                "scope": "original public host graph; ordinary capsule gates apply",
                "whole_model_gate_credit": False,
            }
        }

        # ONE DECLARED STIMULUS RANGE, READ BY ALL THREE SIDES. The golden materializes from the
        # capsule (capsule_golden.materialize_capsule_leaves) while the reference interpreter and the
        # device harness materialize from the command buffer (commandbuffer.materialize_inputs), and
        # they agreed only because both used the same hardcoded default. Propagating the capsule's own
        # declaration into the buffer is what lets a capsule choose a range at all -- and a capsule
        # that needs one is not hypothetical: with the default 0..3 every operand is non-negative, so
        # an accumulator is never negative and a fused ReLU is the identity on every value the program
        # can produce. B1_linear_relu_i8 passes whether or not its activation is applied.
        #
        # A capsule that declares nothing leaves the buffer untouched, so every existing capsule's
        # stimulus, golden and emitted harness are byte-identical.
    _stim_lo, _stim_hi = CG.capsule_stimulus_range(capsule)
    if (_stim_lo, _stim_hi) != _CB.DEFAULT_STIMULUS_RANGE:
        evidence.command_buffer.setdefault("params", {})[_CB.STIMULUS_RANGE_KEY] = [_stim_lo, _stim_hi]

        # A PERFORMANCE MEMBER MUST EMIT THE TRANSFORMATION ITS CYCLES ARE ATTRIBUTED TO.
        #
        # The ABI lets a buffer declare an operand the HARNESS derives from another (a conv's im2col
        # gather). That is legitimate for STIMULUS -- reference, simulator and device must be handed one
        # byte-identical tensor -- and it is untouched for a functional capsule, which is answering
        # whether the arithmetic is right. It is NOT legitimate for a member whose verdict is a cycle
        # count: the gather then costs the emitted program nothing, so the lowering that would perform it
        # never appears in what is measured and the task contract's own rule ("a lever that does not
        # change the emitted code is inert by definition") makes it unreachable as a lever.
        #
        # Scoped by the capsule's own `performance` block, so it fires for every performance member and
        # for no functional one; and derived from that block's PRESENCE rather than from a declared
        # field, so it cannot be switched off by editing a capsule. An obligation this build cannot
        # evaluate is refused too -- never silently satisfied. Judged HERE, right after the compiler's
        # own entrypoints and before the golden and the oracle ladder: a member measuring a program
        # that never did the work is refused before anything is spent measuring it.
    from merlin.analysis.optimization.candidates import lowering_obligation as _LOB

    _lowering = _LOB.assess(capsule, evidence.command_buffer)
    if _lowering.get("status") in _LOB.REFUSING_STATUSES:
        raise CertFailure("lowering_obligation", _cat("PROTOCOL_VIOLATION"), str(_lowering.get("detail")))

        # AN EPILOGUE THE READOUT DOES NOT APPLY IS A WRONG ANSWER, NOT A MISSED OPTIMIZATION.
        #
        # Measured: a capsule declaring `output_dtype='i32' epilogue=['relu']` returned 126 of 256
        # outputs negative (min -85), which is the raw accumulator, while the command-buffer numeric
        # floor and trace both passed -- the compiler's intent was right and the device discarded the
        # activation. Judged here for the same reason as the obligation above: before anything is
        # spent grading a program that computes something other than what it declares.
        #
        # The rule is target-independent and lives in merlin.verification; the CAPABILITY is the target's
        # own declaration. A backend that declares no readouts leaves the check UNAVAILABLE rather
        # than refusing every capsule -- an undeclared target is not a broken one, and the reason is
        # recorded so "not checked" never reads as "checked and fine".
    from merlin.target.capabilities.readout_derivation import epilogue_stage_routes
    from merlin.verification.references import epilogue_applicability as _EPI

    _declared = _readout_epilogue_capabilities(eff_target)
    if _declared:
        _epi = _EPI.assess(
            evidence.command_buffer,
            [
                _EPI.ReadoutCapability(
                    str(r["selector"]),
                    frozenset(str(x) for x in r.get("applies") or ()),
                    str(r.get("evidence") or ""),
                )
                for r in _declared
            ],
            routes=epilogue_stage_routes(eff_target),
        )
        evidence.epilogue_applicability = _epi.to_dict()
        if _epi.refusing:
            raise CertFailure("epilogue_applicability", _cat("PROTOCOL_VIOLATION"), str(_epi.detail))
    else:
        evidence.epilogue_applicability = {
            "schema": "merlin_epilogue_applicability_v1",
            "status": "unavailable",
            "detail": (
                f"target {eff_target!r} declares no readout epilogue capability, so "
                f"whether its readouts apply a declared stage cannot be established"
            ),
            "refusing": False,
        }

    return pkg


def bind_program_inputs(capsule, paths, eff_target, evidence, inputs):
    capsule_dir = capsule.get("__dir__")
    inputs.independent_float = CG.is_independent_float_golden(capsule, capsule_dir)
    evidence.whole_program = (evidence.command_buffer.get("kernel_abi") or {}).get("kind") == "whole_program"
    # Both cases require the same evidence shape for different reasons.  A float program cannot be
    # interpreted by the integer command-buffer engines; a whole-program kernel deliberately uses
    # its command list as the ACCELERATOR projection, while scalar-host regions live only in the
    # submitted ELF.  In either case the complete ELF must be run and compared directly with the
    # capsule's independent semantic golden.  This is stricter than inventing fake commands for host
    # work: no command projection can pass the kernel, and no harness-side arithmetic contributes to
    # the answer.
    inputs.direct_oracle_golden = inputs.independent_float or evidence.whole_program
    inputs.golden_source = CG.golden_source(capsule, capsule_dir)
    inputs.gold = _grade_cpu_step(CG.golden, capsule, capsule_dir)
    # A linalg-on-tensors reference lowering names its output positionally ("out"); the readback base +
    # golden compare are keyed by the capsule's DECLARED output name (the merlin_iface grammar already
    # names it via the commit op). Rename the sole output leaf to the declared name so preload, kernel,
    # readback, and golden all agree on one name+address. Scoped to linalg_positional cbs (no-op else).
    _bind_positional_output_name(evidence.command_buffer, capsule)
    # The semantic roster comes from the independent golden; the public logical ABI alone
    # determines which additional published/intermediate console readbacks are legitimate.
    # Apply the same boundary to ordinary and whole-program kernels and every numeric tier.
    from merlin.compiler.ir.logical_abi import HarnessRenderError, logical_abi, logical_interface

    try:
        readback_outputs = tuple(
            buf.name for buf in logical_interface(evidence.command_buffer, logical_abi()) if buf.kind == "output"
        )
    except HarnessRenderError as ce:
        raise CertFailure("command_buffer", _cat("INTERFACE_MISMATCH"), str(ce)) from ce

    inputs.output_contract = {
        "required_outputs": tuple(inputs.gold),
        "readback_outputs": readback_outputs,
        "golden_source": inputs.golden_source,
    }

    # A program-oracle (self-hosted-ISA) target must run its emitted kernel on the SAME operands the
    # independent golden used. Attach the capsule's canonical leaf-input bytes (golden.yaml raws) to
    # the cb's leaf tensors so the program oracle preloads them by cb-declared base (AW5). Keyed by
    # name; a no-op for integer capsules (no recorded raws) and for adapters that ignore the field.
    _raws = CG.canonical_input_raws(capsule, capsule_dir)
    if _raws:
        import base64 as _b64

        for _tname, _tspec in (evidence.command_buffer.get("tensors") or {}).items():
            if _tspec.get("role") in ("input", "weight", "bias") and _tname in _raws:
                _tspec["preload_b64"] = _b64.b64encode(_raws[_tname]).decode()

                # Attach the golden's DECODED operand values so a self-hosted kernel harness (the fork-free SIMT
                # path) can embed them and run the emitted kernel on the SAME operands the independent golden used.
                # Generic + additive: empty for integer capsules, ignored by adapters that grade the command buffer.
                # A block-scaled MX capsule's operands (quantized codes + corpus-seeded E8M0 block scales) live only
                # in the golden and cannot be rebuilt from the decoded floats — attach them so the reference MX
                # kernel can bake them. Public-capsule reference path (masked for hidden); a no-op for non-MX goldens.
    _mxops = CG.mx_operands(capsule, capsule_dir)
    if _mxops:
        evidence.command_buffer["mx_operands"] = _mxops

    _vals = CG.canonical_input_values(capsule, capsule_dir)
    if evidence.whole_program and not inputs.independent_float:
        # Materialize the golden's exact leaf stimulus: captured host-eager bytes for an
        # exact source-bound integer slice, deterministic leaves for synthetic capsules.
        # Whole-program lowering may rename leaves positionally (A0 -> arg0), so attach the exact
        # semantic stimulus before re-keying it below; otherwise the harness materializes from the
        # new positional names and evaluates the right program on different data.
        _vals = CG.materialized_input_values(capsule)
    elif not _vals and evidence.whole_program:
        # Complete-kernel adapters always need an explicit semantic stimulus. This fallback covers
        # an independent golden whose format carries no decoded input table.
        _vals = CG.materialized_input_values(capsule)
    if _vals:
        evidence.command_buffer["canonical_inputs"] = _vals
        # POSITIONAL FALLBACK for interface grammars whose operands are UNNAMED. The merlin_iface
        # grammar names each leaf (``name = "X"``) so the by-name attach above matches; the
        # linalg-on-tensors grammar has positional ``func.func @forward`` args (``%0``, ``%1`` …), so a
        # lowering names its leaves positionally and NONE match the canonical keys (the capsule's
        # arg_order). When no canonical key is a cb tensor name, re-key by POSITION: the capsule's
        # arg_order and the cb's non-output leaves are both in @forward-argument order, so zip them.
        _tensors = evidence.command_buffer.get("tensors") or {}
        if not (set(_vals) & set(_tensors)):
            from merlin.runtime.engine.command_buffer import whole_program_entry_bindings as _entry_bindings

            _leaves = _entry_bindings(evidence.command_buffer)
            if _leaves is None:
                _leaves = [n for n, s in _tensors.items() if s.get("role") in ("input", "weight", "bias")]
            _ordered = list(_vals.values())
            if len(_leaves) == len(_ordered):
                evidence.command_buffer["canonical_inputs"] = dict(zip(_leaves, _ordered))
                if _raws:
                    import base64 as _b64

                    for _leaf, _rb in zip(_leaves, list(_raws.values())):
                        _tensors[_leaf]["preload_b64"] = _b64.b64encode(_rb).decode()

    if CG.is_exact_pytorch_integer_source(capsule):
        # Reference, simulator, RTL and whole-program harness all consume the same
        # captured bytes as the independently checked Torch output.
        try:
            CG.bind_exact_integer_stimulus(capsule, evidence.command_buffer)
        except ValueError as exc:
            raise CertFailure("canonical_input_binding", _cat("PROTOCOL_VIOLATION"), str(exc)) from exc

            # Fixed-address oracles need a shared canonical layout. A selected
            # logical-pointer harness instead allocates the original tensors and
            # passes their actual addresses, without competing fixed-base metadata.
    from merlin.development.targetgen.grading import golden_inputs as _dram
    from merlin.target.isa.machine_model import dram_base_for
    from merlin.target.registry.experiment_descriptor import load_capability_manifest

    # Place any harness-assigned canonical base INSIDE the target's DRAM region [dram_base, +size):
    # a self-hosted-ISA target maps DRAM at a nonzero region base (derived from its facts; 0 for a
    # 0-based target, keeping the base at DEFAULT_BASE unchanged), and the L2 oracle
    # relocates by that same base — so a submission that omits a base still grades against a layout
    # the model can index. Bases the agent DECLARED are left untouched (inject_bases only fills gaps).
    _dram.inject_bases(
        evidence.command_buffer,
        capsule,
        base=dram_base_for(eff_target) + _dram.DEFAULT_BASE,
        contract=load_capability_manifest(eff_target).contract,
    )


# ---- records -----------------------------------------------------------------------------------
# Observed operation artifacts and numerical reference values passed between grader stages.
#
# These records preserve partial evidence when a later stage refuses or raises.
# They carry observations; they do not select target semantics or acceptance policy.


@dataclass
class ProgramEvidence:
    command_buffer: dict | None = None
    lowered_text: str | None = None
    compiler_source: dict | None = None
    owned_source_compilation: dict = field(default_factory=dict)
    whole_program: bool = False
    tiers: dict = field(default_factory=dict)
    trace_check: dict = field(default_factory=lambda: {"status": "skipped", "violations": []})
    decoded_trace: dict | None = None
    parsed_lowered_module: Any = None
    numeric: dict = field(default_factory=lambda: {"status": "skipped"})
    execution_tiers: set[str] = field(default_factory=set)
    executability: dict = field(default_factory=dict)
    epilogue_applicability: dict = field(default_factory=dict)
    completion_order: dict | None = None
    order_check: dict | None = None
    status: str = "pass"
    failure: dict | None = None
    declined: dict | None = None


@dataclass
class ReferenceData:
    independent_float: bool = False
    direct_oracle_golden: bool = False
    golden_source: str | None = None
    gold: dict | None = None
    reference: dict | None = None
    simulation: dict | None = None
    output_contract: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OracleAttempt:
    completed: bool = False
    result: dict | None = None
    elapsed_s: float | None = None
    carried_built: bool = False


@dataclass(frozen=True)
class OracleMeasurement:
    passed: bool
    mismatch_reason: str
    simulator: str
    oracle: Any


# ---- reference ---------------------------------------------------------------------------------
# Capsule evaluation: reference.


def check_reference(capsule, paths, policy, required, evidence, inputs):
    if inputs.independent_float:
        # The integer reference/simulate engines cannot execute a float (fp8/bf16) datapath, so the
        # integer L0-reference / L1-sim numeric floor is INAPPLICABLE — skipped honestly (not failed),
        # exactly as a non-RoCC target drops the RoCC trace gate. The real numeric grade rides the RTL
        # program-oracle output vs the independent golden.yaml (tolerance_float), scored in the oracle
        # loop below. These skips are marked not_applicable so not_run_is_not_pass does NOT flip them to
        # incomplete (only an unavailable/absent RTL oracle does).
        inputs.reference = inputs.simulation = None
        evidence.numeric = {
            "status": "skipped",
            "policy": policy.get("compare"),
            "golden_source": inputs.golden_source,
            "note": "integer reference/simulate N/A for a float datapath; correctness is graded "
            "by running your artifact on the RTL oracle and checking it computes the "
            "declared operation within tolerance",
        }
        evidence.tiers["L0"] = TierResult(
            "L0",
            "skipped",
            mandatory="L0" in required,
            not_applicable=True,
            reason="integer reference not applicable to float datapath; graded vs independent "
            f"golden ({inputs.golden_source}) at the RTL oracle",
        )
        evidence.tiers["L1"] = TierResult(
            "L1",
            "skipped",
            mandatory="L1" in required,
            not_applicable=True,
            reason="integer simulate not applicable to float datapath",
        )
    elif evidence.whole_program:
        # The command list remains available for structural checks, accelerator work accounting,
        # and decoded-trace correspondence.  It is not a semantic model of the CPU regions between
        # accelerator calls.  Calling reference_outputs here would materialize those intermediates
        # as unrelated deterministic leaves and compare a different program.  Correctness is gated
        # below by executing the complete submitted ELF at every required device tier.
        inputs.reference = inputs.simulation = None
        evidence.numeric = {
            "status": "skipped",
            "policy": policy.get("compare"),
            "golden_source": inputs.golden_source,
            "note": (
                "command-buffer reference/simulate are inapplicable to an explicitly declared "
                "whole-program kernel: its command list projects accelerator regions only; "
                "correctness is graded by executing the complete submitted ELF and comparing "
                "its declared outputs with the capsule golden"
            ),
        }
        evidence.tiers["L0"] = TierResult(
            "L0",
            "skipped",
            mandatory="L0" in required,
            not_applicable=True,
            reason="command-buffer reference cannot interpret scalar regions of a whole-program ELF",
        )
        evidence.tiers["L1"] = TierResult(
            "L1",
            "skipped",
            mandatory="L1" in required,
            not_applicable=True,
            reason="command-buffer simulate cannot execute scalar regions of a whole-program ELF",
        )
    else:
        # Interpreting the AGENT's command buffer (reference/simulate) can fail if the cb is
        # structurally invalid for the reference interpreter's op/shape model. That is the agent's
        # bug, NOT a harness crash: report it as a gradeable command_buffer failure (so the agent gets
        # feedback and both arms are scored identically) instead of a RUNNER_CRASH. The reason reports
        # the ACTUAL exception rather than asserting one hardcoded cause — the interpreter models a
        # fixed op vocabulary, so an unmodeled op/operand-shape/name is one of several possible causes.
        try:
            inputs.reference = _grade_cpu_step(reference_outputs, evidence.command_buffer)
            inputs.simulation = _grade_cpu_step(simulate, evidence.command_buffer)["outputs"]
        except (ValueError, KeyError, IndexError, TypeError) as ce:
            raise CertFailure(
                "command_buffer",
                _cat("STRUCTURAL_INVARIANT_VIOLATION"),
                f"command buffer could not be interpreted by reference/simulate "
                f"({type(ce).__name__}: {ce}); check operand ranks/shapes and that each command's op "
                f"is one the reference interpreter models (e.g. a windowed op lowered to a 2D matmul)",
            ) from ce
        nrep = _grade_cpu_step(
            CG.compare_declared_outputs,
            inputs.gold,
            inputs.reference,
            capsule["numeric_policy"],
            **inputs.output_contract,
        )
        evidence.numeric = {
            "status": nrep["status"],
            "policy": nrep["policy"],
            "max_abs_diff": nrep["max_abs_error"],
            "max_rel_error": nrep["max_rel_error"],
            "mismatch_count": nrep["mismatch_count"],
            "first_mismatch": nrep["first_mismatch"],
            # same DROPPED-store surfacing as the float path (see _absent_outputs): a missing
            # declared output is reported distinctly, never as a phantom 0-magnitude mismatch.
            "per_output": nrep.get("per_output", {}),
            "missing_outputs": _absent_outputs(nrep),
        }
        CG.write_numeric_report(paths.generated / "numeric_report.yaml", nrep)
        evidence.tiers["L0"] = TierResult(
            "L0",
            "pass" if nrep["status"] == "pass" else "fail",
            mandatory="L0" in required or True,
            reason=(
                None if nrep["status"] == "pass" else "your command buffer does not compute the declared operation"
            ),
            evidence="numeric_report.yaml",
        )
        if nrep["status"] != "pass":
            _absent_cb = _absent_output_detail(
                nrep, "command_buffer", inputs.gold, inputs.reference
            ) or _unwritten_output_detail(nrep, "command_buffer")
            raise CertFailure(
                "numeric_golden",
                _cat("FUNCTIONAL_MISMATCH"),
                _absent_cb
                or (
                    "your command buffer does not compute the declared operation "
                    f"(first divergence at index={(nrep['first_mismatch'] or {}).get('index')})"
                ),
            )

        l1_ok = (
            _grade_cpu_step(
                CG.compare_declared_outputs, inputs.reference, inputs.simulation, policy, **inputs.output_contract
            )["status"]
            == "pass"
        )
        evidence.tiers["L1"] = TierResult(
            "L1",
            "pass" if l1_ok else "fail",
            mandatory=True,
            reason=None if l1_ok else "reference(cb) != simulate(cb)",
        )
        if not l1_ok:
            raise CertFailure("command_buffer_reference", _cat("FUNCTIONAL_MISMATCH"), "reference(cb) != simulate(cb)")


# ---- operation ---------------------------------------------------------------------------------
# Capsule evaluation: operation.


def run_capsule(
    capsule: dict,
    package_dir: str | Path,
    *,
    runs_root: str | Path,
    run_id: str | None = None,
    contract: str | Path | None = None,
    oracle_adapters: dict[str, Callable] | None = None,
    pkg: Package | None = None,
    timeout: int = 600,
    target: str | None = None,
    suite: str | None = None,
    dtype: str = "i8xi8_i32",
    config=None,
    perf_extractor: Callable | None = None,
    no_oracle: bool = False,
    workers: int | None = None,
    adapter_managed_tiers: tuple[str, ...] = (),
    model_execution_selection: dict | None = None,
    checked_source_diagnostic=None,
    model_budget_s: float | None = None,
    _defer_result_publication: bool = False,
) -> dict:
    """Run the original target-selected package/capsule ladder and full gates.

    ``model_execution_selection`` selects exact public facts/timer data, never
    live package/resource authority, protected access or waived numeric tiers.
    ``adapter_managed_tiers`` bypass carried certificates for supplied adapters.
    ``workers`` records actual caller fanout; absent means unstated.
    Missing engines remain unavailable; structural counts supply no RTL timing.
    """
    from merlin.development.targetgen.packaging.records import toolchain_shas

    # The effective target comes from the config (authoritative — cfg.target drives the run) when one is
    # supplied, else the explicit ``target`` argument. If NEITHER is given we refuse to run rather than
    # silently defaulting to one target (the OV2 rule: no core path silently operates on a default target).
    eff_target = config.target if config is not None else target
    if eff_target is None:
        raise ValueError("run_capsule requires a target (or a config carrying one); no default target is assumed")
    cfg = config or _config_for_target(eff_target, suite, dtype)
    # L1/oracle output equality uses the capsule's numeric_policy (integer -> exact). A float/SIMT target
    # ALSO declares a force_match_policy (a global float tolerance) so an integer-policy capsule is still
    # graded with tolerance — but the global force must NOT DISCARD the capsule's OWN declared tolerance:
    # a bf16 / low-precision capsule legitimately declares a looser atol/rtol than the global f32 default,
    # and letting the tight global override it makes a CONFORMANT kernel unpassable (RP3). Merge: keep the
    # force policy's compare MODE, take the LOOSER (max) atol/rtol per field.
    policy = _merge_match_policy(cfg.force_match_policy, capsule.get("numeric_policy"))
    name = capsule["name"]
    run_id = run_id or f"{name}"
    # An unrouted grade (oracle_adapters=None) resolves to the TARGET'S OWN endpoint oracle from the
    # contract — never a fixed experiment's simulator pair. `{}` stays honest no-oracle (L0/L1/trace only).
    adapters = oracle_adapters if oracle_adapters is not None else _resolve_oracle_adapters(eff_target)
    if set(adapter_managed_tiers) - set(adapters or {}):
        raise ValueError("adapter-managed tiers require supplied oracle adapters")
    required = set(capsule.get("required_oracle_tiers", []))

    paths = make_run_paths(runs_root, run_id, suite=cfg.suite, target=cfg.target, dtype=cfg.dtype, benchmark=name)

    # Whole-model capsule: graded end to end by compiling the captured model through the merlin
    # whole-model flow (compile_rvv) and gating vs its golden — NOT the per-op tier ladder. Write the
    # same capsule_result.json shape so downstream reporting is uniform.
    if model_execution_selection is not None:
        from merlin.development.targetgen.execution.host.public_program_timing import verify_selection

        verify_selection(capsule, model_execution_selection)
    if checked_source_diagnostic is not None:
        from merlin.development.targetgen.execution.host.checked_source_diagnostics import (
            verify_host_selection,
        )

        verify_host_selection(
            checked_source_diagnostic, capsule, target=eff_target, public_selection=model_execution_selection
        )
    from merlin.development.targetgen.execution.host.selections import selected_source_slice

    if capsule.get("kind") == "model" or selected_source_slice(capsule):
        if _defer_result_publication:
            raise ValueError("operator publication cannot replace whole-model qualification")
            # LEAVE A TRACE BEFORE THE LONG PART. A whole-model grade can legitimately run for hours, and
            # with nothing written until it finishes an operator cannot tell a working run from a wedged
            # one -- which is exactly how a 5h30m stall was read as progress. Written first, so the run
            # directory says what started, when, and under what ceiling.
        _budget = model_budget_seconds() if model_budget_s is None else model_budget_seconds(override=model_budget_s)
        if selected_source_slice(capsule):
            _budget = timeout if _budget is None else min(_budget, timeout)
        if checked_source_diagnostic is not None and (not _budget or _budget <= 0):
            raise ValueError("checked source diagnostic requires the original sealed worker hard deadline")
        paths.run_path.mkdir(parents=True, exist_ok=True)
        (paths.run_path / "model_grade_started.json").write_text(
            json.dumps(
                {
                    "capsule": name,
                    "started_at": _dt.datetime.now(_dt.UTC).isoformat(),
                    "model_budget_s": _budget,
                    "step_timeout_s": timeout,
                    "note": "whole-model capsule; absent capsule_result.json means it is still running or was killed",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        from merlin.development.targetgen.execution.host.selections import owned_model_required

        if (
            owned_model_required(capsule, target=eff_target)
            or model_execution_selection is not None
            or checked_source_diagnostic is not None
        ):
            from merlin.evaluation.capsule.models.runtime import grade_selected_model

            result = grade_selected_model(
                capsule,
                target=eff_target,
                package_dir=package_dir,
                paths=paths,
                runs_root=runs_root,
                run_id=run_id,
                cfg=cfg,
                contract=contract,
                pkg=pkg,
                adapters=adapters,
                budget_s=_budget,
                timeout=timeout,
                model_execution_selection=model_execution_selection,
                checked_source_diagnostic=checked_source_diagnostic,
            )
        else:
            result = _grade_model_capsule(
                capsule, target=eff_target, timeout=timeout, package_dir=package_dir, budget_s=_budget
            )
            # This model verdict does not independently price command-buffer work.
            # A candidate has one submitted whole-program buffer; the legacy graph
            # has many. Neither absence may be mistaken for a zero-work claim.
        from merlin.analysis.programs.work_volume import command_buffer_evidence as _cb_evidence

        volume_scope = (
            "candidate whole-program buffer; volume not independently priced here"
            if bool((capsule.get("semantic") or {}).get("must_accelerate"))
            else "whole-model compilation (no single command buffer)"
        )
        _mw, _ma = _cb_evidence(None, compiler_provenance=volume_scope)
        result.setdefault("work_volume", _mw)
        result.setdefault("command_buffer_artifact", _ma)
        result.setdefault("toolchain_shas", toolchain_shas(eff_target))
        (paths.run_path / "capsule_result.json").write_text(readable_json(result), encoding="utf-8")
        # Persist the ARR coverage certificate as its own durable artifact (not only inside
        # capsule_result.json) so the report/grader can read it back per compilation.
        cert = result.get("coverage_certificate")
        if cert is not None:
            paths.generated.mkdir(parents=True, exist_ok=True)
            (paths.generated / "coverage_certificate.json").write_text(json.dumps(cert, indent=2), encoding="utf-8")
        census = result.get("placement_census")
        if census is not None:
            paths.generated.mkdir(parents=True, exist_ok=True)
            (paths.generated / "placement_census.json").write_text(json.dumps(census, indent=2), encoding="utf-8")
        return result

    from merlin.development.targetgen.execution.host.leaf_execution import (
        grade_owned_host_leaf,
        selected_host_leaf,
    )

    if selected_host_leaf(capsule):
        if _defer_result_publication:
            raise ValueError("operator publication cannot replace the owned-host route")
        return grade_owned_host_leaf(
            capsule,
            target=eff_target,
            paths=paths,
            cfg=cfg,
            run_id=run_id,
            timeout=timeout,
            no_oracle=no_oracle,
            readback_policy=_adapter_readback_policy(adapters),
            reachable_oracle_tiers=frozenset(adapters),
        )

    evidence = ProgramEvidence()
    inputs = ReferenceData()

    try:
        pkg = emit_program(capsule, package_dir, paths, cfg, contract, timeout, eff_target, pkg, evidence)

        # --- golden + L0/L1 -----------------------------------------------------------------
        # The golden is the INDEPENDENT oracle's answer. An exact source-bound PyTorch integer
        # slice is recomputed on its captured typed operands and cross-checked against host eager.
        # Other integer capsules recompute on their declared synthetic stimulus. For a
        # float capsule that ships an independent golden.yaml (e.g. fp8-e4m3 -> bf16, golden_source
        # specir_refmodel_fp8_bf16), the integer engine cannot reproduce the float datapath, so the golden
        # is READ from golden.yaml — resolved by policy+source, never a target name.
        bind_program_inputs(capsule, paths, eff_target, evidence, inputs)

        check_reference(capsule, paths, policy, required, evidence, inputs)

        # Bound whether or not the trace gate runs: `None` means "nobody parsed these bytes here",
        # which the plan evidence answers by parsing them itself.
        check_trace(capsule, paths, cfg, contract, eff_target, required, no_oracle, evidence)

        # --- oracle tiers -------------------------------------------------------------------
        # Run every tier the config declares (tier_sim ladder) OR an injected adapter provides — so a
        # target whose RTL tier is supplied by an adapter rather than a static tier_sim (e.g. arc L3 with
        # an empty tier_sim) still runs, while a declared ladder is unchanged. The L0/L1 math floor
        # is handled above (reference/simulate), NOT here. Sorted for a stable L2<..<L5 ladder order.
        # program_oracle (the external_backend adapters) raises its OWN OracleUnavailable, a RuntimeError
        # that is a DIFFERENT class from this module's OracleUnavailable(Exception) — unrelated by mro. If
        # we catch only the local one, an honest "oracle unavailable" from the program oracle (missing
        # assembler / model venv / cosim import) falls through to the generic handler and is mislabeled a
        # TOOL_CRASH, so the unavailable->incomplete / not_gradeable_no_oracle path never fires for a self-hosted ISA.
        # Catch BOTH so unavailability is honest for every endpoint.
        run_oracle_ladder(
            capsule,
            package_dir,
            paths,
            cfg,
            contract,
            timeout,
            eff_target,
            target,
            name,
            run_id,
            adapters,
            required,
            policy,
            workers,
            adapter_managed_tiers,
            perf_extractor,
            evidence,
            inputs,
        )

    except BackendDeclined as bd:
        # A STATED REFUSAL IS NOT A WRONG ANSWER. It is still not a pass -- the capsule stays in the
        # denominator, uncertified -- but recording it as a numeric mismatch told the agent its
        # arithmetic was wrong about a program it had never emitted, and no feedback could say which
        # shapes it had declined.
        evidence.status = "declined"
        evidence.declined = bd.to_dict()
        evidence.failure = {"plane": "backend_declined", "category": "DECLINED", "detail": bd.reason}
    except InfraFailure as inf:
        # The harness could not stage this capsule's declared inputs, so NOTHING about the submission was
        # measured. `fail` would be a lie in the agent's direction and `pass` a lie in ours; this is the
        # third thing, and it is excluded from the pass/fail denominators (NOT_MEASURED_STATUSES) while
        # `capsule_grade.grade` refuses to call the run gradeable at all. Caught BEFORE CertFailure
        # because InfraFailure IS one -- the handler order is the whole mechanism, and with the narrower
        # clause absent a staging outage was recorded as 31 of 33 structurally invalid SUBMISSIONS.
        evidence.status = "infrastructure_fault"
        cat = inf.category.value if hasattr(inf.category, "value") else str(inf.category)
        evidence.failure = {"plane": inf.plane, "category": cat, "detail": inf.detail}
    except CertFailure as cf:
        evidence.status = getattr(cf, "status", "fail")  # a NotMeasuredFailure names its not-measured status
        cat = cf.category.value if hasattr(cf.category, "value") else str(cf.category)
        evidence.failure = {"plane": cf.plane, "category": cat, "detail": cf.detail, **getattr(cf, "facts", {})}
    except Exception as e:  # internal harness bug
        evidence.status = "error"
        evidence.failure = {
            "plane": "runner_internal",
            "category": "RUNNER_CRASH",
            "detail": f"{type(e).__name__}: {e}",
            "traceback": _traceback.format_exc(),
        }

        # WHAT WORK THIS RESULT'S CYCLES BOUGHT, said by the result itself. A performance consumer
        # attributes a member's cycles to a compute axis it derives from the emitted command buffer, and
        # it needs both halves: the counted work AND the raw buffer under the digest they agree on, so it
        # can recount the program and refuse the pair when the two disagree. Emitted for EVERY graded
        # capsule, including one whose entrypoints never produced a buffer -- that case carries an
        # explicit UNKNOWN (`exact_macs: null` plus the counter's refusals), never a zero and never an
        # absent key. An absent key reads as "no compute axis applies here"; a zero reads as "this
        # program does no work", which on a perf bench means "infinitely fast". Both are lies.
    return publish_operation(
        capsule,
        package_dir,
        paths,
        cfg,
        contract,
        eff_target,
        name,
        run_id,
        required,
        no_oracle,
        _defer_result_publication,
        evidence,
    )
