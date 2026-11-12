"""Capsule evaluation: inputs."""

from __future__ import annotations

import json

from merlin.development.backend.feedback import setup_timing as _host_timing
from merlin.evaluation.capsule.cpu import _grade_cpu_step
from merlin.evaluation.capsule.numerics import _bind_positional_output_name, _readout_epilogue_capabilities
from merlin.evaluation.capsule.source import _capture_compiler_source
from merlin.runtime import commandbuffer as _CB
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen.capsule_common import _cat, run_entrypoints
from merlin.targetgen.package_runtime import CertFailure


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
        from merlin.compiler.build import siblings as owned_shared_source

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
    from merlin.perf import lowering_obligation as _LOB

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
    from merlin.targetgen.readout_facet import epilogue_stage_routes
    from merlin.verification import epilogue_applicability as _EPI

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
            from merlin.runtime.commandbuffer import whole_program_entry_bindings as _entry_bindings

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
    from merlin.targetgen import capsule_dram as _dram
    from merlin.targetgen.dram_facts import dram_base_for
    from merlin.targetgen.target_experiment import load_capability_manifest

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
