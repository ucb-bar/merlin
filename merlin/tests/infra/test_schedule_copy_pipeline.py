"""Ordinary fixed-slot copy motion through actual final bytes; neutral machines only."""

import struct
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import test_copy_pipeline as leaf

from merlin.compile.scheduling.program_limits import (
    AddressWindow,
    DecodedInstruction,
    EncodedAccess,
    ProgramDecoder,
    ProgramLimits,
)
from merlin.sched.ir import Const, evaluate


@pytest.fixture
def modules():
    from merlin.compile.scheduling import copy_pipeline, schedule_pipeline

    return schedule_pipeline, copy_pipeline


def project(call):
    effect = leaf.project(call)
    values = tuple(evaluate(call.arg(k), {}) for k in ("width", "host", "token", "view"))
    return replace(effect, semantic_key=repr((call.instr, values))) if effect is not None else None


def rewrite(call, accesses):
    if not accesses:
        return call
    local = next(access for access in accesses if access.space in ("bytes", "words"))
    return replace(
        call, args=tuple((key, Const(local.start) if key == "address" else value) for key, value in call.args)
    )


def setup(modules, *, width=7, panels=7, slots=2, alias=True, word_format="<BQQQQQ"):
    pipeline, copies = modules
    kernel, objects, banks, admission, model = leaf.fixture(width=width, panels=panels, slots=slots, alias=alias)
    admission = copies.CopyPipelineAdmission(**asdict(admission))
    adapter = replace(leaf.ADAPTER, project=project, rewrite=rewrite)
    word = struct.Struct(word_format)
    names = ("setup", "fetch", "await", "affine", "store", "descriptor")

    def encode(kernel):
        return b"".join(
            word.pack(
                names.index(call.instr),
                *(evaluate(call.arg(k), {}) for k in ("address", "width", "host", "token", "view")),
            )
            for call in kernel.body
        )

    def decode_calls(image):
        calls = []
        for offset in range(0, len(image), word.size):
            opcode, a, n, h, token, view = word.unpack_from(image, offset)
            if opcode >= len(names) or view not in (0, 1):
                raise ValueError("unknown actual instruction or memory selector")
            calls.append(leaf.op(names[opcode], address=a, width=n, host=h, token=token, view=view))
        return tuple(calls)

    def decode_program(image):
        pending, instructions = set(), []
        for index, call in enumerate(decode_calls(image)):
            v = dict((k, evaluate(value, {})) for k, value in call.args)
            a, n, h, token = (v[k] for k in ("address", "width", "host", "token"))
            local = ("bytes", "words")[v["view"]]
            accesses, completion = (), "synchronous"
            if call.instr == "fetch":
                accesses = (EncodedAccess("input", h, n), EncodedAccess(local, a, n, True))
                pending.add(token)
                completion = "asynchronous"
            elif call.instr == "await":
                if token not in pending:
                    raise ValueError("actual token wait without launch")
                pending.remove(token)
                completion = "synchronous" if pending else "drain"
            elif call.instr == "affine":
                accesses = (EncodedAccess(local, a, n), EncodedAccess(local, a, n, True))
            elif call.instr == "store":
                accesses = (EncodedAccess(local, a, n), EncodedAccess("output", h, n, True))
            instructions.append(
                DecodedInstruction(index * word.size, word.size, accesses, completion=completion, relocatable=True)
            )
        if pending:
            raise ValueError("actual return before completion")
        return tuple(instructions)

    codec = pipeline.ScheduleEncoding(
        encode,
        decode_calls,
        ProgramDecoder(decode_program, adapter.configuration_identity, "independent-neutral-token-decoder"),
        adapter.numerical_policy_identity,
        "neutral-byte-codec",
    )
    limits = ProgramLimits(
        word.size * len(kernel.body),
        len(kernel.body),
        1,
        (*model["external_windows"], *(AddressWindow(b.name, 0, b.capacity) for b in banks)),
        (),
    )
    return kernel, objects, banks, adapter, codec, limits, admission, model


def compile_pair(modules, data, *, policy=None, optimize=False, max_repairs=4):
    pipeline, copies = modules
    kernel, objects, banks, adapter, codec, limits, admission, model = data
    budgets = pipeline.ScheduleBudgets(
        eliminate_copies=False, max_fusions=0, optimize_storage=optimize, max_repairs=max_repairs
    )
    motion = copies.copy_pipeline_motion(
        admission,
        policy=policy or copies.CopyPipelinePolicy(distance=12, max_moves=32, max_pending_copies=3),
        budgets=budgets,
    )
    baseline = pipeline.compile_schedule(kernel, objects, banks, adapter, codec, limits, budgets=budgets, **model)
    candidate = pipeline.compile_schedule(
        kernel, objects, banks, adapter, codec, limits, budgets=budgets, copy_pipeline=motion, **model
    )
    return baseline, candidate


@pytest.mark.parametrize(
    "width,panels,slots,alias,fmt",
    [
        (1, 5, 2, True, "<BQQQQQ"),
        (7, 8, 2, False, ">BIIIII"),
        (13, 11, 3, True, "<BQQQQQ"),
    ],
)
def test_ordinary_pipeline_actual_words_complete_outputs_and_three_states(modules, width, panels, slots, alias, fmt):
    data = setup(modules, width=width, panels=panels, slots=slots, alias=alias, word_format=fmt)
    baseline, candidate = compile_pair(modules, data)
    kernel, _, banks, _, codec, _, _, _ = data
    carry = bytes((i * 31 + 253) % 256 for i in range(width * panels))
    for _ in range(3):
        results = []
        for compiled in (baseline, candidate):
            actual = replace(kernel, body=codec.decode_calls(b"".join(compiled.program.segments)))
            execution = leaf.run(actual, banks, carry)
            expected = bytes([197, 197]) + bytes((x * 3 + 7) % 256 for x in carry) + bytes([197, 197])
            assert execution[0] == expected
            writes = {bank.backing: set() for bank in banks}
            for call in actual.body:
                if call.instr in ("fetch", "affine"):
                    v = dict((k, evaluate(value, {})) for k, value in call.args)
                    backing = banks[v["view"]].backing
                    writes[backing].update(range(v["address"], v["address"] + v["width"]))
            for backing, memory in execution[1].items():
                assert all(value == 131 for i, value in enumerate(memory) if i not in writes[backing])
            results.append(execution)
        assert results[0][:3] == results[1][:3]
        assert results[0][3] == 1 and results[1][3] == slots
        carry = results[1][0][2:-2]
    proof = candidate.report["final_conformance"]["copy_pipeline"]
    assert proof["status"] == "applied" and proof["maximum_pending_copies"] == slots
    assert proof["removed_completion_instructions"] == 0
    assert proof["storage"]["lifetime_calls"] == candidate.report["storage"]["lifetime_calls"]
    assert len(candidate.kernel.body) == len(baseline.kernel.body)
    assert [r["name"] for r in candidate.report["passes"]] == [r["name"] for r in baseline.report["passes"]]
    assert candidate.report["context"]["library"] != baseline.report["context"]["library"]
    assert proof["timing"] == "UNKNOWN"


def test_one_bounded_original_slot_repair_only_selected_for_real_motion(modules):
    data = setup(modules, panels=2, slots=2)
    baseline, candidate = compile_pair(modules, data, optimize=True)
    assert "before_copy_slot_repair" in candidate.report["storage"]
    assert candidate.report["repair_rounds"] == 1
    assert candidate.report["final_conformance"]["copy_pipeline"]["status"] == "applied"
    assert baseline.report["storage"]["addresses"] != candidate.report["storage"]["addresses"]
    for result in (baseline, candidate):
        actual = replace(data[0], body=data[4].decode_calls(b"".join(result.program.segments)))
        assert leaf.run(actual, data[2], bytes(range(14)))[0][2:-2] == bytes((x * 3 + 7) % 256 for x in range(14))


@pytest.mark.parametrize("bound", [1, 2, 3])
def test_queue_capacity_is_bound_independently_of_motion_policy(modules, bound):
    data = list(setup(modules, slots=3))
    data[6] = replace(data[6], queue_capacity=bound)
    _, candidate = compile_pair(modules, data)
    proof = candidate.report["final_conformance"]["copy_pipeline"]
    actual = replace(data[0], body=data[4].decode_calls(b"".join(candidate.program.segments)))
    assert leaf.run(actual, data[2], bytes(range(49)), capacity=bound)[3] == bound
    assert proof["maximum_pending_copies"] == bound


@pytest.mark.parametrize("mutation", ["drop_wait", "wrong_address", "wrong_token", "false_partial_drain"])
def test_actual_encoded_mutations_cannot_inherit_checked_motion(modules, mutation):
    pipeline, copies = modules
    data = list(setup(modules))
    codec = data[4]
    if mutation == "false_partial_drain":
        data[4] = replace(
            codec,
            program_decoder=replace(
                codec.program_decoder,
                decode=lambda image: tuple(
                    replace(i, completion="drain") if call.instr == "await" else i
                    for i, call in zip(codec.program_decoder.decode(image), codec.decode_calls(image), strict=True)
                ),
            ),
        )
    else:

        def encode(kernel):
            calls = list(kernel.body)
            index = next(
                i for i, c in enumerate(calls) if c.instr == ("await" if mutation != "wrong_address" else "fetch")
            )
            if mutation == "drop_wait":
                calls.pop(index)
            else:
                key = "token" if mutation == "wrong_token" else "address"
                calls[index] = replace(
                    calls[index], args=tuple((k, Const(999) if k == key else v) for k, v in calls[index].args)
                )
            return codec.encode(replace(kernel, body=tuple(calls)))

        data[4] = replace(codec, encode=encode)
    with pytest.raises((ValueError, AssertionError), match="completion|token|effects|window|extent"):
        compile_pair(modules, data)


@pytest.mark.parametrize("field", ["admission", "policy"])
def test_original_owner_cannot_mutate_live_controls_in_encode(modules, field):
    pipeline, copies = modules
    data = setup(modules)
    policy = copies.CopyPipelinePolicy(distance=12, max_pending_copies=2)
    budgets = pipeline.ScheduleBudgets(eliminate_copies=False, optimize_storage=False)
    motion = copies.copy_pipeline_motion(data[6], policy=policy, budgets=budgets)
    codec = data[4]

    def encode(kernel):
        object.__setattr__(
            data[6] if field == "admission" else policy, "queue_capacity" if field == "admission" else "distance", 999
        )
        return codec.encode(kernel)

    with pytest.raises(ValueError, match="changed"):
        pipeline.compile_schedule(
            *data[:4], replace(codec, encode=encode), data[5], budgets=budgets, copy_pipeline=motion, **data[7]
        )


def test_missing_motion_preserves_legacy_partial_wait_refusal(modules):
    pipeline, copies = modules
    data = setup(modules)
    concurrent, _ = copies.pipeline_private_copies(
        data[0],
        data[1],
        data[2],
        data[3],
        data[6],
        policy=copies.CopyPipelinePolicy(distance=12, max_pending_copies=2),
        **data[7],
    )
    calls = []
    with pytest.raises(leaf.StorageOptimizationUnavailable, match="partial waits"):
        pipeline.compile_schedule(
            concurrent,
            *data[1:4],
            replace(data[4], encode=lambda k: calls.append(k)),
            data[5],
            budgets=pipeline.ScheduleBudgets(optimize_storage=False),
            **data[7],
        )
    assert calls == []


@pytest.mark.parametrize("disabled", ["missing", "no_pairs", "no_repair_budget"])
def test_unavailable_motion_preserves_legal_original_order(modules, disabled):
    pipeline, copies = modules
    data = list(setup(modules, panels=2, slots=2))
    budgets = pipeline.ScheduleBudgets(eliminate_copies=False, max_repairs=0 if disabled == "no_repair_budget" else 4)
    baseline = pipeline.compile_schedule(*data[:6], budgets=budgets, **data[7])
    if disabled == "missing":
        motion = copies.copy_pipeline_motion(None)
    else:
        admission = replace(data[6], crossing_pairs=()) if disabled == "no_pairs" else data[6]
        motion = copies.copy_pipeline_motion(
            admission, policy=copies.CopyPipelinePolicy(distance=12, max_pending_copies=2), budgets=budgets
        )
    actual = pipeline.compile_schedule(*data[:6], budgets=budgets, copy_pipeline=motion, **data[7])
    assert actual.kernel == baseline.kernel and actual.program.segments == baseline.program.segments
    assert actual.report["repair_rounds"] == 0


class NoDeletion:
    def propose(self, *args):
        return None

    def verify(self, *args):
        raise AssertionError("no deletion was proposed")

    def revalidate(self):
        pass

    def record(self):
        return {"scope": "original neutral no-op deletion test"}


def test_existing_deletion_slot_has_explicit_composition_fallback(modules):
    pipeline, copies = modules
    data = setup(modules)
    budgets = pipeline.ScheduleBudgets(eliminate_copies=False, optimize_storage=False)
    owner = NoDeletion()
    deletion = pipeline.PostStorageRewrite(
        "neutral_no_deletion", owner.propose, owner.verify, owner.revalidate, owner.record, (Path(__file__),)
    )
    motion = copies.copy_pipeline_motion(
        data[6], policy=copies.CopyPipelinePolicy(distance=12, max_pending_copies=2), budgets=budgets
    )
    actual = pipeline.compile_schedule(
        *data[:6], budgets=budgets, copy_pipeline=motion, post_storage=deletion, **data[7]
    )
    assert actual.kernel == data[0]
    assert actual.report["storage"]["copy_pipeline"]["reason"] == "unsupported_post_storage_composition"
    assert actual.report["storage"]["post_storage_rewrite"]["proof"]["status"] == "unchanged"


@pytest.mark.parametrize("mutation", ["membership", "cross_setup"])
def test_original_bad_proposer_needs_separate_checker_and_exact_positions(modules, mutation):
    pipeline, copies = modules
    data = setup(modules)
    budgets = pipeline.ScheduleBudgets(eliminate_copies=False, optimize_storage=False)
    motion = copies.copy_pipeline_motion(
        data[6], policy=copies.CopyPipelinePolicy(distance=12, max_pending_copies=2), budgets=budgets
    )
    propose = motion.propose

    def wrong(*args):
        actual, positions = propose(*args)
        if mutation == "membership":
            return actual, (positions[0],) * len(positions)
        order = list(positions)
        index = next(i for i, c in enumerate(actual.body) if c.instr == "fetch")
        order.insert(0, order.pop(index))
        return replace(actual, body=tuple(args[0].body[i] for i in order)), tuple(order)

    bad = replace(motion, propose=wrong, sources=(*motion.sources, Path(__file__)))
    with pytest.raises(ValueError, match="membership|unqualified inversion"):
        pipeline.compile_schedule(*data[:6], budgets=budgets, copy_pipeline=bad, **data[7])


@pytest.mark.parametrize("version", [6, 7, 8])
def test_exact_sdk_inventories_remain_import_closed_with_selected_companion_only_in_v8(tmp_path, modules, version):
    from merlin.common.paths import repo_root
    from merlin.compile.scheduling.export import namespace_replacements, public_modules
    from merlin.compile.scheduling.library_sources import library_modules
    from merlin.targetgen.compiler_library import freeze_compiler_library

    library = tmp_path / "library"
    overrides = {module.__name__: Path(module.__file__) for module in modules}
    namespaces = namespace_replacements(version)
    sources = []
    for name in library_modules(version):
        stem = repo_root() / "src" / Path(*name.split("."))
        source = stem.with_suffix(".py") if stem.with_suffix(".py").is_file() else stem / "__init__.py"
        relative = source.relative_to(repo_root() / "src")
        destination = library / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = overrides.get(name, source).read_bytes()
        if name in namespaces:
            payload = namespaces[name].encode()
        destination.write_bytes(payload)
        sources.append((relative.as_posix(), name))
    contract = freeze_compiler_library(
        library,
        review_id="neutral-copy-import-closure-only",
        public_modules=public_modules(version),
        sources=tuple(sources),
    )
    assert contract.permits("merlin.compile.scheduling.copy_pipeline") is (version == 8)


@pytest.mark.parametrize("binding", ["snapshot_reader", "checker", "owner_reader", "completion", "emitted_effects"])
def test_original_callback_cannot_replace_verification_code_after_motion(modules, monkeypatch, binding):
    pipeline, copies = modules
    data = setup(modules)
    budgets = pipeline.ScheduleBudgets(eliminate_copies=False, optimize_storage=False)
    motion = copies.copy_pipeline_motion(
        data[6], policy=copies.CopyPipelinePolicy(distance=12, max_pending_copies=2), budgets=budgets
    )
    codec = data[4]

    def encode(kernel):
        if binding == "snapshot_reader":
            monkeypatch.setattr(pipeline, "_rewrite_snapshot", lambda value: ())
        elif binding == "checker":
            monkeypatch.setattr(copies, "verify_copy_pipeline", lambda *args, **kwargs: {})
        elif binding in ("completion", "emitted_effects"):
            helper = "_program_completions" if binding == "completion" else "_emission_effects"
            original = getattr(pipeline, helper)
            monkeypatch.setattr(pipeline, helper, lambda *args, **kwargs: original(*args, **kwargs))
        else:
            monkeypatch.setattr(copies._CopyPipelineOwner, "_functions", staticmethod(lambda: ()))
        return codec.encode(kernel)

    with pytest.raises(ValueError, match="changed"):
        pipeline.compile_schedule(
            *data[:4], replace(codec, encode=encode), data[5], budgets=budgets, copy_pipeline=motion, **data[7]
        )


@pytest.mark.parametrize("returns_none", [False, True])
def test_proposer_cannot_retag_actual_allocation_before_source_is_retained(modules, returns_none):
    pipeline, copies = modules
    data = setup(modules)
    budgets = pipeline.ScheduleBudgets(eliminate_copies=False, optimize_storage=True)
    motion = copies.copy_pipeline_motion(
        data[6], policy=copies.CopyPipelinePolicy(distance=12, max_pending_copies=2), budgets=budgets
    )
    propose = motion.propose

    def changed(*args):
        assert args[0] is not data[0], "test requires actual relocated kernel"
        object.__setattr__(args[0], "name", "retagged-actual-allocation")
        return None if returns_none else propose(*args)

    bad = replace(motion, propose=changed, sources=(*motion.sources, Path(__file__)))
    with pytest.raises(ValueError, match="allocated source or slots changed"):
        pipeline.compile_schedule(*data[:6], budgets=budgets, copy_pipeline=bad, **data[7])


@pytest.mark.parametrize("field", ["fusion_emitters", "spill_emitters"])
def test_first_original_record_cannot_retarget_later_binding_controls(modules, field):
    pipeline, copies = modules
    data = setup(modules)
    controls = {}
    budgets = pipeline.ScheduleBudgets(eliminate_copies=False, optimize_storage=False)
    motion = copies.copy_pipeline_motion(data[6], budgets=budgets)
    record = motion.record

    def changed_record():
        controls["foreign"] = lambda *args: None
        return record()

    bad = replace(motion, record=changed_record, sources=(*motion.sources, Path(__file__)))
    with pytest.raises(ValueError, match="source, inputs or resources changed"):
        pipeline.compile_schedule(*data[:6], budgets=budgets, copy_pipeline=bad, **{field: controls}, **data[7])


def test_independent_verifier_cannot_mutate_realized_candidate(modules):
    pipeline, copies = modules
    data = setup(modules)
    budgets = pipeline.ScheduleBudgets(eliminate_copies=False, optimize_storage=False)
    motion = copies.copy_pipeline_motion(
        data[6], policy=copies.CopyPipelinePolicy(distance=12, max_pending_copies=2), budgets=budgets
    )
    verify = motion.verify

    def changed_verify(*args):
        result = verify(*args)
        object.__setattr__(args[1], "name", "retagged-after-independent-check")
        return result

    bad = replace(motion, verify=changed_verify, sources=(*motion.sources, Path(__file__)))
    with pytest.raises(ValueError, match="realized candidate changed"):
        pipeline.compile_schedule(*data[:6], budgets=budgets, copy_pipeline=bad, **data[7])


@pytest.mark.parametrize("helper", ["_project", "_overlap"])
def test_first_original_record_cannot_rebind_actual_storage_proof_helper(modules, monkeypatch, helper):
    from merlin.compile.scheduling import temporary_storage

    pipeline, copies = modules
    data = setup(modules)
    budgets = pipeline.ScheduleBudgets(eliminate_copies=False, optimize_storage=False)
    motion = copies.copy_pipeline_motion(data[6], budgets=budgets)
    record = motion.record

    def changed_record():
        monkeypatch.setattr(temporary_storage, helper, lambda *args, **kwargs: None)
        return record()

    bad = replace(motion, record=changed_record, sources=(*motion.sources, Path(__file__)))
    with pytest.raises(ValueError, match="implementation changed"):
        pipeline.compile_schedule(*data[:6], budgets=budgets, copy_pipeline=bad, **data[7])


def test_shared_helper_source_is_frozen_before_first_original_record(tmp_path, modules, monkeypatch):
    from merlin.compile.scheduling import temporary_storage

    pipeline, copies = modules
    observed_source = tmp_path / "original-storage-owner.py"
    observed_source.write_bytes(Path(temporary_storage.__file__).read_bytes())
    monkeypatch.setattr(temporary_storage, "__file__", str(observed_source))
    data = setup(modules)
    budgets = pipeline.ScheduleBudgets(eliminate_copies=False, optimize_storage=False)
    motion = copies.copy_pipeline_motion(data[6], budgets=budgets)
    record = motion.record

    def changed_record():
        observed_source.write_bytes(observed_source.read_bytes() + b"\n# changed original source\n")
        return record()

    bad = replace(motion, record=changed_record, sources=(*motion.sources, Path(__file__)))
    with pytest.raises(ValueError, match="implementation changed|resources changed"):
        pipeline.compile_schedule(*data[:6], budgets=budgets, copy_pipeline=bad, **data[7])


def test_declined_motion_source_remains_frozen_through_final_emission(modules):
    pipeline, copies = modules
    data = setup(modules)
    budgets = pipeline.ScheduleBudgets(eliminate_copies=False, optimize_storage=True)
    motion = copies.copy_pipeline_motion(
        data[6], policy=copies.CopyPipelinePolicy(distance=12, max_pending_copies=2), budgets=budgets
    )
    propose = motion.propose
    observed = []

    def decline(*args):
        observed.append(args[0])
        return None if len(observed) == 1 else propose(*args)

    codec = data[4]

    def encode(kernel):
        assert observed and observed[0] is not data[0]
        object.__setattr__(observed[0], "name", "late-modified-declined-allocation")
        return codec.encode(kernel)

    bad = replace(motion, propose=decline, sources=(*motion.sources, Path(__file__)))
    with pytest.raises(ValueError, match="allocated source or slots changed"):
        pipeline.compile_schedule(
            *data[:4], replace(codec, encode=encode), data[5], budgets=budgets, copy_pipeline=bad, **data[7]
        )
