"""Ordinary allocation→dead payload→actual byte decoding and native execution."""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from test_dead_payload_writes import changed, fixture, project
from test_private_copy_elimination import command, rewrite
from test_schedule_pipeline import encoding, limits_for
from test_selected_storage_contracts import inputs, native  # noqa: F401

from merlin.compile.scheduling import dead_writes as D
from merlin.compile.scheduling.dead_write_pipeline import compile_schedule_with_dead_writes as compile_schedule
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets


def ordinary(width=7, *, separate=False, word_format="<BBBQQQ", **options):
    source, objects, banks, adapter, model = fixture(width, separate=separate)
    adapter = replace(adapter, rewrite=rewrite)
    codec, word = encoding(word_format)
    codec = replace(
        codec,
        numerical_policy_identity=adapter.numerical_policy_identity,
        program_decoder=replace(codec.program_decoder, configuration_identity=adapter.configuration_identity),
    )
    limits = limits_for(banks, model, word)
    return (source, objects, banks, adapter, codec, limits), word, model | options


def build(args, model, *, law=True, **kwargs):
    values = dict(model)
    deletion = values.pop("deletion_law")
    return compile_schedule(
        *args,
        **values,
        **({"deletion_law": deletion} if law else {}),
        budgets=kwargs.pop("budgets", ScheduleBudgets(eliminate_copies=False)),
        **kwargs,
    )


@pytest.mark.parametrize("optimize", [False, True])
def test_realization_runs_after_actual_placement_rebuilds_lifetime_and_invalidates_old_analyses(optimize):
    args, word, model = ordinary()
    budgets = ScheduleBudgets(eliminate_copies=False, optimize_storage=optimize)
    baseline, result = build(args, model, law=False, budgets=budgets), build(args, model, budgets=budgets)
    assert [row["name"] for row in baseline.report["passes"]] == [
        "semantic_admission",
        "storage",
        "completion",
        "emission",
        "final_program",
    ]
    assert [row["name"] for row in result.report["passes"]] == [
        "semantic_admission",
        "storage",
        "dead_writes",
        "completion",
        "emission",
        "final_program",
    ]
    stage = result.report["passes"][2]
    assert stage["invalidated"] == ("realization", "storage")
    assert stage["before"]["source_sha256"] != stage["after"]["source_sha256"]
    assert stage["produced"] == ("storage", "realization", "dead_writes")
    storage = result.report["storage"]
    assert storage["post_storage_rewrite"]["proof"]["deleted_calls"] == 1
    assert storage["lifetime_calls"] == {"a": [0, 3], "b": [3, 4]}
    assert storage["before_post_storage"]["lifetime_calls"] == {"a": [0, 4], "b": [4, 5]}
    assert storage["addresses"] == baseline.report["storage"]["addresses"]
    assert storage["static_commands"] == result.report["final_conformance"]["actual_commands"] == 6
    assert storage["requested_write_bytes"] == result.report["final_conformance"]["requested_write_bytes"] == 70
    assert result.report["final_conformance"]["post_storage_rewrite"]["proof"]["omitted_source_indices"] == [0]
    assert result.report["conditional_sources_sha256"] and "post_storage_rewrite" in result.report["resources"]
    assert (
        "conditional_sources_sha256" not in baseline.report
        and "post_storage_rewrite" not in baseline.report["resources"]
    )
    assert result.report["library_sources_sha256"] == baseline.report["library_sources_sha256"]
    assert result.program.report["encoded_bytes"] == 6 * word.size


@pytest.mark.parametrize("width,separate,word_format", [(3, False, ">BBBIII"), (13, True, "<BBBQQQ")])
def test_ordinary_selected_facts_path_executes_complete_native_state_from_actual_final_bytes(
    tmp_path,
    native,  # noqa: F811
    width,
    separate,
    word_format,  # noqa: F811
):
    selected_model = inputs(tmp_path, width=width, separate=separate, word_format=word_format, reserve=False)
    selected = selected_model[0]
    args, word, model = ordinary(width, separate=separate, word_format=word_format)
    source, objects, _, adapter, codec, limits = args
    adapter = replace(
        adapter, configuration_identity=selected.hardware_identity, numerical_policy_identity=selected.numerical_policy
    )
    _, _, _, checked_adapter = selected.storage.bind(
        selected.isa, adapter, banks=None, reservations=(), external_windows=None, private_ownership=True
    )
    # The original neutral C qualifier covers every legal copy in this selected
    # domain. Restrict that law to exact effective semantic forms; projecting
    # a form does not independently establish the deletion premises.
    law = replace(
        model["deletion_law"],
        configuration_identity=adapter.configuration_identity,
        numerical_policy_identity=adapter.numerical_policy_identity,
        removable_forms=tuple(
            dict.fromkeys(
                (call.instr, checked_adapter.project(call).semantic_key) for call in source.body if call.instr == "copy"
            )
        ),
    )
    model = dict(private_ownership=True, deletion_law=law, selected_contracts=selected)
    codec, limits = selected_model[4:6]
    args = source, objects, None, adapter, codec, limits
    arms = []
    for enabled in (False, True):
        result = build(args, model, max_write_deletions=128 if enabled else 0)
        image = b"".join(result.program.segments)
        (tmp_path / ("selected" if enabled else "original")).mkdir()
        directory = tmp_path / ("selected" if enabled else "original")
        (directory / "program.bin").write_bytes(image)
        state, calls = bytes((i * 19 + 251) % 256 for i in range(2 * width)), []
        for epoch in range(3):
            raw = bytes((i * 31 + 253 + epoch) % 256 for i in range(2 * width))
            packed = raw + state + bytes([73] * 8)
            # Selected external window can be larger than the actual live values.
            capacity = selected.storage.external[0].size
            data = bytearray(packed + bytes([73] * (capacity - len(packed))))
            output = bytearray([91] * capacity)
            first, second = bytearray([173] * 192), bytearray([149] * 192)
            before = bytes(data)
            counters = native(
                image,
                first,
                second,
                data,
                output,
                width=4 if word_format.endswith("I") else 8,
                big=word_format[0] == ">",
                separate=separate,
            )
            expected = bytes((x + 1) % 256 for x in state) + raw + bytes([91] * (capacity - 4 * width))
            assert bytes(output) == expected and bytes(data) == before
            occupied = {}
            for obj in result.report["storage"]["objects"]:
                backing = dict((b["name"], b["backing"]) for b in result.report["storage"]["banks"])[obj["bank"]]
                occupied.setdefault(backing, []).append(
                    (result.report["storage"]["addresses"][obj["name"]], obj["size"])
                )
            for memory, backing, guard in ((first, "first", 173), (second, "second", 149)):
                assert all(
                    value == guard
                    for i, value in enumerate(memory)
                    if not any(start <= i < start + size for start, size in occupied.get(backing, []))
                )
            assert counters[:3] == (
                6 if enabled else 7,
                (10 if enabled else 12) * width,
                (10 if enabled else 12) * width,
            )
            calls.append(
                dict(
                    epoch=epoch,
                    input_before=before.hex(),
                    input_after=bytes(data).hex(),
                    state=state.hex(),
                    output=bytes(output).hex(),
                    first=bytes(first).hex(),
                    second=bytes(second).hex(),
                    counters=counters,
                )
            )
            state = bytes(output[: 2 * width])
        arms.append(dict(enabled=enabled, report=result.report, calls=calls))
    assert [c["output"] for c in arms[0]["calls"]] == [c["output"] for c in arms[1]["calls"]]
    assert arms[0]["report"]["context"]["library"] == arms[1]["report"]["context"]["library"]
    (tmp_path / "pair.json").write_text(
        json.dumps(
            dict(
                scope="neutral ordinary schedule execution; no target/timing qualification",
                selected_contracts=selected.record(),
                native_implementation={
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (tmp_path / "machine.c", tmp_path / "machine.so")
                },
                compiler={
                    "path": shutil.which("cc"),
                    "sha256": hashlib.sha256(Path(shutil.which("cc")).read_bytes()).hexdigest(),
                },
                word_format=word_format,
                arms=arms,
            ),
            indent=2,
        )
    )


def test_original_law_missing_or_disabled_does_not_remove_any_program_work():
    args, word, model = ordinary()
    default = build(args, model, law=False)
    disabled = build(args, model, max_write_deletions=0)
    unavailable = build(args, model | {"deletion_law": replace(model["deletion_law"], no_faults=False)})
    assert default.program.segments == disabled.program.segments == unavailable.program.segments
    assert disabled.report["storage"]["post_storage_rewrite"]["proof"]["deleted_calls"] == 0
    assert unavailable.report["final_conformance"]["post_storage_rewrite"]["proof"] is None


@pytest.mark.parametrize("mutation", ["ignore", "address", "read", "omit"])
def test_actual_encoder_cannot_ignore_or_corrupt_selected_candidate(mutation):
    args, word, model = ordinary()
    source, *_, codec, limits = args

    def broken(kernel):
        calls = list(kernel.body)
        if mutation == "ignore":
            return codec.encode(source)
        if mutation == "address":
            calls[0] = changed(calls[0], destination=1)
        elif mutation == "read":
            calls[2] = replace(calls[2], instr="copy")
        else:
            calls.pop()
        return codec.encode(replace(kernel, body=tuple(calls)))

    with pytest.raises(ValueError, match="effects|window|coverage|original|instruction"):
        build((*args[:4], replace(codec, encode=broken), limits), model)


def test_lying_request_decoder_cannot_substitute_for_independent_actual_access_decoder():
    args, word, model = ordinary()
    *_, codec, limits = args
    captured = []

    def broken(kernel):
        captured[:] = kernel.body
        return codec.encode(replace(kernel, body=(changed(kernel.body[0], destination=1), *kernel.body[1:])))

    lying = replace(codec, encode=broken, decode_calls=lambda image: tuple(captured))
    with pytest.raises(ValueError, match="matching straight-line access"):
        build((*args[:4], lying, limits), model)


@pytest.mark.parametrize("role", ["law", "source", "helper", "adapter"])
def test_late_encoder_mutations_refuse_original_law_source_code_and_adapter(role, monkeypatch):
    args, word, model = ordinary()
    *_, codec, limits = args
    original = D.eliminate_dead_writes
    captured = []

    def observe(source, *args, **kwargs):
        captured.append(source)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(D, "eliminate_dead_writes", observe)

    def mutate(kernel):
        image = codec.encode(kernel)
        if role == "law":
            object.__setattr__(model["deletion_law"], "qualification_identity", "changed-original")
        elif role == "source":
            object.__setattr__(captured[0], "name", "changed-before-deletion-source")
        elif role == "helper":
            monkeypatch.setattr(D, "eliminate_dead_writes", lambda *a, **k: None)
        else:
            object.__setattr__(args[3], "project", lambda call: project(call))
        return image

    with pytest.raises(ValueError, match="original|resources"):
        build((*args[:4], replace(codec, encode=mutate), limits), model)


def test_unknown_source_effects_remain_an_admission_refusal_before_encoding():
    args, word, model = ordinary()
    source, objects, banks, adapter, codec, limits = args
    source = replace(source, body=(replace(source.body[0], instr="opaque"), *source.body[1:]))
    codec = replace(codec, encode=lambda kernel: pytest.fail("unknown effects reached encoding"))
    with pytest.raises(ValueError, match="unknown instruction"):
        build((source, objects, banks, adapter, codec, limits), model)


def test_explicit_order_cannot_consume_old_storage_after_selected_deletion():
    args, word, model = ordinary()
    with pytest.raises(ValueError, match="ordering|prerequisites"):
        build(
            args,
            model,
            pass_order=("semantic_admission", "storage", "completion", "dead_writes", "emission", "final_program"),
        )


@pytest.mark.parametrize("identity", ["configuration_identity", "numerical_policy_identity", "qualification_identity"])
def test_selected_binding_still_freezes_original_raw_adapter_scalar_ids(tmp_path, identity):
    selected_model = inputs(tmp_path, reserve=False)
    selected, _, _, original_adapter, codec, limits, *_ = selected_model
    args, word, model = ordinary()
    source, objects = args[:2]
    adapter = replace(
        args[3], configuration_identity=selected.hardware_identity, numerical_policy_identity=selected.numerical_policy
    )
    law = replace(
        model["deletion_law"],
        configuration_identity=adapter.configuration_identity,
        numerical_policy_identity=adapter.numerical_policy_identity,
    )
    original = codec.encode

    def mutate(kernel):
        image = original(kernel)
        object.__setattr__(adapter, identity, "changed-original-raw-adapter")
        return image

    with pytest.raises(ValueError, match="original post-storage adapter identities"):
        compile_schedule(
            source,
            objects,
            None,
            adapter,
            replace(codec, encode=mutate),
            limits,
            selected_contracts=selected,
            private_ownership=True,
            deletion_law=law,
            budgets=ScheduleBudgets(eliminate_copies=False),
        )


def test_async_copy_and_wait_stay_on_the_original_source_route(tmp_path):
    from test_selected_storage_contracts import provider_project

    selected_model = inputs(tmp_path, reserve=False)
    selected, _, _, original_adapter, codec, limits, *_ = selected_model
    args, word, model = ordinary()
    source, objects = args[:2]
    first = source.body[0]
    asynchronous = replace(first, instr="async_copy")
    wait = command("wait", 0, 0, 14)
    source = replace(source, body=(first, asynchronous, wait, *source.body[1:]))
    adapter = replace(original_adapter, project=provider_project)
    _, _, _, checked_adapter = selected.storage.bind(
        selected.isa, adapter, banks=None, reservations=(), external_windows=None, private_ownership=True
    )
    law = replace(
        model["deletion_law"],
        configuration_identity=adapter.configuration_identity,
        numerical_policy_identity=adapter.numerical_policy_identity,
        adapter_qualification_identity=adapter.qualification_identity,
        removable_forms=((first.instr, checked_adapter.project(first).semantic_key),),
    )
    result = compile_schedule(
        source,
        objects,
        None,
        adapter,
        codec,
        limits,
        selected_contracts=selected,
        private_ownership=True,
        deletion_law=law,
        budgets=ScheduleBudgets(eliminate_copies=False, optimize_storage=False),
    )
    assert result.kernel == source
    assert result.report["storage"]["post_storage_rewrite"]["proof"]["deleted_calls"] == 0
    assert result.report["final_conformance"]["post_storage_rewrite"]["proof"] is None


class OriginalRewrite:
    """Neutral original owner used to exercise the base callback boundaries."""

    def __init__(self, action=lambda boundary, source, objects: None):
        self.action = action
        self.events = []

    def visit(self, boundary, source=None, objects=None):
        first = boundary not in self.events
        self.events.append(boundary)
        if first:
            return self.action(boundary, source, objects)

    def record(self):
        self.visit("record")
        return {"original_law": "neutral-test-only"}

    def revalidate(self):
        self.visit("revalidate")

    def propose(self, source, objects, *args):
        return self.visit("propose", source, objects)

    def verify(self, source, actual, objects, *args):
        self.visit("verify", actual, objects)
        return {"deleted_calls": len(source.body) - len(actual.body)}


def slot(owner, *, sources=None):
    from merlin.common.paths import merlin_dir
    from merlin.compile.scheduling.schedule_pipeline import PostStorageRewrite

    return PostStorageRewrite(
        "original_deletion",
        owner.propose,
        owner.verify,
        owner.revalidate,
        owner.record,
        sources if sources is not None else (merlin_dir() / "tests/infra/test_schedule_dead_writes.py",),
    )


def slot_build(args, model, owner, *, budgets=None):
    from merlin.compile.scheduling.schedule_pipeline import compile_schedule as ordinary_compile

    return ordinary_compile(
        *args,
        **{key: value for key, value in model.items() if key != "deletion_law"},
        budgets=budgets or ScheduleBudgets(eliminate_copies=False),
        post_storage=slot(owner),
    )


@pytest.mark.parametrize("boundary", ["record", "revalidate"])
@pytest.mark.parametrize("field", ["source", "body", "object", "bank", "budget"])
def test_first_owner_callback_cannot_rebase_original_source_or_resources(boundary, field):
    args, _, model = ordinary()
    budgets = ScheduleBudgets(eliminate_copies=False)

    def mutate(event, source, objects):
        if event != boundary:
            return
        if field == "source":
            object.__setattr__(args[0], "name", "replaced-original-source")
        elif field == "body":
            object.__setattr__(args[0], "body", args[0].body[1:])
        elif field == "object":
            object.__setattr__(args[1][0], "size", args[1][0].size + 1)
        elif field == "bank":
            object.__setattr__(args[2][0], "capacity", args[2][0].capacity + 1)
        else:
            object.__setattr__(budgets, "max_calls", budgets.max_calls + 1)

    with pytest.raises(ValueError, match="original post-storage source, resources or controls"):
        slot_build(args, model, OriginalRewrite(mutate), budgets=budgets)


@pytest.mark.parametrize("field", ["name", "body", "object"])
def test_proposer_cannot_mutate_the_original_checked_placed_source(field):
    args, _, model = ordinary()

    def mutate(event, source, objects):
        if event == "propose":
            assert source is not args[0]  # Actual physical placement produced this kernel.
            if field == "object":
                object.__setattr__(objects[0], "size", objects[0].size + 1)
            elif field == "name":
                object.__setattr__(source, "name", "changed-placed-source")
            else:
                object.__setattr__(source, "body", source.body[1:])

    with pytest.raises(ValueError, match="original post-storage source or lifetimes"):
        slot_build(args, model, OriginalRewrite(mutate))


@pytest.mark.parametrize("mutation", ["address", "arithmetic", "order"])
def test_base_slot_refuses_rewriting_surviving_instructions_even_with_approving_verifier(mutation):
    args, _, model = ordinary()

    def propose(event, source, objects):
        if event == "propose":
            calls = list(source.body[1:])
            if mutation == "address":
                calls[0] = changed(calls[0], destination=123)
            elif mutation == "arithmetic":
                calls[0] = replace(calls[0], instr="increment")
            else:
                calls[0], calls[1] = calls[1], calls[0]
            return replace(source, body=tuple(calls))

    owner = OriginalRewrite(propose)
    with pytest.raises(ValueError, match="unchanged original instructions in order"):
        slot_build(args, model, owner)
    assert "verify" not in owner.events


def test_base_slot_checks_candidate_again_after_original_verifier():
    args, _, model = ordinary()

    def mutate(event, source, objects):
        if event == "propose":
            return replace(source, body=source.body[1:])
        if event == "verify":
            object.__setattr__(source, "name", "changed-after-proof")

    with pytest.raises(ValueError, match="verified post-storage candidate changed"):
        slot_build(args, model, OriginalRewrite(mutate))


def test_slot_requires_pinnable_python_callbacks_and_their_actual_source_owners():
    from merlin.common.paths import merlin_dir

    class CallableVerifier:
        def __call__(self, *args):
            pytest.fail("unsupported callable instance executed")

    owner = OriginalRewrite()
    with pytest.raises(ValueError, match="separate verifier"):
        replace(slot(owner), verify=CallableVerifier())
    args, _, model = ordinary()
    from merlin.compile.scheduling.schedule_pipeline import compile_schedule as ordinary_compile

    with pytest.raises(ValueError, match="actual Python source owners"):
        ordinary_compile(
            *args,
            **{key: value for key, value in model.items() if key != "deletion_law"},
            post_storage=slot(owner, sources=(merlin_dir() / "tests/infra/test_dead_payload_writes.py",)),
        )
    assert owner.events == []


def test_over_budget_source_refuses_before_any_original_owner_callback():
    args, _, model = ordinary()
    owner = OriginalRewrite()
    with pytest.raises(ValueError, match="original source or declarations exceed typed work budgets"):
        slot_build(args, model, owner, budgets=ScheduleBudgets(max_calls=1))
    assert owner.events == []


@pytest.mark.parametrize("replacement", ["function", "code"])
def test_first_record_cannot_replace_the_original_input_snapshot_reader(monkeypatch, replacement):
    from merlin.compile.scheduling import schedule_pipeline as P

    args, _, model = ordinary()

    def mutate(event, source, objects):
        if event == "record":
            if replacement == "function":
                monkeypatch.setattr(P, "_rewrite_snapshot", lambda value: ())
            else:
                monkeypatch.setattr(P._rewrite_snapshot, "__code__", (lambda value: ()).__code__)

    with pytest.raises(ValueError, match="original post-storage snapshot reader changed"):
        slot_build(args, model, OriginalRewrite(mutate))


@pytest.mark.parametrize(
    "name", ["encoded", "final_conformance", "configuration_candidate", "fusion_candidate", "prefetch_candidate"]
)
def test_slot_name_cannot_shadow_existing_pipeline_analysis(name):
    owner = OriginalRewrite()
    with pytest.raises(ValueError, match="bounded original name"):
        replace(slot(owner), name=name)
    assert owner.events == []


@pytest.mark.parametrize("name", ["objects", "addresses", "lifetime_calls", "actual_commands", "image_sha256"])
def test_original_slot_name_cannot_overwrite_allocation_or_actual_emission_report_fields(name):
    from merlin.compile.scheduling.schedule_pipeline import compile_schedule as ordinary_compile

    args, _, model = ordinary()
    owner = OriginalRewrite(
        lambda event, source, objects: replace(source, body=source.body[1:]) if event == "propose" else None
    )
    result = ordinary_compile(
        *args,
        **{key: value for key, value in model.items() if key != "deletion_law"},
        budgets=ScheduleBudgets(eliminate_copies=False),
        post_storage=replace(slot(owner), name=name),
    )
    storage, actual = result.report["storage"], result.report["final_conformance"]
    assert storage["addresses"].keys() == {"a", "b"} and len(storage["objects"]) == 2
    assert storage["lifetime_calls"] == {"a": [0, 3], "b": [3, 4]}
    assert actual["actual_commands"] == 6
    assert actual["image_sha256"] == hashlib.sha256(b"".join(result.program.segments)).hexdigest()
    assert storage["post_storage_rewrite"] == {"name": name, "proof": {"status": "applied", "deleted_calls": 1}}
    assert actual["post_storage_rewrite"] == {"name": name, "proof": {"deleted_calls": 1}}
