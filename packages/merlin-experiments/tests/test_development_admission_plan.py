"""Planning must preserve original custody without issuing execution authority."""

import hashlib
import json
from pathlib import Path

import pytest
from merlin_experiments.phase0 import development_admission_plan as P

LIMITS = dict(cpu_seconds=90, wall_seconds=300, rss_bytes=1024**3, file_bytes=64 * 1024**2, output_bytes=128 * 1024**2)


def pin(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def fixture(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_text("public source byte fixture; no semantic or execution authority")
    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"independent reference byte fixture")
    ids = list(P.INITIAL) + [f"retained:{i}" for i in range(239)]
    numerical = dict(compare="exact_int", dtype="i32")
    members, history, ledger, inventory = [], [], [], []
    for i, identity in enumerate(ids):
        role = dict(role="objective" if i == 0 else "functional_guard", reason="original role")
        request = hashlib.sha256(identity.encode()).hexdigest()
        stages = dict(reference="retained", original_refusal="old budget denial retained")
        old_role = dict(counts_toward205=i == 0, guard=i != 0, value="original public intent")
        history.append(dict(identity=identity, prospective_setup_role=role, original_request_sha256=request))
        ledger.append(
            dict(
                identity=identity,
                numerical_policy=numerical,
                original_stages_preserved=stages,
                counts_toward_original205=i == 0,
                guard_not_performance_objective=i != 0,
                role=old_role["value"],
            )
        )
        members.append(
            dict(
                identity=identity,
                prospective_role=role,
                prospective_role_sha256_canonical_json=P._digest(role),
                numerical_policy=numerical,
                numerical_policy_sha256_canonical_json=P._digest(numerical),
                original_request_sha256=request,
                source_declaration_sha256="b" * 64,
                original_role=old_role,
                required_complete_accelerator_work=dict(products=1),
                required_coverage_classes=["raw_tail"] if i < 4 else [],
                historical_V23=dict(all_refusals_retained=["old denial"], original_stages=stages),
                current_V26=dict(status="NOT_RUN"),
                complete_independent_reference_stage="retained_complete_independent_reference",
                source_pins_recorded_not_live_verified=[pin(source)],
            )
        )
        if i < 4:
            inventory.append(
                dict(
                    ordinal=int(identity.rsplit(":", 1)[1]),
                    request_sha256=request,
                    reference=reference.name,
                    reference_identity={k: v for k, v in pin(reference).items() if k != "path"},
                )
            )

    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return pin(path)

    # engineering0 does not overlap the other selected original ordinals.
    inventory_pin = write("inventory.json", dict(members=inventory))
    for row in ledger:
        row["original_cohort_metadata"] = inventory_pin
    requirements = [
        dict(original_requirement=dict(requirement="raw_tail" if i == 0 else f"class_{i}", members=[]))
        for i in range(90)
    ]
    validation = dict(
        members_unmapped_to_retained90=[
            dict(identity=ids[4], original_source_request=dict(purposes=["raw_tail"])),
            dict(identity=ids[5], original_source_request=dict(purposes=["unreviewed_guess"])),
        ]
    )

    def bundle():
        inputs = dict(
            ledger=write("ledger.json", dict(members=ledger)),
            all243=write("original243.json", dict(members=history)),
            all90=write("original90.json", dict(requirements=requirements)),
            full16_handoff=write(
                "compiler-input.json", dict(compiler=dict(sha256="a" * 64, files={"compiler.py": pin(source)}))
            ),
        )
        pins = write(
            "data-pins.json",
            {
                "all243-members.json": write("members.json", dict(members=members)),
                "all90-requirements.json": write("requirements.json", dict(requirements=requirements)),
                "summary.json": write("summary.json", dict(input_pins=inputs)),
            },
        )
        handoff = write("handoff.json", {"pins.json": pins, "validation.json": write("validation.json", validation)})
        return write("final.json", dict(existing_handoff=handoff))

    return dict(bundle=bundle, members=members, ledger=ledger, source=source, reference=reference)


def run(f):
    p = f["bundle"]()
    return P.plan(Path(p["path"]), sha256=p["sha256"], limits=LIMITS)


def test_complete_deterministic_plan_is_not_admission(tmp_path):
    f = fixture(tmp_path)
    first, second = run(f), run(f)
    assert first == second
    assert len(first["members"]) == 243
    assert first["initial_order"] == list(P.INITIAL)
    assert [job["identity"] for job in first["jobs"]] == list(P.INITIAL)
    assert not any(
        first[k]
        for k in (
            "source_sealed",
            "writer_admitted",
            "jobs_executed",
            "jobs_authorized",
            "performance_objective_changed",
        )
    )
    for row, old in zip(first["members"], f["members"], strict=True):
        assert row["original_role"] == old["original_role"]
        assert row["historical_stages"]["original_refusal"] == "old budget denial retained"
    assert first["members"][4]["coverage_status"] == "exact_original_declared_purpose_binding"
    assert first["members"][5]["coverage_status"] == "REFUSED_UNMAPPED_REQUIREMENT"


@pytest.mark.parametrize("attack", ["drop", "duplicate", "role", "numeric", "refusal"])
def test_coherently_repinned_overlay_cannot_change_originals(tmp_path, attack):
    f = fixture(tmp_path)
    if attack == "drop":
        f["members"].pop()
    elif attack == "duplicate":
        f["members"][-1] = f["members"][0]
    elif attack == "role":
        f["members"][1]["prospective_role"] = dict(role="objective", reason="speed")
        f["members"][1]["prospective_role_sha256_canonical_json"] = P._digest(f["members"][1]["prospective_role"])
    elif attack == "numeric":
        f["members"][0]["numerical_policy"] = dict(compare="approximate")
        f["members"][0]["numerical_policy_sha256_canonical_json"] = P._digest(f["members"][0]["numerical_policy"])
    else:
        f["members"][0]["historical_V23"]["original_stages"] = dict(reference="retained")
    with pytest.raises(ValueError):
        run(f)


@pytest.mark.parametrize("leaf", ["source", "reference"])
def test_stale_original_source_or_reference_refuses(tmp_path, leaf):
    f = fixture(tmp_path)
    selected = f["bundle"]()
    f[leaf].write_bytes(b"changed")
    with pytest.raises(ValueError, match="input bytes changed"):
        P.plan(Path(selected["path"]), sha256=selected["sha256"], limits=LIMITS)


def test_mutation_during_planning_is_rechecked(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    old = P._assemble

    def change(*args):
        rows = old(*args)
        f["reference"].write_bytes(b"changed after initial read")
        return rows

    monkeypatch.setattr(P, "_assemble", change)
    with pytest.raises(ValueError, match="input bytes changed"):
        run(f)


def test_trace_cap_and_bool_budget_are_not_accepted(tmp_path):
    f = fixture(tmp_path)
    selected = f["bundle"]()
    for limits in (dict(LIMITS, cpu_seconds=True), dict(LIMITS, file_bytes=64 * 1024**2 + 1)):
        with pytest.raises(ValueError, match="finite original limits"):
            P.plan(Path(selected["path"]), sha256=selected["sha256"], limits=limits)


def test_supplement_must_rejoin_original_index_not_overlay_claim(tmp_path):
    f = fixture(tmp_path)
    source = f["members"][0]
    output = dict(path=f["reference"].name, **{k: v for k, v in pin(f["reference"]).items() if k != "path"})
    index = tmp_path / "supplement.json"
    index.write_text(
        json.dumps(
            dict(
                source_manifest_sha256=f["ledger"][0]["original_cohort_metadata"]["sha256"],
                members=[
                    dict(
                        ordinal=166,
                        request_sha256=source["original_request_sha256"],
                        status="complete_exact_reference_materialized",
                        numeric_policy=source["numerical_policy"],
                        outputs=[output],
                    )
                ],
            )
        )
    )
    source["complete_independent_reference_stage"] = dict(kind="supplement", index=pin(index), outputs=[output])
    assert run(f)["jobs"][0]["identity"] == "original190:166"
    # Coherently re-signing the overlay does not change original index ownership.
    source["complete_independent_reference_stage"]["outputs"] = [dict(output, sha256="0" * 64)]
    with pytest.raises(ValueError, match="supplemental reference correspondence"):
        run(f)


def test_source_leaf_symlink_is_not_a_pin(tmp_path):
    f = fixture(tmp_path)
    selected = f["bundle"]()
    target = tmp_path / "actual.mlir"
    f["source"].rename(target)
    f["source"].symlink_to(target)
    with pytest.raises(ValueError, match="canonical direct"):
        P.plan(Path(selected["path"]), sha256=selected["sha256"], limits=LIMITS)
