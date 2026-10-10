"""Form-scale holdout: private members, a counts-only commitment and a standard v2 reveal."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import checkpoint_admission as AD
from merlin_experiments.phase2 import checkpoint_cli as CLI
from merlin_experiments.phase2 import checkpoint_controller as CTRL
from merlin_experiments.phase2 import form_holdout as FH
from merlin_experiments.phase2 import holdout_corpus as HC
from merlin_experiments.phase2 import revealed_corpus as RC


def _capsule(root, name, m, k, n, *, application, family="PW"):
    directory = root / "_perf" / name
    directory.mkdir(parents=True)
    document = {
        "name": name,
        "inputs": [
            {"name": "A0", "role": "lhs", "shape": [m, k], "dtype": "i8"},
            {"name": "W", "role": "weight", "shape": [k, n], "dtype": "i8"},
        ],
        "operation": {"op": "matmul", "attributes": {"lhs": "A0", "weight": "W", "out": "Y0", "epilogue": []}},
        "numeric_policy": {"compare": "exact_int"},
        "performance": {"family": family, "form": {"representative": {"application": application}}},
    }
    (directory / "capsule.yaml").write_text(yaml.safe_dump(document))
    (directory / "capsule.interface.mlir").write_text(f"// {name}\n")
    return directory


def _generated(tmp_path, shapes, *, application="cnn_h"):
    root = tmp_path / "private-run"
    for index, (m, k, n) in enumerate(shapes):
        _capsule(root, f"PW{index:02d}_form", m, k, n, application=application)
    _capsule(root, "PW90_iteration", 8, 16, 16, application="iteration_app")
    _capsule(root, "PK00_law", 16, 16, 16, application="cnn_h", family="PK")
    return root


def _tuning(tmp_path, shapes):
    root = tmp_path / "tuning"
    for index, (m, k, n) in enumerate(shapes):
        _capsule(root, f"PW{index:02d}_tune", m, k, n, application="cnn")
    return root


def _seal(tmp_path, name):
    from merlin.benchharness import hash_tree

    tree = tmp_path / f"{name}-tree"
    tree.mkdir()
    (tree / "compiler.py").write_text("pass\n")
    (tree / "compiler.py").chmod(0o444)
    tree.chmod(0o555)
    record = tmp_path / f"{name}.json"
    record.write_text(
        json.dumps(
            {
                "state": "sealed",
                "candidate": {"read_only": True, "sha256": hash_tree(tree)["sha256"], "path": str(tree)},
                "admission": {"consumable": True},
            }
        )
    )
    record.chmod(0o444)
    return record


_HELD = [(40, 72, 24), (24, 48, 96), (1, 56, 112), (36, 40, 80)]


def test_only_private_roster_form_members_are_selected(tmp_path):
    root = _generated(tmp_path, _HELD)
    names = [path.name for path in FH.select_members(root, applications=["cnn_h"])]
    assert names == ["PW00_form", "PW01_form", "PW02_form", "PW03_form"]


def test_commit_refuses_a_member_the_tuning_set_already_measures(tmp_path):
    root = _generated(tmp_path, _HELD)
    tuning = _tuning(tmp_path, [(2304, 72, 24), _HELD[2]])
    with pytest.raises(HC.HoldoutError, match="1 holdout member"):
        FH.commit_form_holdout(
            root,
            tmp_path / "commitment.json",
            tmp_path / "private",
            target="synthetic",
            applications=["cnn_h"],
            tuning_root=tuning,
            candidate_ids=["c0"],
        )


def test_commit_publishes_counts_only_and_reveal_is_a_standard_v2_corpus(tmp_path):
    root = _generated(tmp_path, _HELD)
    tuning = _tuning(tmp_path, [(2304, 72, 24), (96, 768, 4096)])
    paths = FH.commit_form_holdout(
        root,
        tmp_path / "commitment.json",
        tmp_path / "private",
        target="synthetic",
        applications=["cnn_h"],
        tuning_root=tuning,
        candidate_ids=["c0"],
    )
    public_text = paths["public_commitment"].read_text()
    assert "784" not in public_text and "cnn_h" not in public_text and "PW00" not in public_text
    public = json.loads(public_text)
    assert public["member_count"] == 4 and public["disjoint_from_tuning"] is True
    assert stat.S_IMODE(paths["state"].stat().st_mode) == 0o600
    manifest = FH.reveal_form_holdout(
        paths["public_commitment"],
        paths["host_private_dir"],
        tmp_path / "reveal",
        candidate_seals={"c0": _seal(tmp_path, "c0")},
    )
    members = RC.load_revealed_members(manifest, expected_target="synthetic")
    assert sorted((m.name, m.cohort, m.family) for m in members) == [
        (f"PW{i:02d}_form", FH.COHORT, "PW") for i in range(4)
    ]


def test_reveal_refuses_changed_members_or_missing_seals(tmp_path):
    root = _generated(tmp_path, _HELD)
    tuning = _tuning(tmp_path, [(2304, 72, 24)])
    paths = FH.commit_form_holdout(
        root,
        tmp_path / "commitment.json",
        tmp_path / "private",
        target="synthetic",
        applications=["cnn_h"],
        tuning_root=tuning,
        candidate_ids=["c0"],
    )
    with pytest.raises(HC.HoldoutError, match="incomplete or foreign"):
        FH.reveal_form_holdout(
            paths["public_commitment"], paths["host_private_dir"], tmp_path / "r1", candidate_seals={}
        )
    (root / "_perf" / "PW01_form" / "capsule.interface.mlir").write_text("// changed\n")
    with pytest.raises(HC.HoldoutError, match="changed before the reveal"):
        FH.reveal_form_holdout(
            paths["public_commitment"],
            paths["host_private_dir"],
            tmp_path / "r2",
            candidate_seals={"c0": _seal(tmp_path, "c0")},
        )


def test_too_small_a_private_cohort_is_refused(tmp_path):
    root = _generated(tmp_path, _HELD[:2])
    with pytest.raises(HC.HoldoutError, match="need"):
        FH.commit_form_holdout(
            root,
            tmp_path / "commitment.json",
            tmp_path / "private",
            target="synthetic",
            applications=["cnn_h"],
            tuning_root=_tuning(tmp_path, [(1, 2, 3)]),
            candidate_ids=["c0"],
        )


# --- Controller inputs: preflight blockers, the private spec and the labelled measurement matrix ----


def _preflight_config(tmp_path, generated, applications=("cnn_h",)):
    return SimpleNamespace(
        root=tmp_path / "experiment",
        context=SimpleNamespace(stage_root=tmp_path / "stages"),
        form_holdout_generated_root=generated,
        form_holdout_applications=tuple(applications),
        form_holdout_family="PW",
    )


def test_preflight_admits_a_private_cohort_outside_every_agent_reachable_root(tmp_path):
    tuning = _tuning(tmp_path, [(2304, 72, 24)])
    target = SimpleNamespace(capsule_corpus=tuning / "public")
    config = _preflight_config(tmp_path, _generated(tmp_path, _HELD))
    assert AD.form_holdout_blockers(config, target) == []


def test_preflight_blocks_a_private_cohort_an_agent_could_read_or_too_small_a_roster(tmp_path):
    tuning = _tuning(tmp_path, [(2304, 72, 24)])
    target = SimpleNamespace(capsule_corpus=tuning / "public")
    inside = _generated(tmp_path / "stages", _HELD)
    blockers = AD.form_holdout_blockers(_preflight_config(tmp_path, inside), target)
    assert any("agent-reachable" in blocker for blocker in blockers)
    small = _generated(tmp_path, _HELD[:2])
    blockers = AD.form_holdout_blockers(_preflight_config(tmp_path, small), target)
    assert blockers == [f"form-scale holdout has 2 member(s); need {HC.MIN_MEMBERS}"]
    missing = AD.form_holdout_blockers(_preflight_config(tmp_path, tmp_path / "absent"), target)
    assert len(missing) == 1 and "absent or linked" in missing[0]


def test_private_spec_keeps_the_roster_out_of_the_invocation(tmp_path):
    spec = tmp_path / "form.yaml"
    spec.write_text(yaml.safe_dump({"generated_root": str(tmp_path / "run"), "applications": ["a", "b"]}))
    raw = {"form_holdout_generated_root": None, "form_holdout_applications": None, "form_holdout_family": None}
    fields = CLI._form_holdout_fields(spec, raw)
    assert fields == {
        "form_holdout_generated_root": tmp_path / "run",
        "form_holdout_applications": ("a", "b"),
    }
    assert raw == {}
    flags = CLI._form_holdout_fields(
        None,
        {
            "form_holdout_generated_root": tmp_path / "run",
            "form_holdout_applications": "a, b",
            "form_holdout_family": "PX",
        },
    )
    assert flags["form_holdout_applications"] == ("a", "b") and flags["form_holdout_family"] == "PX"
    absent = {"form_holdout_generated_root": None, "form_holdout_applications": None, "form_holdout_family": None}
    assert CLI._form_holdout_fields(None, absent) == {}


@pytest.mark.parametrize(
    "document",
    [{"generated_root": "/x"}, {"generated_root": "/x", "applications": ["a"], "members": ["leak"]}, ["a"]],
)
def test_malformed_private_spec_is_refused(tmp_path, document):
    spec = tmp_path / "form.yaml"
    spec.write_text(yaml.safe_dump(document))
    with pytest.raises(AD.ExperimentError):
        AD.load_form_holdout_spec(spec)


def test_half_given_form_flags_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        CLI._form_holdout_fields(
            None,
            {"form_holdout_generated_root": tmp_path, "form_holdout_applications": None, "form_holdout_family": None},
        )


def test_form_cells_follow_each_pk_cell_under_their_own_address(tmp_path):
    handoff = SimpleNamespace(corpus_root=tmp_path / "tuning", corpus_manifest_sha256="1" * 64, corpus_sha256="2" * 64)
    config = SimpleNamespace(experiment_id="exp", context=SimpleNamespace(measurement_root=tmp_path / "runs"))

    def reveal(name):
        return {
            "root": str(tmp_path / name),
            "manifest": str(tmp_path / name / "holdout_manifest.json"),
            "manifest_sha256": "3" * 64,
            "capsules_sha256": "4" * 64,
        }

    certificates = {name: SimpleNamespace(path=Path(name), sha256=name[0] * 64) for name in ("a", "b", "c")}
    cells = CTRL._measurement_cells(
        config,
        dict.fromkeys(AD.TRIALS, handoff),
        reveal("pk"),
        tuning_certificate=certificates["a"],
        heldout_certificate=certificates["b"],
        additional_heldout=[(AD.FORM_HOLDOUT_MEASUREMENT_LABEL, reveal("form"), certificates["c"])],
    )
    label = AD.FORM_HOLDOUT_MEASUREMENT_LABEL
    assert [(cell.stage, cell.phase, cell.run_id) for cell in cells[:3]] == [
        ("measurement:trial_00:tuning", "tuning", "exp__trial_00__tuning"),
        ("measurement:trial_00:held_out", "held_out", "exp__trial_00__held_out"),
        (f"measurement:trial_00:{label}", "held_out", f"exp__trial_00__{label}"),
    ]
    assert [cell.corpus_label for cell in cells[:3]] == ["tuning", "held_out", label]
    assert cells[2].corpus_root == tmp_path / "form" and cells[2].certificate is certificates["c"]
    assert len(cells) == 3 * len(AD.TRIALS)
    # One leading cell per corpus before the fan-out: three corpora, three cells.
    stages = [CTRL.ChildStage(cell.stage, lambda: None, lambda: None) for cell in cells]
    assert CTRL.baseline_lead_prefix(stages, [cell.corpus_label for cell in cells]) == 3


def _heldout_layers(tmp_path, gemms, *, mode=0o600):
    from merlin_experiments.phase0 import heldout_layers as HL

    document = {
        "schema": HL.SCHEMA,
        "networks": {
            "net": {"contractions": [{"M": m, "K": k, "N": n, "layer": f"l{i}"} for i, (m, k, n) in enumerate(gemms)]}
        },
    }
    path = tmp_path / "operator" / "heldout-layers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    path.chmod(mode)
    return path


def test_commit_and_reveal_refuse_a_member_that_is_a_heldout_network_layer(tmp_path):
    from merlin_experiments.phase0 import heldout_layers as HL

    root = _generated(tmp_path, _HELD)
    tuning = _tuning(tmp_path, [(2304, 72, 24)])
    layers = _heldout_layers(tmp_path, [_HELD[1]])
    with pytest.raises(HL.HeldoutLayerError, match=r"(?s)OPERATOR: 1 form-holdout member.*24x48x96 = net l0"):
        FH.commit_form_holdout(
            root,
            tmp_path / "commitment.json",
            tmp_path / "private",
            target="synthetic",
            applications=["cnn_h"],
            tuning_root=tuning,
            candidate_ids=["c0"],
            heldout_layers=layers,
        )
    assert not (tmp_path / "commitment.json").exists()
    # A clean cohort commits; the reveal re-checks against the operator's file of the day.
    clean = _heldout_layers(tmp_path / "clean", [(1, 24, 10)])
    paths = FH.commit_form_holdout(
        root,
        tmp_path / "commitment.json",
        tmp_path / "private",
        target="synthetic",
        applications=["cnn_h"],
        tuning_root=tuning,
        candidate_ids=["c0"],
        heldout_layers=clean,
    )
    with pytest.raises(HL.HeldoutLayerError, match="form-holdout member"):
        FH.reveal_form_holdout(
            paths["public_commitment"],
            paths["host_private_dir"],
            tmp_path / "reveal",
            candidate_seals={"c0": _seal(tmp_path, "c0")},
            heldout_layers=layers,
        )
    assert not (tmp_path / "reveal").exists()


def test_the_heldout_layer_file_is_operator_private(tmp_path):
    from merlin_experiments.phase0 import heldout_layers as HL

    with pytest.raises(HL.HeldoutLayerError, match="owner-only"):
        HL.load(_heldout_layers(tmp_path, [(1, 2, 3)], mode=0o644))
    path = _heldout_layers(tmp_path / "ok", [(1, 2, 3)])
    with pytest.raises(HL.HeldoutLayerError, match="outside the repository"):
        HL.load(path, repository=tmp_path)
    loaded = HL.load(path)
    assert loaded.gemm(1, 2, 3) == ("net", "l0") and loaded.gemm(3, 2, 1) is None
    assert set(loaded.summary()) == {"schema", "heldout_layer_shapes_sha256", "networks", "gemm_shapes", "conv_windows"}


def test_the_spec_carries_the_private_layer_file(tmp_path):
    spec = tmp_path / "form-holdout.yaml"
    spec.write_text(
        yaml.safe_dump(
            {"generated_root": str(tmp_path), "applications": ["a_h"], "heldout_layer_shapes": "/private/layers.json"}
        )
    )
    fields = AD.load_form_holdout_spec(spec)
    assert fields["form_holdout_heldout_layers"] == Path("/private/layers.json")
