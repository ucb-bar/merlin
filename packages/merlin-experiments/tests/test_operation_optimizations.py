"""Per-operation optimization principles: the reviewed data file, family derivation, staging and links."""

from __future__ import annotations

import copy
import importlib.util
import json

import pytest
import yaml
from merlin_experiments.phase1 import capability_inventory as C
from merlin_experiments.phase1 import operation_optimizations as O
from merlin_experiments.phase1.feedback import performance_brief as PB

from merlin.common.paths import repo_root

ROOT = repo_root()
CATALOG = json.loads((ROOT / O.CATALOG).read_text())
LOOP = "loop_descriptor"


def _inputs(*shapes, roles=None):
    roles = roles or ["input"] + ["weight"] * (len(shapes) - 1)
    return [
        {"name": f"T{i}", "role": role, "shape": list(shape), "dtype": "i8"}
        for i, (shape, role) in enumerate(zip(shapes, roles, strict=True))
    ]


def matmul(m, k=32, n=16, **attrs):
    return {
        "name": f"mm_m{m}",
        "kind": "layer",
        "label": "public",
        "inputs": _inputs((m, k), (k, n)),
        "operation": {"op": "matmul", "attributes": {"lhs": "T0", "weight": "T1", **attrs}},
    }


def conv():
    return {
        "name": "cv",
        "kind": "layer",
        "label": "public",
        "inputs": _inputs((1, 8, 8, 3), (3, 3, 3, 16)),
        "operation": {
            "op": "conv2d",
            "attributes": {
                "ifm": "T0",
                "weight": "T1",
                "layout": "NHWC_HWIO",
                "kh": 3,
                "kw": 3,
                "stride": [1, 1],
                "padding": [1, 1, 1, 1],
            },
        },
    }


def single(op, **extra):
    return {
        "name": op,
        "kind": "model_slice",
        "label": "public",
        "inputs": _inputs((4, 16), roles=["input"]),
        "operation": {"op": op, "attributes": {}},
        **extra,
    }


def program():
    return {
        "name": "prog",
        "kind": "model_slice",
        "label": "public",
        "inputs": _inputs((16, 32), (32, 16), (32, 8)),
        "operation": {
            "op": "component_program",
            "attributes": {
                "program": {
                    "nodes": [
                        {"name": "p0", "op": "matmul", "inputs": ["T0", "T1"]},
                        {"name": "p1", "op": "matmul", "inputs": ["T0", "T2"]},
                    ]
                }
            },
        },
    }


# --- the reviewed data file -------------------------------------------------------------------------


def test_catalog_validates_against_its_schema_and_cross_references():
    assert O.validate(copy.deepcopy(CATALOG), repo=ROOT)["schema"] == O.SCHEMA
    assert [entry["id"] for entry in CATALOG["families"]] == list(O.FAMILIES)
    assert CATALOG["review"]["status"] == "reviewed"


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c["families"][0]["mechanisms"].append({"kind": "catalog", "id": "no_such_pass"}), "no_such_pass"),
        (lambda c: c["families"][0]["mechanisms"].append({"kind": "skill", "id": "no-such-skill"}), "no-such-skill"),
        (lambda c: c["families"][0]["mechanisms"].append({"kind": "recipe", "id": "heavy-pass-recipes/x.py"}), "x.py"),
        (lambda c: c["families"][0]["principles"][0].update(uses_instruction_roles=["teleport"]), "teleport"),
        (lambda c: c["families"][1].update(id="contraction"), "unique"),
        (lambda c: c["families"][0].update(related=["no_family"]), "no_family"),
        (lambda c: c["families"][0].update(threshold=4), "Additional properties"),
        (lambda c: c["general"]["principles"].append(dict(c["general"]["principles"][0])), "unique"),
    ],
)
def test_validation_refuses_broken_references_and_schema_violations(edit, message):
    catalog = copy.deepcopy(CATALOG)
    edit(catalog)
    with pytest.raises(ValueError, match=message):
        O.validate(catalog, repo=ROOT)


def _texts(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _texts(item)


def test_catalog_text_names_no_target_run_or_measurement():
    spec = importlib.util.spec_from_file_location("_roster", ROOT / "build_tools" / "scripts" / "_target_roster.py")
    roster = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roster)
    names = {name.lower() for name in roster.target_names(ROOT)}
    assert names
    words = set()
    for text in _texts([CATALOG["scope"], CATALOG["claim_policy"], CATALOG["general"], CATALOG["families"]]):
        lowered = text.lower()
        assert "fresh" not in lowered and "%" not in lowered and " x " not in lowered, text
        words.update("".join(c if c.isalnum() or c == "_" else " " for c in lowered).split())
        # The only number a principle states is the definition of a single-row product.
        assert all(not c.isdigit() for c in lowered.replace("m=1", "")), text
    assert not words & names


# --- families from declarations ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "declaration, families",
    [
        (matmul(64), ("contraction",)),
        (matmul(1), ("matrix_vector",)),
        (matmul(64, epilogue=["relu"]), ("contraction", "elementwise_epilogue")),
        (matmul(1, acc_scale=0.5), ("matrix_vector", "elementwise_epilogue")),
        (conv(), ("convolution",)),
        (single("attention_qk"), ("attention",)),
        (single("softmax"), ("reduction_normalization",)),
        (single("rmsnorm"), ("reduction_normalization",)),
        (single("residual_add"), ("elementwise_epilogue",)),
        (single("movement"), ("data_movement",)),
        (program(), ("connected_chain", "contraction")),
        (
            single("model", kind="model", semantic={"semantic_family": "contraction"}),
            ("connected_chain", "contraction"),
        ),
        (single("not_an_operation"), ()),
        ({"name": "empty"}, ()),
        (None, ()),
    ],
)
def test_families_come_from_the_declaration(declaration, families):
    assert O.families_of(declaration) == families


def test_family_ignores_the_case_name():
    renamed = matmul(64)
    renamed["name"] = "conv2d_attention_decode_m1"
    assert O.families_of(renamed) == ("contraction",)


def test_sibling_products_sharing_an_operand_are_a_contraction_first():
    shared = matmul(16)
    shared["operation"]["attributes"]["matmuls"] = [{"lhs": "T0"}, {"lhs": "T0"}]
    assert O.families_of(shared) == ("contraction", "connected_chain")


# --- selection and staging --------------------------------------------------------------------------


def _roots(tmp_path):
    ws, run = tmp_path / "workspace", tmp_path / "run"
    for root in (ws, run):
        root.mkdir()
        (root / "TASK.md").write_text("Task\n")
    return ws, run


def test_staging_writes_only_the_present_families_and_one_task_section(tmp_path):
    ws, run = _roots(tmp_path)
    (ws / O.SUGGESTIONS).write_text(json.dumps({"all_mechanisms": [{"id": "capacity_tiles", "status": "supported"}]}))
    record = O.stage(ws, run, declarations=[matmul(64), matmul(32), conv()], policy={}, repo=ROOT)
    assert record["families_present"] == {"contraction": 2, "convolution": 1}
    for root in (ws, run):
        staged = json.loads((root / O.FILENAME).read_text())
        assert staged["schema"] == O.SELECTION_SCHEMA
        assert [entry["id"] for entry in staged["entries"]] == ["contraction", "convolution"]
        assert staged["general"]["principles"] and "matrix_vector" in staged["families_not_in_corpus"]
        assert (root / "TASK.md").read_text().count(O.HEADING) == 1
    assert (ws / "TASK.md").read_bytes() == (run / "TASK.md").read_bytes()
    contraction = staged["entries"][0]
    # related families are kept only when they are staged too
    assert all(family in ("contraction", "convolution") for family in contraction["related"])
    tiles = next(m for m in contraction["mechanisms"] if m["id"] == "capacity_tiles")
    assert tiles["api"] == "merlin.compile.capacity:capacity_fit" and tiles["capability_status"] == "supported"
    assert "uses_instruction_roles" not in json.dumps(staged)


def test_prohibited_roles_are_read_from_the_policy_field_and_filter_every_entry(tmp_path):
    allowed = O.select(CATALOG, {"contraction": 1, "convolution": 1})
    assert any(p["id"] == "loop_descriptor_candidate" for e in allowed["entries"] for p in e["principles"])
    for policy in (
        {O.ROLES_FIELD: [LOOP]},
        {"policy": {O.ROLES_FIELD: [LOOP]}},
        {"instruction_policy": {O.ROLES_FIELD: [LOOP, "sync"]}},
    ):
        assert LOOP in O.prohibited_roles(policy)
    arm = {"performance": {"arms": {"candidate": {"instruction_policy": {O.ROLES_FIELD: [LOOP]}}}}}
    assert O.prohibited_roles({}, None, arm) == {LOOP}
    assert O.prohibited_roles({O.ROLES_FIELD: "loop_descriptor"}) == frozenset()

    ws, run = _roots(tmp_path)
    O.stage(ws, run, declarations=[matmul(64), conv()], policy={O.ROLES_FIELD: [LOOP]}, repo=ROOT)
    staged = json.loads((ws / O.FILENAME).read_text())
    assert staged["prohibited_instruction_roles"] == [LOOP]
    assert {row["principle"] for row in staged["omitted_for_policy"]} == {"loop_descriptor_candidate"}
    text = json.dumps({key: staged[key] for key in ("general", "entries")}).lower()
    assert "loop nest" not in text and "hardware loop" not in text and "sequencer" not in text
    assert "prohibits" in (ws / "TASK.md").read_text().split(O.HEADING)[1]


def test_a_role_mechanism_is_filtered_like_a_principle():
    catalog = copy.deepcopy(CATALOG)
    catalog["families"][0]["mechanisms"].append({"kind": "skill", "id": "fusion", "uses_instruction_roles": [LOOP]})
    staged = O.select(catalog, {"contraction": 1}, prohibited=[LOOP])
    assert {"entry": "contraction", "mechanism": "fusion"} in staged["omitted_for_policy"]
    assert all(m["id"] != "fusion" for m in staged["entries"][0]["mechanisms"])


def test_staging_failure_is_reported_and_never_raises(tmp_path, capsys):
    ws, run = _roots(tmp_path)
    assert O.stage_selected(ws, run, declarations=[matmul(4)], repo=None) is None
    assert O.stage_selected(ws, run, declarations=[matmul(4)], repo=tmp_path) is None
    assert "operation-optimizations" in capsys.readouterr().out
    assert (ws / "TASK.md").read_text() == "Task\n" and not (ws / O.FILENAME).exists()


def test_capability_inventory_stages_the_principles_for_its_roster(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        C,
        "build",
        lambda native, scalability, programs, policy: {"total": len(native) + len(scalability), "cohorts": {}},
    )
    monkeypatch.setattr(C, "stage", lambda workspace, run_dir, inventory: inventory)
    monkeypatch.setattr(O, "stage_selected", lambda workspace, run_dir, **kw: seen.update(kw))
    from types import SimpleNamespace

    from merlin_experiments.phase1 import joint_control
    from merlin_experiments.phase1.feedback import public_programs, rolling_scalability

    from merlin.targetgen import capsule_common

    policy = {O.ROLES_FIELD: [LOOP]}
    monkeypatch.setattr(joint_control, "selected", lambda: {"policy": policy})
    monkeypatch.setattr(capsule_common, "discover_capsules", lambda *a, **kw: [matmul(64)])
    monkeypatch.setattr(capsule_common, "load_capsule", lambda *a, **kw: conv())
    monkeypatch.setattr(
        rolling_scalability, "Selection", lambda context: SimpleNamespace(cases={"cv": tmp_path}, verify=lambda: None)
    )
    monkeypatch.setattr(public_programs, "selected", lambda context: None)
    C.stage_selected(
        tmp_path,
        tmp_path,
        context=SimpleNamespace(repo=ROOT),
        public_roots=[tmp_path],
        contract=tmp_path,
        excluded=lambda names: [],
    )
    assert [d["name"] for d in seen["declarations"]] == ["mm_m64", "cv"]
    assert seen["policy"] is policy and seen["repo"] == ROOT


# --- optimization_next linkage ----------------------------------------------------------------------


def _declare(ws, declaration):
    folder = ws / "layers" / declaration["name"]
    folder.mkdir(parents=True)
    (folder / "capsule.yaml").write_text(yaml.safe_dump(declaration))


def test_optimization_next_links_each_case_to_its_staged_entry(tmp_path):
    ws, run = _roots(tmp_path)
    gemv, epilogue, norm = matmul(1), matmul(48, epilogue=["relu"]), single("softmax")
    gemv["name"], epilogue["name"], norm["name"] = "case_a", "case_b", "case_c"
    for declaration in (gemv, epilogue, norm):
        _declare(ws, declaration)
    # softmax is not in the staged corpus: its case gets no link
    O.stage(ws, run, declarations=[gemv, epilogue], policy={}, repo=ROOT)
    rows = [{"capsule": name, "pass": True} for name in ("case_a", "case_b", "case_c")]
    document = PB.attach({"per_capsule": rows}, rows, workspace=ws)
    links = {row["capsule"]: (row.get("optimization_next") or {}).get("operation") for row in document["per_capsule"]}
    assert links["case_a"] == {
        "family": "matrix_vector",
        "entry": "matrix_vector",
        "file": O.FILENAME,
        "focus": next(e["focus"] for e in CATALOG["families"] if e["id"] == "matrix_vector"),
        "panel": "python agent_selfcheck.py --panel gemv",
    }
    assert links["case_b"]["entry"] == "contraction" and links["case_b"]["also"] == ["elementwise_epilogue"]
    assert links["case_c"] is None
    staged = json.loads((ws / O.FILENAME).read_text())
    assert {link["entry"] for link in links.values() if link} <= {entry["id"] for entry in staged["entries"]}


def test_no_staged_file_means_no_link(tmp_path):
    ws, _ = _roots(tmp_path)
    _declare(ws, matmul(64))
    rows = [{"capsule": "mm_m64", "pass": True}]
    document = PB.attach({"per_capsule": rows}, rows, workspace=ws)
    assert "operation" not in (document["per_capsule"][0].get("optimization_next") or {})


# --- composition routes and source-composition requests -----------------------------------------------


def test_task_section_names_the_composition_command_and_every_request_option(tmp_path):
    from author_skill_fixtures import load_script

    routes = load_script("heavy-pass-recipes", "composition_routes")
    assert O.SOURCE_COMPOSITION_OPTIONS == tuple(name for name, _ in routes.REQUESTS)
    assert O.COMPOSITION_COMMAND == routes.discovery_command("NAME")
    ws, run = _roots(tmp_path)
    O.stage(ws, run, declarations=[matmul(64)], policy={}, repo=ROOT)
    section = (ws / "TASK.md").read_text().split(O.HEADING)[1]
    assert O.COMPOSITION_COMMAND in section and "submission/source_composition.json" in section
    assert all(f"`{name}`" in section for name in O.SOURCE_COMPOSITION_OPTIONS)
    assert "optimization_next.composition" in section and "reference/source-composition.md" in section
    assert "speedup" in section


def test_chain_families_point_at_the_composition_routes_recipe():
    recipe = "heavy-pass-recipes/composition_routes.py"
    entries = {entry["id"]: entry for entry in CATALOG["families"]}
    for name in ("connected_chain", "elementwise_epilogue", "convolution", "attention"):
        assert any(m["kind"] == "recipe" and m["id"] == recipe for m in entries[name]["mechanisms"]), name
    assert any(m["id"] == recipe for m in CATALOG["general"]["mechanisms"])
    assert "chains_have_composition_routes" in {p["id"] for p in CATALOG["general"]["principles"]}
    requests = next(p for p in entries["connected_chain"]["principles"] if p["id"] == "source_composition_requests")
    assert "requested, selected and applied" in requests["text"]
    staged = O.select(CATALOG, {"connected_chain": 1})
    assert any(m["id"] == recipe for m in staged["entries"][0]["mechanisms"])
