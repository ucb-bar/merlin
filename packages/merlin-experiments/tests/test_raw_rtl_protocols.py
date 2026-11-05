"""Revised information boundaries cannot silently reuse the historical inputs."""
import copy
import hashlib

import pytest

from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1 import neutral_public_contracts as N
from merlin_experiments.phase1 import raw_rtl_inputs as R


def test_raw_el1_has_no_framework_sources_tools_experts_or_extracted_declarations():
    profile = A.resolve("xs-el-raw-rtl-v2/el1")
    assert profile.entries == () and profile.tools == ()
    assert profile.raw_kernel_references is False
    assert profile.disclosure == dict(raw_rtl=True, rtl_facts=False, software_spec=False,
        derived_target_contract=False, derived_test_metadata=False, input_format="neutral_math_v1")
    assert "merlin_iface" not in A.task(profile)
    assert "C++/MLIR build block" in A.task(profile)
    assert all(A.resolve(name).disclosure is None for name in (*A.SCRATCH_NAMES, *A.NO_EXPERT_NAMES))


def test_raw_el2_exports_only_semantic_frontend_readers(tmp_path):
    from merlin.common.paths import repo_root
    from merlin_experiments.phase1 import raw_interface_sources as I
    profile = A.resolve("xs-el-raw-rtl-v2/el2")
    selection = A.prepare_sources(repo_root(), tmp_path/"sources", profile=profile,
                                  review_id="test revised frontend boundary")
    A.verify_sources(profile, selection)
    assert {module for _, _, module in selection.members} == I.MODULES
    assert len(selection.members) == 15
    assert profile.tools == ()
    assert profile.disclosure["input_format"] == "shared_interface_v1"
    assert profile.disclosure["rtl_facts"] is False
    assert profile.disclosure["software_spec"] is False
    assert "already-lowered test inputs" in A.task(profile)
    assert "C++/MLIR build block" in A.task(profile)


@pytest.fixture
def bundle(tmp_path):
    facts = tmp_path / "private/facts.json"
    facts.parent.mkdir()
    facts.write_text('{"private":true}')
    raw = tmp_path / "public/target.fir"
    raw.parent.mkdir()
    raw.write_text("FIRRTL version 3.3.0\n")
    view = tmp_path / "public/tests"
    view.mkdir()
    aliases = N.FILES | {N.INPUT_DOCUMENT} | {"schemas/" + name for name in N.SCHEMAS}
    return dict(allowed=[dict(path=str(raw), mode="ro"), dict(path=str(view), mode="ro")],
        selected_raw_rtl_files=[dict(path=str(raw), sha256=hashlib.sha256(raw.read_bytes()).hexdigest())],
        judge_rtl_facts_file=dict(path=str(facts), sha256=hashlib.sha256(facts.read_bytes()).hexdigest()),
        neutral_public_contract_files=[dict(alias=name, path=str(tmp_path/"protocol"/name), sha256="a"*64)
                                       for name in sorted(aliases)],
        author_input_views=[dict(source=str(tmp_path/"original/tests"), destination=str(view), sha256="b"*64)])


def test_relocated_numerical_view_requires_private_originals_not_regrants(bundle):
    profile = A.resolve("xs-el-raw-rtl-v2/el1")
    R.verify(bundle, profile=profile)
    for broad in (False, True):
        changed = copy.deepcopy(bundle)
        from pathlib import Path
        path = Path(changed["author_input_views"][0]["source"])
        changed["allowed"].append(dict(path=str(path.parent if broad else path), mode="ro"))
        with pytest.raises(ValueError, match="withheld derived inputs"):
            R.verify(changed, profile=profile)


@pytest.mark.parametrize("alias", ["facts.json", "../private", "ordinary/facts.json"])
def test_private_facts_cannot_be_regranted_under_a_new_name(bundle, alias):
    changed = copy.deepcopy(bundle)
    changed["allowed"].append(dict(path=changed["judge_rtl_facts_file"]["path"], mode="ro", **{"as": alias}))
    with pytest.raises(ValueError, match="withheld derived inputs"):
        R.verify(changed, profile=A.resolve("xs-el-raw-rtl-v2/el1"))


def test_original_dialect_contracts_and_missing_views_are_refused(bundle):
    profile = A.resolve("xs-el-raw-rtl-v2/el1")
    changed = copy.deepcopy(bundle)
    changed["allowed"].append(dict(path="merlin/contract/interface_grammar.md", mode="ro"))
    with pytest.raises(ValueError, match="original interface input contracts"):
        R.verify(changed, profile=profile)
    bundle.pop("author_input_views")
    with pytest.raises(ValueError, match="numerical input views"):
        R.verify(bundle, profile=profile)


def test_raw_hardware_requires_original_consistent_pin(tmp_path):
    import json
    raw = tmp_path / "target.fir"
    raw.write_text("FIRRTL version 3.3.0\n")
    facts = tmp_path / "facts.json"
    doc = dict(source_consistency=dict(status="verified"), inputs=dict(firrtl_inputs=[
        dict(path=str(raw), sha256=hashlib.sha256(raw.read_bytes()).hexdigest())]))
    facts.write_text(json.dumps(doc))
    assert R._raw_sources(facts) == [raw]
    raw.write_text("changed hardware")
    with pytest.raises(ValueError, match="original selected pin"):
        R._raw_sources(facts)


def test_neutral_release_keeps_private_judge_and_canonical_author_view_separate(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import yaml
    from merlin.common.paths import repo_root
    from merlin.targetgen.sandbox import bwrap as B
    from merlin_experiments.phase1 import neutral_program_inputs as Q
    from merlin_experiments.phase1 import public_contract_preflight as P
    from merlin_experiments.phase1.public_contract_staging import PUBLIC_FILES

    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    repo = repo_root()
    original = tmp_path/"original"
    original.mkdir()
    raw = original/"target.fir"
    raw.write_text("FIRRTL version 3.3.0\n")
    facts = original/"rtl_facts/facts.json"
    facts.parent.mkdir()
    facts.write_text(json.dumps(dict(source_consistency=dict(status="verified"),
        inputs=dict(firrtl_inputs=[dict(path=str(raw), sha256=hashlib.sha256(raw.read_bytes()).hexdigest())]))))
    capsule = original/"tests/case"
    capsule.mkdir(parents=True)
    source = capsule/"capsule.interface.mlir"
    source.write_text('''module attributes {merlin_iface.version = "0.1", merlin_iface.target = "test", merlin_iface.abi_version = "0.1"} {
  %A = merlin_iface.tensor {name = "A", role = "input"} : tensor<1x1xi8>
  %B = merlin_iface.tensor {name = "B", role = "weight"} : tensor<1x1xi8>
  %R = merlin_iface.resident_pack %B {layout = "packed_rhs"} : (tensor<1x1xi8>) -> !merlin_iface.resident
  %acc = merlin_iface.matmul %A, %R : (tensor<1x1xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y = merlin_iface.commit %acc {name = "Y", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<1x1xi32>
  merlin_iface.evict %R : (!merlin_iface.resident) -> ()
}
''')
    (capsule/"capsule.yaml").write_text(yaml.safe_dump(dict(name="case", interface_mlir=source.name,
        numeric_policy=dict(compare="exact_int"), software_screen=dict(secret="WITHHELD_FACT_MARKER"))))
    (capsule/"golden.npz").write_bytes(b"PRIVATE_ANSWER_MARKER")
    contract = original/"target_contract.yaml"
    contract.write_text(yaml.safe_dump(dict(secret="WITHHELD_FACT_MARKER",
        name="test", harness_abi=dict(entry_symbol="selected_runtime_entry"))))
    software = original/"software-spec.yaml"
    software.write_text("secret: WITHHELD_FACT_MARKER\n")
    common = [dict(path="merlin/contract/"+name, mode="ro") for name in sorted(PUBLIC_FILES)]
    common += [dict(path="merlin/contract/schemas/", mode="ro"),
        dict(path=str(original/"tests"), mode="ro", note="capsule corpus"),
        dict(path=str(facts.parent), mode="ro"),
        dict(path=str(contract), mode="ro", note="selected capability contract (ALL arms)"),
        dict(path=str(software), mode="ro", note="admitted software semantics")]
    selected = dict(allowed=common, host_inputs=[], selected_rtl_facts_file=str(facts))
    before = {path: path.read_bytes() for path in original.rglob("*") if path.is_file()}
    experiment = tmp_path/"experiment"
    experiment.mkdir()
    profile = A.resolve("xs-el-raw-rtl-v2/el1")
    result, public = R.prepare(selected, common, profile=profile, experiment=experiment,
                               repo=repo, corpus_paths=[str(original/"tests")])
    result.update(allowed=public, author_profile=profile.record(), tools=[])
    public_paths = {row["path"] for row in public}
    result["host_inputs"] += [dict(path=row["path"]) for row in common if row["path"] not in public_paths]
    A.verify_bundle(profile, result)
    assert all(path.read_bytes() == data for path, data in before.items())
    view = Path(result["author_input_views"][0]["destination"])
    assert not (view/"case/golden.npz").exists()
    payload = (view/"case/capsule.interface.program.json").read_bytes()
    assert payload == Q.encode_source(source.read_bytes())
    assert b"merlin_iface" not in payload
    assert all(b"WITHHELD_FACT_MARKER" not in p.read_bytes() for p in view.rglob("*") if p.is_file())
    workspace = tmp_path/"workspace"
    workspace.mkdir()
    B.materialize_bundle_inputs(workspace, result, repo=repo)
    assert B.frozen_selected_rtl_facts(workspace, result, repo=repo).read_bytes() == before[facts]
    reads = P.expected_reads(workspace, result, repo=repo)
    assert "neutral_input_contract.md" in reads["expected_files"]
    assert not any("interface_grammar" in key or "interface_dialect" in key for key in reads["expected_files"])
    argv = B.base_argv(workspace, result, repo=repo, include_claude_home=False)
    assert not B.is_exposed(argv, source)
    assert not B.is_exposed(argv, facts)
    assert not B.is_exposed(argv, software)
    assert not B.is_exposed(argv, contract)
