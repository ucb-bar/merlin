"""Private synthetic hashes gate actual typed nodes before public promotion.

No protected source, graph or shape file is read by these tests. Passing this
publication selection establishes neither numerical nor runtime qualification.
"""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import test_component_host_curriculum as H
import test_phase0_m2m_preflight as P
import yaml
from merlin_experiments.phase0 import generation as G
from merlin_experiments.phase0 import operator_exclusion as E
from merlin_experiments.phase0.staging import STAGING_PREFIX, write_staged
from merlin_experiments.phase2 import corpus as C

from merlin.targetgen import component_program


def original():
    return {
        "schema": component_program.SCHEMA_V2,
        "inputs": [
            {"name": "A", "role": "input", "shape": [2, 3], "dtype": "operand"},
            {"name": "W", "role": "weight", "shape": [3, 2], "dtype": "operand"},
            {"name": "V", "role": "weight", "shape": [2, 4], "dtype": "operand"},
        ],
        "nodes": [
            {"name": "product", "op": "matmul", "inputs": ["A", "W"]},
            {"name": "narrow", "op": "cast", "inputs": ["product"], "dtype": "operand"},
            {"name": "final", "op": "matmul", "inputs": ["narrow", "V"]},
        ],
        "outputs": [{"name": "Y", "value": "final"}],
    }


def selected(tmp_path, *, sources=(), signatures=()):
    data = {
        "schema": E.SCHEMA,
        "excluded_source_sha256": list(sources),
        "excluded_contraction_sha256": list(signatures),
    }
    if not sources and not signatures:
        data["excluded_source_sha256"] = [hashlib.sha256(b"synthetic unrelated protected source").hexdigest()]
    path = tmp_path / "operator-private.json"
    path.write_text(json.dumps(data))
    path.chmod(0o600)
    return E.select(path, hashlib.sha256(path.read_bytes()).hexdigest()), path


def writer(root, *, cap=None, source=None):
    if cap is None:
        cap, source = H.capsule(original())
    cap = deepcopy(cap)
    cap.update(name="slice", label="dev", performance={"family": "CLOSED", "claim": "DIFFERENTIAL"})
    path = root / "_perf/slice"
    path.mkdir(parents=True)
    (path / "capsule.yaml").write_text(yaml.safe_dump(cap))
    (path / "capsule.interface.mlir").write_text(source)
    return path


def test_every_contraction_uses_actual_typed_shapes_dtypes_and_semantics():
    cap, _ = H.capsule(original())
    hashes = E.contraction_signature_hashes(cap)
    assert len(hashes) == 2 and hashes[0] != hashes[1]
    for index, (rows, reduction, cols) in enumerate(((2, 3, 2), (2, 2, 4))):
        single = original()
        single["inputs"] = [
            {"name": "X", "role": "input", "shape": [rows, reduction], "dtype": "operand"},
            {"name": "K", "role": "weight", "shape": [reduction, cols], "dtype": "operand"},
        ]
        single["nodes"] = [{"name": "renamed", "op": "matmul", "inputs": ["X", "K"]}]
        single["outputs"] = [{"name": "Z", "value": "renamed"}]
        standalone, _ = H.capsule(single)
        assert E.contraction_signature_hashes(standalone) == (hashes[index],)
    cap["numerical_semantics"]["overflow"] = "bounded_exact"
    assert E.contraction_signature_hashes(cap) != hashes
    cap["component_program"]["nodes"][0]["shape"] = [9, 9]
    with pytest.raises(ValueError, match=E.REFUSAL):
        E.contraction_signature_hashes(cap)


def test_synthetic_pass_promotes_without_private_selection_or_hashes(tmp_path):
    exclusion, path = selected(tmp_path)
    root = tmp_path / "public"
    result = write_staged(writer, root, member="_perf/slice", operator_exclusion=exclusion)
    assert result == root / "_perf/slice"
    private = json.loads(path.read_bytes())["excluded_source_sha256"][0]
    for product in result.rglob("*"):
        if product.is_file():
            data = product.read_text()
            assert private not in data and str(path) not in data and E.SCHEMA not in data
    assert str(path) not in repr(exclusion) and private not in repr(exclusion)
    with pytest.raises(AttributeError):
        exclusion._sha256 = "f" * 64
    assert not list(root.glob(STAGING_PREFIX + "*"))


@pytest.mark.parametrize("basis", ["source", "inner_contraction", "policy_drift", "source_drift", "untyped"])
def test_refusal_has_no_promoted_member_or_private_feedback(tmp_path, basis):
    cap, source = H.capsule(original())
    signatures = E.contraction_signature_hashes(cap)
    sources = [hashlib.sha256(source.encode()).hexdigest()] if basis == "source" else []
    exclusions = [signatures[1]] if basis == "inner_contraction" else []
    selection, path = selected(tmp_path, sources=sources, signatures=exclusions)
    root = tmp_path / "public"

    def create(stage):
        if basis == "policy_drift":
            path.write_text("changed private operator selection")
        if basis == "source_drift":
            return writer(stage, cap=cap, source=source + "// changed source\n")
        if basis == "untyped":
            cap.pop("component_program")
        return writer(stage, cap=cap, source=source)

    with pytest.raises(E.PublicationRefusal) as caught:
        write_staged(create, root, member="_perf/slice", operator_exclusion=selection)
    assert str(caught.value) == E.REFUSAL
    assert str(path) not in str(caught.value) and not any(
        value in str(caught.value) for value in signatures + tuple(sources)
    )
    assert not (root / "_perf/slice").exists()
    assert not list(root.rglob("capsule.yaml")) and not list(root.glob(STAGING_PREFIX + "*"))


def test_rejected_batch_promotes_none_of_its_members(tmp_path):
    cap, source = H.capsule(original())
    exclusion, _ = selected(tmp_path, signatures=E.contraction_signature_hashes(cap))
    root = tmp_path / "public"

    def create(stage):
        # A host-only candidate passes independently; the later closed graph refuses.
        host = original()
        host["inputs"] = host["inputs"][:1]
        host["nodes"] = [{"name": "moved", "op": "transpose", "inputs": ["A"]}]
        host["outputs"] = [{"name": "Y", "value": "moved"}]
        host_cap, host_source = H.capsule(host)
        first = writer(stage, cap=host_cap, source=host_source)
        first.rename(first.with_name("a_host"))
        return writer(stage, cap=cap, source=source)

    with pytest.raises(E.PublicationRefusal, match=E.REFUSAL):
        write_staged(create, root, operator_exclusion=exclusion)
    assert not list(root.rglob("capsule.yaml")) and not list(root.glob(STAGING_PREFIX + "*"))


def test_no_arbitrary_callback_or_private_file_in_publication_tree(tmp_path):
    with pytest.raises(ValueError, match=E.REFUSAL):
        write_staged(writer, tmp_path / "not-created", operator_exclusion=lambda _: True)
    assert not (tmp_path / "not-created").exists()
    root = tmp_path / "public"
    root.mkdir()
    selection, _ = selected(root)
    with pytest.raises(ValueError, match=E.REFUSAL):
        write_staged(writer, root, operator_exclusion=selection)
    assert not list(root.rglob("capsule.yaml"))


def test_ordinary_generator_refuses_without_matched_member_diagnostics(monkeypatch, tmp_path, capsys):
    cap, _ = H.capsule(original())
    exclusion, path = selected(tmp_path, signatures=E.contraction_signature_hashes(cap))
    entry = {
        "name": "candidate",
        "cat": "_perf",
        "op": "component_program",
        "program": original(),
        "numerical_semantics": cap["numerical_semantics"],
        "performance": {"family": "CLOSED"},
    }
    run, _, _ = P._generation_with_entries(monkeypatch, tmp_path, [entry])
    monkeypatch.setattr(
        G.CS,
        "derive_binding",
        lambda *args, **kwargs: H.corpus_spec.CorpusBinding(
            target="fixture",
            tile_dim=1,
            operand_dtype="int8",
            accum_dtype="i32",
            integer=True,
            tiers=[],
            compare="exact_int",
        ),
    )
    monkeypatch.setattr(G, "_write_capsule", lambda entry, binding, root, *args, **kwargs: writer(root))
    with pytest.raises(E.PublicationRefusal) as caught:
        run(operator_exclusion=exclusion)
    assert str(caught.value) == E.REFUSAL
    assert capsys.readouterr().out == ""
    assert not list((tmp_path / "capsules").rglob("capsule.yaml"))
    assert not (tmp_path / "capsules/MANIFEST.yaml").exists()
    assert str(path) not in str(caught.value)


def live(tmp_path):
    root = tmp_path / "live"
    writer(root)
    (root / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "generated": ["_perf/slice"],
                "hand_authored": [],
                "performance_generation": {
                    "fixture": {
                        "errors": [],
                        "phase": {"category": "_perf", "label": "dev", "included_in_functional_grade": False},
                    }
                },
            }
        )
    )
    return C.discover_performance_corpus(
        SimpleNamespace(target="fixture", capsule_corpus=root / "isa", graded_roots=lambda: [])
    )


def test_normal_seal_rechecks_private_selection_without_copying_it(tmp_path):
    corpus = live(tmp_path)
    selection, path = selected(tmp_path)
    frozen = C.freeze_performance_corpus(corpus, tmp_path / "frozen", operator_exclusion=selection)
    C.verify_frozen_performance_corpus(frozen)
    private = path.read_text()
    assert not any(product.name == path.name for product in frozen.root.rglob("*"))
    assert private not in frozen.manifest_path.read_text()
    path.write_text("drift after original selected pass")
    with pytest.raises(ValueError, match=E.REFUSAL):
        C.freeze_performance_corpus(corpus, tmp_path / "refused", operator_exclusion=selection)
    assert not (tmp_path / "refused").exists()
