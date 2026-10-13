"""Original selected-source negative ceilings retain every unknown/owner cut."""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.targetgen.compute_units import compute_units, semantic_capability_map
from merlin.targetgen.rtl import result_format_ceiling as C


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def selected(tmp_path):
    repo = tmp_path / "hardware"
    repo.mkdir()
    config = repo / "Config.scala"
    config.write_text(
        "class Selected extends Config(new DemoConfig ++ new Base)\n"
        "class DemoConfig[T <: Data](config: Any = Defaults.defaultConfig) extends Config()\n"
        "val defaultConfig = ArrayConfig[SInt, Float, Float](\n"
        " inputType = SInt(8.W), accType = SInt(32.W), mvinScale = None, accScale = None)\n"
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "Config.scala"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "hardware",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    firrtl = tmp_path / "Design.fir"
    firrtl.write_text(
        "FIRRTL version 3.0.0\ncircuit Root :\n  module Root :\n"
        "    input source : SInt<8>\n    input acc : SInt<32>\n"
        "    input enable : UInt<1>\n    input full : UInt<1>\n"
        "    output result : UInt<32>\n    output valid : UInt<1>\n"
        "    wire payload : UInt<32>\n    invalidate payload\n"
        "    when enable :\n      connect payload, pad(asUInt(source), 32)\n"
        "    when and(enable, full) :\n      connect payload, asUInt(acc)\n"
        "    connect result, payload\n    connect valid, enable\n"
    )
    origin, sums = tmp_path / "SOURCE.md", tmp_path / "SHA256SUMS"
    origin.write_text("CONFIG=Selected in " + str(repo) + " revision " + revision + "\n")
    sums.write_text(_sha(firrtl.read_bytes()) + "  Design.fir\n")
    contract = {
        "name": "sample",
        "compute_units": [
            {
                "name": "unit",
                "kind": "systolic",
                "dtypes": ["int8"],
                "semantic_capabilities": [{"family": "elementwise_map", "dtypes": ["int8"]}],
            }
        ],
    }
    # JSON is also YAML; keep the exact original contract bytes unchanged.
    cb = json.dumps(contract).encode()
    fb = json.dumps(
        {"inputs": {"target": "sample", "fir_path": str(firrtl), "fir_sha256": _sha(firrtl.read_bytes())}}
    ).encode()
    selection = {
        "schema": C.SCHEMA,
        "target": "sample",
        "original_contract_sha256": _sha(cb),
        "original_facts_sha256": _sha(fb),
        "sources": {
            name: {"path": str(path), "sha256": _sha(path.read_bytes())}
            for name, path in {"config": config, "origin": origin, "sums": sums}.items()
        },
        "origin": {
            "record_source": "origin",
            "checksums_source": "sums",
            "repositories": [{"name": "hw", "root": str(repo), "revision": revision, "sources": ["config"]}],
        },
        "configuration": {
            "top_source": "config",
            "top_symbol": "Selected",
            "provider_constructor": "DemoConfig",
            "provider_source": "config",
            "provider_symbol": "DemoConfig",
            "parameter": "config",
            "value_source": "config",
            "value_symbol": "Defaults.defaultConfig",
            "scalar_roles": {"element": "inputType", "accumulator": "accType"},
            "scale_fields": ["mvinScale", "accScale"],
            "integer_conversion_class": "IntegerConvert",
        },
        "firrtl": {"path": str(firrtl), "sha256": _sha(firrtl.read_bytes())},
        "families": ["elementwise_map"],
        "projection": {
            "modules": ["Root"],
            "endpoints": [{"module": "Root", "reference": "result", "valid_when": "valid"}],
            "terminals": [
                {"module": "Root", "reference": "source", "kind": "original_numeric_operand", "role": "element"},
                {"module": "Root", "reference": "acc", "kind": "original_numeric_operand", "role": "accumulator"},
            ],
        },
    }
    return cb, fb, selection


def _issue(selected):
    cb, fb, selection = selected
    raw = json.dumps(selection).encode()
    return C.verify_original_selection(
        original_contract_bytes=cb, original_facts_bytes=fb, selection_bytes=raw, expected_selection_sha256=_sha(raw)
    )


def _change_firrtl(selected, old, new):
    cb, fb, selection = selected
    path = Path(selection["firrtl"]["path"])
    path.write_text(path.read_text().replace(old, new))
    digest = _sha(path.read_bytes())
    selection["firrtl"]["sha256"] = digest
    facts = json.loads(fb)
    facts["inputs"]["fir_sha256"] = digest
    fb = json.dumps(facts).encode()
    selection["original_facts_sha256"] = _sha(fb)
    sums = Path(selection["sources"]["sums"]["path"])
    sums.write_text(digest + "  " + path.name + "\n")
    selection["sources"]["sums"]["sha256"] = _sha(sums.read_bytes())
    return cb, fb, selection


def test_complete_valid_native_outputs_supply_negative_evidence_only(selected):
    proof = _issue(selected)
    capabilities = semantic_capability_map(compute_units(json.loads(selected[0])))
    assert proof.record()["native_formats"] == ["int32", "int8"]
    assert proof.refusal("elementwise_map", "bf16", capabilities)["refusal"] == "result_dtype"
    assert proof.refusal("elementwise_map", "fp32", capabilities)["refusal"] == "result_dtype"
    assert proof.refusal("elementwise_map", "int32", capabilities) is None
    assert proof.refusal("elementwise_map", "int8", capabilities) is None
    assert proof.refusal(None, None, capabilities) is None
    assert proof.refusal("movement", "bf16", capabilities) is None
    assert capabilities["elementwise_map"].result_dtypes is None
    proof.revalidate_sources()


@pytest.mark.parametrize(
    "old,new",
    [
        ("connect valid, enable", "connect valid, UInt<1>(1)"),
        ("connect valid, enable", "when full :\n      invalidate payload\n    connect valid, enable"),
        ("connect result, payload", "connect result, external"),
        ("connect result, payload", "node cycle = cycle\n    connect result, cycle"),
        ("connect result, payload", "connect result, add(payload, UInt<32>(1))"),
    ],
)
def test_missing_undefined_external_and_unexplained_output_paths_refuse(selected, old, new):
    with pytest.raises(ValueError, match="UNKNOWN"):
        _issue(_change_firrtl(selected, old, new))


def test_valid_mux_masks_undefined_branch(selected):
    changed = _change_firrtl(selected, "connect result, payload", "connect result, mux(enable, payload, UInt<32>(0))")
    assert _issue(changed).record()["native_formats"] == ["int32", "int8"]


@pytest.mark.parametrize("which", ["contract", "facts", "selection", "config", "source_note", "source_sums"])
def test_original_identity_and_source_mutations_refuse(selected, which):
    cb, fb, selection = selected
    if which == "contract":
        cb += b" "
    elif which == "facts":
        fb += b" "
    elif which == "selection":
        selection["original_contract_sha256"] = "0" * 64
    else:
        key = {"config": "config", "source_note": "origin", "source_sums": "sums"}[which]
        path = Path(selection["sources"][key]["path"])
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        _issue((cb, fb, selection))


def test_floating_prototype_cannot_be_named_signed_from_carrier_width(selected):
    cb, fb, selection = selected
    path = Path(selection["sources"]["config"]["path"])
    text = path.read_text().replace("SInt(8.W)", "Float(8.W)")
    sources = {"config": text}
    with pytest.raises(ValueError, match="signed integer constructor"):
        C._configuration(selection, sources)


def test_issued_owner_cannot_be_constructed_copied_mutated_or_shared(selected):
    proof = _issue(selected)
    with pytest.raises(ValueError, match="live original"):
        replace(proof).record()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ValueError, match="live original"):
            pool.submit(proof.record).result()
    object.__setattr__(proof, "formats", ("bf16",))
    with pytest.raises(ValueError, match="live original"):
        proof.record()


def test_changed_sources_and_foreign_capability_map_refuse_after_issue(selected):
    proof = _issue(selected)
    with pytest.raises(ValueError, match="another original capability map"):
        proof.refusal(None, None, {})
    path = Path(selected[2]["sources"]["config"]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source bytes changed"):
        proof.revalidate_sources()


def test_inherited_process_and_expired_issuance_refuse(selected, monkeypatch):
    proof = _issue(selected)
    pid = C.os.getpid()
    with monkeypatch.context() as patch:
        patch.setattr(C.os, "getpid", lambda: pid + 1)
        with pytest.raises(ValueError, match="live original"):
            proof.record()
    del C._ISSUED[proof]
    with pytest.raises(ValueError, match="live original"):
        proof.revalidate_sources()


def test_source_file_pin_without_original_revision_binding_refuses(selected):
    selected[2]["origin"]["repositories"][0]["sources"] = []
    with pytest.raises(ValueError, match="complete original revision"):
        _issue(selected)


def test_post_issuance_firrtl_content_mutation_refuses(selected):
    proof = _issue(selected)
    path = Path(selected[2]["firrtl"]["path"])
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"SInt<8>", b"UInt<8>"))
    with pytest.raises(ValueError, match="source bytes changed"):
        proof.revalidate_sources()


def test_proof_modules_are_grader_only():
    from merlin.targetgen import tool_profile as P

    for name in ("merlin.targetgen.rtl.result_format_ceiling", "merlin.targetgen.rtl.result_format_validity"):
        assert name in P.AUTOMATIC_CHECKER_MODULES_V3
        assert name not in P.COMMON_PUBLIC_MODULES_V2
