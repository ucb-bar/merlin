"""Closed numerical extension intake preserves actual companion source semantics."""

import copy
import hashlib

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.target.spec import SpecError, load_spec
from merlin.target.spec.source import parse
from merlin.targetgen.quantization_spec import build_quantization_contract, capture_recipe_candidates
from merlin.targetgen.software_spec import load_software_spec, software_spec_identity, validate_software_spec


@pytest.fixture
def active_source():
    companion = repo_root().parent / "merlin-backends" / "mx-gemmini-mlir"
    path = companion / "mx_gemmini_support" / "contracts" / "software-spec-2029218-candidate.yaml"
    if not path.is_file():
        pytest.skip("actual OOT numerical source is not present")
    return path


def test_actual_source_legacy_and_compact_intake_preserve_semantics(active_source, tmp_path, monkeypatch):
    from merlin.target.spec import quantization as declarations
    from merlin.targetgen import quantization_spec as controller

    for name in (
        "validate_quantization_declarations",
        "validate_element_formats",
        "validate_output_requantization",
        "semantic_extensions",
        "_dtype",
        "_unknown",
    ):
        assert getattr(controller, name) is getattr(declarations, name)
    raw = active_source.read_bytes()
    authored = yaml.safe_load(raw)
    projected = load_software_spec(active_source)
    expanded = load_spec(active_source)
    assert expanded.software_spec() == projected
    assert projected["quantization"] == authored["quantization"]
    assert projected["numerical_semantics"] == authored["numerical_semantics"]
    for row in projected["operations"]:
        original = authored["operations"][row["id"]]
        if "numerical_contract" in original:
            assert row["numerical_contract"] == original["numerical_contract"]
    assert projected["transfer_contracts"] == validate_software_spec(authored)["transfer_contracts"]
    identity = software_spec_identity(active_source, projected)
    assert identity["sha256"] == hashlib.sha256(raw).hexdigest()
    record = expanded.record(expanded=True)
    tools = {row["module"] for row in record["tools"]}
    assert "merlin.target.spec.quantization" in tools
    assert "merlin.targetgen.quantization_spec" not in tools
    assert record["source_sha256"] == identity["sha256"]
    assert record["projection"]["status"] == "supported"
    assert all(row["execution_status"] == "unknown" for row in record["semantic_extensions"])
    assert record["software_spec"]["quantization"] == authored["quantization"]
    # Existing OOT arithmetic/RTL consumers keep their own exact source contract.
    monkeypatch.syspath_prepend(str(active_source.parents[2]))
    from mx_gemmini_support.contract import compile_contract

    contract = compile_contract(raw)
    assert contract["source_sha256"] == identity["sha256"]
    assert contract["output_requantization"] == authored["quantization"]["output_requantization"]
    for name, row in contract["formats"].items():
        assert row["element"] == authored["quantization"]["element_formats"][name]

    compact = copy.deepcopy(authored)
    compact["schema"] = "merlin.software_spec.v2"
    compact["definitions"] = {}
    for field in ("element_formats", "output_requantization"):
        compact["definitions"][field] = {"kind": f"quantization_{field}", "value": compact["quantization"][field]}
        compact["quantization"][field] = {"$ref": field}
    # v2 transfer records use an explicit declarations wrapper.
    compact["transfer_contracts"] = {"declarations": compact["transfer_contracts"]}
    path = tmp_path / "compact.yaml"
    path.write_text(yaml.safe_dump(compact, sort_keys=False), encoding="utf-8")
    selected = load_spec(path)
    assert selected.software_spec() == projected
    assert selected.record()["semantic_extensions"] == record["semantic_extensions"]
    path.write_bytes(path.read_bytes() + b"\n# changed selected source\n")
    with pytest.raises(SpecError, match="changed after expansion"):
        selected.software_spec()
    # Changing a well-formed selector changes custody/identity, without implying
    # the new selector is meaningful to the unchanged OOT compiler or hardware.
    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_bytes(raw)
    original_selection = load_spec(legacy_path)
    authored["quantization"]["output_requantization"]["bf16_readout_code"] += 1
    legacy_path.write_text(yaml.safe_dump(authored, sort_keys=False), encoding="utf-8")
    with pytest.raises(SpecError, match="changed after expansion"):
        original_selection.software_spec()
    changed = load_spec(legacy_path).record()
    assert changed["source_sha256"] != record["source_sha256"]
    assert changed["canonical_sha256"] != record["canonical_sha256"]
    assert changed["semantic_extensions"][1]["execution_status"] == "unknown"


def test_legacy_merges_preserve_overrides_and_refuse_duplicates_cycles_and_fanout(tmp_path):
    raw = (
        b"schema: merlin.software_spec.v1\n"
        b"base: &base {value: 1}\n"
        b"next: &next {<<: *base, value: 2}\n"
        b"last: {<<: *next, value: 3}\n"
    )
    assert parse(raw, "legacy", legacy=True)["last"] == {"value": 3}
    with pytest.raises(SpecError, match="unique strings"):
        parse(b"schema: merlin.software_spec.v1\nvalue: 1\nvalue: 2\n", "duplicate", legacy=True)
    with pytest.raises(SpecError, match="cyclic"):
        parse(b"value: &recursive [*recursive]\n", "cycle", legacy=True)
    rows = ["x0: &x0 [0]"]
    rows.extend(f"x{index}: &x{index} [*x{index - 1}, *x{index - 1}]" for index in range(1, 17))
    with pytest.raises(SpecError, match="alias/node/depth limit"):
        parse("\n".join(rows).encode(), "fanout", legacy=True)
    path = tmp_path / "compact-with-merge.yaml"
    path.write_bytes(raw.replace(b"merlin.software_spec.v1", b"merlin.software_spec.v2"))
    with pytest.raises(SpecError):
        load_spec(path)


def test_extension_semantics_never_become_generic_capture_recipe(active_source):
    spec = load_software_spec(active_source)
    snapshot = {
        "quantization_candidates": [
            {
                "format": "mxfp8",
                "unit": "matrix",
                "status": "derived",
                "recipe": {
                    "status": "derived",
                    "families": ["contraction"],
                    "weight": {"dtype": "mxfp8", "mode": "static", "granularity": "block", "block": 32},
                    "activation": {"dtype": "mxfp8", "mode": "dynamic", "granularity": "block", "block": 32},
                },
            }
        ],
        "readout_facets": [{"unit": "matrix", "accumulator_dtype": "bf16"}],
    }
    diagnostic = build_quantization_contract(spec, snapshot)
    assert diagnostic["semantic_extensions"]
    match = diagnostic["formats"][0]["hardware_matches"][0]
    assert match["status"] == "unknown"
    assert "quantization.element_formats.execution_adapter" in match["unknowns"]
    assert diagnostic["summary"]["format_candidates"] == 0
    assert capture_recipe_candidates(spec, diagnostic) == []
    # A forged/stale diagnostic match does not bypass the source-side boundary.
    match["status"] = "candidate"
    assert capture_recipe_candidates(spec, diagnostic) == []


@pytest.mark.parametrize(
    "field,key,value",
    [
        ("element", "unexpected", 1),
        ("element", "exponent_bits", True),
        ("element", "fraction_bits", -1),
        ("element", "exponent_bits", 64),
        ("element", "max_positive_code", 128),
        ("element", "max_finite", 449),
        ("element", "max_finite", float("inf")),
        ("output", "unexpected", True),
        ("output", "formats", ["mxfp8", "mxfp8"]),
        ("output", "formats", ["int8"]),
        ("output", "source", "input"),
        ("output", "scale_resident", 1),
        ("output", "requires_direct_consumer", "true"),
        ("output", "bf16_readout_code", True),
    ],
)
def test_malformed_extension_refuses(active_source, field, key, value):
    source = yaml.safe_load(active_source.read_bytes())
    row = (
        source["quantization"]["element_formats"]["mxfp8"]
        if field == "element"
        else source["quantization"]["output_requantization"]
    )
    row[key] = value
    with pytest.raises(ValueError):
        validate_software_spec(source)
