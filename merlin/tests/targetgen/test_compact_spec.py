"""A selected compact authority composes existing records without hidden fallback."""

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from merlin.target.spec import SpecError, load_spec
from merlin.targetgen.software_spec import (
    admit_operation,
    load_software_spec,
    software_spec_identity,
    software_spec_references,
    validate_software_spec,
)


def _numerics():
    return {
        "model": {"engine": "integer_reference"},
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "readout_dtype": "i32",
        "subnormal_operand_flush": False,
        "overflow": "wrap_internal_mac",
        "internal_arithmetic": {
            "mac_result_bits": 20,
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
        },
        "readout": {
            "acc_scale_rounding": "half_even",
            "requant_rounding": "half_up",
            "hardware_integer_shift_rounding": "rne",
        },
    }


def _document():
    return {
        "schema": "merlin.software_spec.v2",
        "target": "fixture_target",
        "status": "reviewed",
        "definitions": {
            "numeric": {"kind": "numerical_semantics", "value": _numerics()},
            "signature": {
                "kind": "signature",
                "value": {"operand_dtypes": ["int8"], "accumulator_dtype": "i32", "ranks": [2]},
            },
            "compute": {
                "kind": "operation",
                "value": {"families": ["contraction"], "placement": "accelerator", "signature": {"$ref": "signature"}},
            },
        },
        "numerical_semantics": {"$ref": "numeric"},
        "operations": {"dot": {"$ref": "compute"}},
        "quantization": {
            "formats": [
                {
                    "operand_dtype": "int8",
                    "accumulator_dtype": "i32",
                    "eligible_operations": ["dot"],
                    "scale_encoding": "unknown",
                }
            ]
        },
    }


def _write(path, document):
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_lossless_composition_and_existing_operation_consumer(tmp_path):
    document = _document()
    path = _write(tmp_path / "spec.yaml", document)
    expanded = load_spec(path, target="fixture_target")
    projected = expanded.software_spec()
    expected = {
        "schema": "merlin.software_spec.v1",
        "target": document["target"],
        "status": "reviewed",
        "numerical_semantics": _numerics(),
        "operations": {
            "dot": {
                "families": ["contraction"],
                "placement": "accelerator",
                "signature": document["definitions"]["signature"]["value"],
            }
        },
        "quantization": document["quantization"],
    }
    assert projected == validate_software_spec(expected)
    assert load_software_spec(path) == projected
    assert projected["numerical_semantics"]["internal_arithmetic"]["mac_result_bits"] == 20
    assert projected["numerical_semantics"]["readout"]["requant_rounding"] == "half_up"
    assert projected["quantization"]["formats"][0]["scale_encoding"] == "unknown"
    decision = admit_operation(
        projected, "dot", {"operand_dtype": "int8", "accum_dtype": "i32", "rank": 2}, "accelerator"
    )
    assert decision["status"] == "admitted"
    record = expanded.record(expanded=True)
    assert record["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["software_spec"] == projected
    assert record["authored_text"] == path.read_text(encoding="utf-8")
    assert record["format_registry"]["aliases"]["i8"] == "int8"
    assert all(row["used"] for row in record["definitions"])
    record["software_spec"]["target"] = "mutated_copy"
    assert expanded.software_spec()["target"] == "fixture_target"
    assert "software_spec" not in expanded.record()


def test_pinned_imports_identity_and_fresh_source_custody(tmp_path):
    document = _document()
    library = {"schema": "merlin.software_spec_library.v1", "definitions": document.pop("definitions")}
    library_path = _write(tmp_path / "library.yaml", library)
    document["imports"] = {
        "numeric_library": {"path": "library.yaml", "sha256": hashlib.sha256(library_path.read_bytes()).hexdigest()}
    }
    document["numerical_semantics"] = {"$ref": "numeric_library.numeric"}
    document["operations"] = {"dot": {"$ref": "numeric_library.compute"}}
    path = _write(tmp_path / "spec.yaml", document)
    expanded = load_spec(path)
    projected = expanded.software_spec()
    identity = software_spec_identity(path, projected)
    assert identity["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["sha256"] != identity["source_sha256"]
    references = software_spec_references(path, projected)
    assert references["semantic_import_0"] == library_path.resolve()
    assert "semantic_format_registry" in references
    altered = copy.deepcopy(projected)
    altered["numerical_semantics"]["readout"]["requant_rounding"] = "rne"
    with pytest.raises(ValueError, match="differs from the current compact expansion"):
        software_spec_identity(path, altered)
    library_path.write_bytes(library_path.read_bytes() + b"\n# changed authored source\n")
    with pytest.raises(SpecError, match="changed after expansion"):
        expanded.software_spec()
    with pytest.raises(SpecError, match="differs from its SHA-256 pin"):
        software_spec_identity(path, projected)
    document["imports"]["numeric_library"]["sha256"] = hashlib.sha256(library_path.read_bytes()).hexdigest()
    _write(path, document)
    updated = software_spec_identity(path)
    assert updated["sha256"] != identity["sha256"]
    assert updated["canonical_sha256"] == identity["canonical_sha256"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d.update(pseudocode="acc += x * y"), "allowed"),
        (lambda d: d["operations"]["dot"].update(overrides={}), "allowed"),
        (lambda d: d.update(numerical_semantics={"$ref": "signature"}), "expected kind"),
        (lambda d: d.update(numerical_semantics={"$ref": "missing"}), "unknown semantic definition"),
        (lambda d: d["definitions"]["signature"]["value"].update(unchecked_tolerance=1), "allowed"),
        (
            lambda d: d["definitions"]["numeric"]["value"]["internal_arithmetic"].update(unknown_rounding="truncate"),
            "internal arithmetic",
        ),
        (lambda d: d["quantization"].update(methodology={}), "application methodology"),
        (lambda d: d["definitions"]["numeric"].update(value={"$ref": "numeric"}), "cyclic"),
    ],
)
def test_unknown_fields_and_incompatible_references_refuse(tmp_path, mutation, message):
    document = _document()
    mutation(document)
    with pytest.raises(ValueError, match=message):
        load_spec(_write(tmp_path / "spec.yaml", document))


def _computation():
    return {
        "kind": "linalg.generic",
        "indexing_maps": ["affine_map<(d0) -> (d0)>"] * 3,
        "iterator_types": ["parallel"],
        "scalar_body": {
            "arguments": ["i32", "i32", "i32"],
            "operations": [
                {
                    "op": "arith.addi",
                    "operands": ["arg:0", "arg:1"],
                    "results": ["i32"],
                    "effects": [],
                    "attributes": {},
                    "regions": 0,
                }
            ],
            "yields": ["op:0:0"],
            "effects": [],
        },
    }


def test_existing_typed_scalar_representation_is_visible_and_projection_refuses(tmp_path):
    from merlin.target.spec import computation as scalar_records
    from merlin.target.spec import instructions, signature
    from merlin.target.spec.source import Source

    document = _document()
    document["definitions"]["add_body"] = {"kind": "scalar_body", "value": _computation()["scalar_body"]}
    computation = _computation()
    computation["scalar_body"] = {"$ref": "add_body"}
    document["computations"] = {"add": computation}
    path = _write(tmp_path / "spec.yaml", document)
    expanded = load_spec(path)
    record = expanded.record(expanded=True)
    owners = {owner.__name__: owner for owner in (scalar_records, instructions, signature)}
    observed = {row["module"]: row["sha256"] for row in record["tools"]}
    selected = expanded.source_references()
    assert all(str(owner.__file__) in {str(path) for path in selected.values()} for owner in owners.values())
    for name, owner in owners.items():
        assert observed[name] == hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest()
    # Check stale-source refusal using private copies of the real owner bytes.
    # No live compiler source is mutated while other integration cases run.
    copies = {}
    for pin in expanded._tools:
        if pin.namespace in owners:
            copy_path = tmp_path / (pin.namespace.rsplit(".", 1)[-1] + ".py")
            copy_path.write_bytes(pin.raw)
            copies[pin.namespace] = Source(copy_path, pin.raw, pin.namespace)
    pinned = replace(expanded, _tools=tuple(copies.get(pin.namespace, pin) for pin in expanded._tools))
    pinned.verify_sources()
    for pin in copies.values():
        pin.path.write_bytes(pin.raw + b"\n# changed private owner observation\n")
        with pytest.raises(SpecError, match="source changed after expansion"):
            pinned.verify_sources()
        pin.path.write_bytes(pin.raw)
    assert record["computations"]["add"]["computation"]["scalar_body"]["operations"][0]["op"] == "arith.addi"
    assert record["projection"]["status"] == "unsupported"
    with pytest.raises(SpecError, match="no admitted v1 execution/proof projection"):
        expanded.software_spec()
    with pytest.raises(SpecError, match="no admitted v1 execution/proof projection"):
        load_software_spec(path)
    document["definitions"]["add_body"]["value"]["operations"][0]["ignored_extension"] = "must survive"
    with pytest.raises(SpecError, match="scalar operation"):
        load_spec(_write(path, document))


def test_duplicate_keys_registry_pin_and_ambient_overlay_refuse(tmp_path, monkeypatch):
    path = tmp_path / "spec.yaml"
    path.write_text("schema: merlin.software_spec.v2\nschema: merlin.software_spec.v1\n", encoding="utf-8")
    with pytest.raises(SpecError, match="unique strings"):
        load_spec(path)
    # A tiny alias graph has a large expanded value; inspect must bound the
    # expansion before canonical JSON construction, even for malformed input.
    rows = ["schema: merlin.software_spec.v2", "a0: &a0 [1, 2, 3]"]
    rows.extend(f"a{index}: &a{index} [" + ", ".join([f"*a{index - 1}"] * 10) + "]" for index in range(1, 7))
    path.write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(SpecError, match="node/depth limit"):
        load_spec(path)
    document = _document()
    document["format_registry_sha256"] = "0" * 64
    with pytest.raises(SpecError, match="registry differs"):
        load_spec(_write(path, document))
    document.pop("format_registry_sha256")
    monkeypatch.setenv("MERLIN_QUANT_FORMATS", str(tmp_path / "ambient.yaml"))
    with pytest.raises(SpecError, match="overlays"):
        load_spec(_write(path, document))


def test_unchanged_legacy_v1_inspection_and_raw_identity(tmp_path):
    document = _document()
    document = {
        "schema": "merlin.software_spec.v1",
        "target": document["target"],
        "numerical_semantics": _numerics(),
        "operations": {"dot": {"families": ["contraction"], "placement": "accelerator", "operand_dtypes": ["int8"]}},
    }
    path = _write(tmp_path / "legacy.yaml", document)
    assert load_spec(path).software_spec() == validate_software_spec(document)
    assert load_spec(path).record()["source_schema"] == "merlin.software_spec.v1"
    assert software_spec_identity(path)["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_composed_isa_uses_existing_fact_bound_resolution_and_effects(tmp_path):
    from merlin.targetgen.isa_spec import IsaSpecError, resolve_isa

    document = _document()
    pin = "1" * 64
    isa = {
        "schema": "merlin.isa_spec.v1",
        "namespace": "fixture",
        "facts_sha256": pin,
        "word_bits": {"fact": ["facts", "word_bits"]},
        "resources": [{"name": "configuration", "kind": "state", "unit": "word"}],
        "instructions": [
            {
                "name": "configure",
                "operands": [],
                "results": [],
                "immediates": [],
                "encoding": [
                    {
                        "offset": {"fact": ["facts", "offset"]},
                        "width": {"fact": ["facts", "word_bits"]},
                        "fixed": {"fact": ["facts", "encoding"]},
                    }
                ],
                "semantics": {"primitive": "configuration.write", "bindings": {}, "parameters": {"mode": "explicit"}},
                "effects": {"reads": [], "writes": ["configuration"]},
            }
        ],
    }
    document["definitions"]["instructions"] = {"kind": "isa", "value": isa}
    document["isa"] = {"$ref": "instructions"}
    path = _write(tmp_path / "spec.yaml", document)
    projected = load_spec(path).software_spec()
    assert projected["isa"] == isa
    facts = {"facts": {"target": "fixture_target", "word_bits": 4, "offset": 0, "encoding": 3}}
    resolved = resolve_isa(projected["isa"], facts, target="fixture_target", facts_sha256=pin)
    assert resolved.instruction("configure").effects_json == '{"reads":[],"writes":["configuration"]}'
    with pytest.raises(IsaSpecError, match="differ from the ISA declaration pin"):
        resolve_isa(projected["isa"], facts, target="fixture_target", facts_sha256="0" * 64)


def test_imported_relative_model_owner_and_legacy_base_inheritance_refuse(tmp_path):
    document = _document()
    definitions = document.pop("definitions")
    definitions["numeric"]["value"]["model"]["source_root_path"] = "model"
    library_path = _write(
        tmp_path / "library.yaml",
        {
            "schema": "merlin.software_spec_library.v1",
            "definitions": definitions,
        },
    )
    document["imports"] = {
        "library": {"path": "library.yaml", "sha256": hashlib.sha256(library_path.read_bytes()).hexdigest()}
    }
    document["numerical_semantics"] = {"$ref": "library.numeric"}
    document["operations"] = {"dot": {"$ref": "library.compute"}}
    path = _write(tmp_path / "spec.yaml", document)
    with pytest.raises(SpecError, match="relative model source_root_path"):
        load_spec(path)
    library_path = _write(library_path, {"schema": "merlin.software_spec.v1"})
    document["imports"]["library"]["sha256"] = hashlib.sha256(library_path.read_bytes()).hexdigest()
    with pytest.raises(SpecError, match="importing a complete legacy spec is unsupported"):
        load_spec(_write(path, document))
