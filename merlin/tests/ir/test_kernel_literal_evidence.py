"""Independent source-literal custody checks do not import the producer."""

from copy import deepcopy

import pytest

from merlin.frontends.kernel_literal_evidence import checked_literal_words, checked_trace_literal_words


def declaration(dtype="f32", shape=None):
    width, word = {"f16": (16, 0xFC01), "bf16": (16, 0xFF81), "f32": (32, 0xFF800001), "f64": (64, 0xFFF0000000000001)}[
        dtype
    ]
    data = word.to_bytes(width // 8, "little")
    type_ = dtype if shape is None else "tensor<" + "x".join(map(str, shape)) + "x" + dtype + ">"
    info = {
        "dtype": dtype,
        "type": type_,
        "word_bits": width,
        "encoding": "little-endian-IEEE-words",
        "words": data.hex(),
        "nan_count": 1,
        "shape": shape,
    }
    value = f"0x{word:X} : {dtype}" if shape is None else f'dense<"0x{data.hex().upper()}"> : {type_}'
    source = {
        "id": "ttir.op.0001",
        "opcode": "arith.constant",
        "output_types": [type_],
        "properties": {"value": value},
        "floating_literal_words": deepcopy(info),
    }
    report = {
        "kind": "source-floating-literal-words",
        "source_ids": [source["id"]],
        "location": 'loc("ttir.op.0001")',
        **info,
    }
    return [source], [report]


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("shape", [None, [4], [2, 2]])
def test_scalar_and_splat_storage_is_checked_independently(dtype, shape):
    records, reports = declaration(dtype, shape)
    checked_literal_words(records, reports)
    checked_trace_literal_words({"source_operations": records, "transformations": reports})


@pytest.mark.parametrize(
    "change",
    [
        "width",
        "encoding",
        "unknown-format",
        "type",
        "trailing-type",
        "shape",
        "bool-shape",
        "count",
        "finite",
        "word-count",
        "uppercase",
        "primitive",
        "primitive-type",
        "primitive-words",
        "missing",
        "duplicate",
        "source",
        "extra",
    ],
)
def test_changed_or_rehashed_declarations_do_not_supply_literal_semantics(change):
    records, reports = declaration("f16", [4])
    record, report = records[0], reports[0]
    if change == "width":
        report["word_bits"] = 16.0
    elif change == "encoding":
        report["encoding"] = "big-endian"
    elif change == "unknown-format":
        report["dtype"] = "fp16"
    elif change in ("type", "trailing-type"):
        report["type"] = "tensor<4xbf16>" if change == "type" else "tensor<4xf16> trailing"
    elif change == "shape":
        report["shape"] = [2, 2]
    elif change == "bool-shape":
        report["shape"] = [True]
    elif change == "count":
        report["nan_count"] = True
    elif change == "finite":
        report["words"] = "003c"
    elif change == "word-count":
        report["words"] *= 2
        report["nan_count"] = 2
    elif change == "uppercase":
        report["words"] = "01FC"
    elif change == "primitive":
        record["opcode"] = "arith.addf"
    elif change == "primitive-type":
        record["output_types"] = ["tensor<4xbf16>"]
    elif change == "primitive-words":
        report["words"] = "01fe"
        record["floating_literal_words"]["words"] = "01fe"
    elif change == "missing":
        reports.clear()
    elif change == "duplicate":
        reports.append(deepcopy(report))
    elif change == "source":
        report["source_ids"] = []
    elif change == "extra":
        report["grants_semantics"] = True
    with pytest.raises(ValueError, match="floating literal"):
        checked_literal_words(records, reports)


def test_functional_tensor_staging_checks_literal_words_before_early_return():
    from merlin.frontends.kernel_compile_inputs import _checked_memory_members

    records, reports = declaration()
    provenance = {"source_operations": records, "transformations": reports}

    def unexpected_member(name):
        pytest.fail("literal custody requires no evaluator or additional compiler member: " + name)

    assert _checked_memory_members(unexpected_member, provenance) == {}
    reports[0]["words"] = "0100c0ff"
    with pytest.raises(ValueError, match="floating literal"):
        _checked_memory_members(unexpected_member, provenance)


@pytest.mark.parametrize("change", [None, "primitive", "format", "missing"])
def test_protected_allocation_staging_reopens_rehashed_literal_declarations(change):
    import json

    from test_kernel_allocation_sources import encoded, fixture, record

    from merlin.frontends.kernel_allocation_evidence import checked_allocation_sources

    trace, files, bound = fixture(normalized=True, protected=True)
    records, reports = declaration("bf16", [4])
    report = json.loads(files["allocation-protection.json"])
    report["source_graph"]["operations"].extend(records)
    report["literal_conversions"] = reports
    if change == "primitive":
        records[0]["properties"]["value"] = 'dense<"0x81FF"> : tensor<4xf16>'
    elif change == "format":
        reports[0]["dtype"] = "f16"
    elif change == "missing":
        del report["literal_conversions"]
    files["allocation-protection.json"] = encoded(report)
    trace["source_argument_binding"]["source_capsule"]["artifacts"]["allocation-protection.json"] = record(
        files["allocation-protection.json"]
    )
    if change is None:
        assert checked_allocation_sources(files.__getitem__, trace, bound) == files
    else:
        with pytest.raises(ValueError, match="floating literal"):
            checked_allocation_sources(files.__getitem__, trace, bound)
