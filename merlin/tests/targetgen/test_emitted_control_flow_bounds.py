"""Unsupported aggregate/recursive source refuses before parser allocation."""

import json
import sys
from pathlib import Path

import pytest

import merlin
from merlin.common import invocation_record
from merlin.targetgen.contract.emitted_control_flow import observe_emitted_control_flow
from merlin.targetgen.contract.emitted_dataflow import DataflowUnavailable

_SCALAR = "module { llvm.func @entry(%p: !llvm.ptr) { llvm.return } }"


def observe(text=_SCALAR, **kwargs):
    return observe_emitted_control_flow(text, entry_symbol="entry", pointer_bits=64, **kwargs)


def prohibit_parser(monkeypatch):
    from xdsl.parser import Parser

    def unexpected_parse(*args, **kwargs):
        raise AssertionError("unsupported source reached shaped/recursive parser allocation")

    monkeypatch.setattr(Parser, "parse_module", unexpected_parse)


def test_explicit_source_limit_counts_utf8_bytes_before_parser(monkeypatch):
    source = _SCALAR + " // \u03bb"
    assert len(source) < len(source.encode())
    assert len(observe(source, max_source_bytes=len(source.encode())).blocks) == 1
    prohibit_parser(monkeypatch)
    with pytest.raises(DataflowUnavailable, match="source byte bound"):
        observe(source, max_source_bytes=len(source))


@pytest.mark.parametrize(
    "selection",
    [
        {"max_source_bytes": True},
        {"max_source_bytes": 0},
        {"max_source_bytes": -1},
        {"max_nesting": True},
        {"max_nesting": 0},
        {"max_nesting": -1},
    ],
)
def test_source_limits_require_actual_bounded_integer_selections(monkeypatch, selection):
    prohibit_parser(monkeypatch)
    with pytest.raises(DataflowUnavailable, match="bound"):
        observe(**selection)


@pytest.mark.parametrize(
    "value",
    [
        "dense<0> : tensor<100000000xi8>",
        'dense<"0x00"> : tensor<100000000xi8>',
        "dense\t<0> : tensor<100000000xi8>",
        "dense\n// actual token pair\n<0> : tensor<100000000xi8>",
        "dense_resource<unavailable> : tensor<100000000xi8>",
    ],
)
def test_unsupported_aggregate_tokens_never_reach_parser(monkeypatch, value):
    prohibit_parser(monkeypatch)
    with pytest.raises(DataflowUnavailable, match="aggregate literals"):
        observe("module attributes {payload = " + value + "} { llvm.func @entry(%p: !llvm.ptr) { llvm.return } }")


@pytest.mark.parametrize("dtype", ["i10000000000", "si10000000000", "ui10000000000", "i0"])
def test_already_unsupported_scalar_width_refuses_before_integer_attribute_verification(monkeypatch, dtype):
    prohibit_parser(monkeypatch)
    with pytest.raises(DataflowUnavailable, match="scalar width"):
        observe(
            "module { llvm.func @entry(%p: !llvm.ptr) { "
            f"%zero = llvm.mlir.constant(0 : {dtype}) : {dtype} llvm.return }} }}"
        )


def test_recursive_syntax_refuses_before_parser_and_small_actual_boundary_is_retained(monkeypatch):
    assert len(observe(max_nesting=2).blocks) == 1
    prohibit_parser(monkeypatch)
    nested = "module attributes {payload = " + "[" * 3000 + "0 : i8" + "]" * 3000 + "} {}"
    with pytest.raises(DataflowUnavailable, match="syntax nesting"):
        observe(nested)


def test_comments_strings_and_ssa_names_do_not_mint_aggregate_or_scalar_type_tokens():
    source = """// dense<0> : tensor<100000000xi8> i10000000000 [[[[
module { llvm.func @entry(%i10000000000: !llvm.ptr) {
  %value = llvm.mlir.constant(1 : i0008) : i0008
  llvm.inline_asm has_side_effects "dense<0> dense_resource<ignored> i10000000000 [[[[", "r" %value : (i8) -> ()
  llvm.return
} }"""
    result = observe(source)
    assert len(result.arguments) == 1
    assert next(row for row in result.values if row.definition is not None).bits == 8
    assert "instruction_effects" in result.unknown
    assert (
        "dense<0>"
        in dict(next(row.properties for row in result.operations if row.name == "llvm.inline_asm"))["asm_string"]
    )


@pytest.mark.parametrize("source", [_SCALAR + "\ud800", "module { @ }", "module { ]"])
def test_malformed_encoding_or_lexical_structure_refuses(source):
    with pytest.raises(DataflowUnavailable):
        observe(source)


def test_lexical_bounds_cannot_promote_a_remaining_parser_recursion_failure(monkeypatch):
    from xdsl.parser import Parser

    def recursive_parser(*args, **kwargs):
        raise RecursionError("a dialect parser exceeded its independent stack budget")

    monkeypatch.setattr(Parser, "parse_module", recursive_parser)
    with pytest.raises(DataflowUnavailable, match="parsed and verified"):
        observe()


@pytest.mark.parametrize("case", ["dense", "wide", "recursive"])
def test_actual_bounded_child_handles_original_memory_and_recursion_counterexamples(tmp_path, case):
    sources = {
        "dense": "module { llvm.mlir.global internal constant @zeros(dense<0> : tensor<100000000xi8>) "
        ": !llvm.array<100000000 x i8> llvm.func @entry(%p: !llvm.ptr) { llvm.return } }",
        "wide": "module { llvm.func @entry(%p: !llvm.ptr) { %zero = llvm.mlir.constant(0 : i10000000000) "
        ": i10000000000 llvm.return } }",
        "recursive": "module attributes {payload = " + "[" * 3000 + "0 : i8" + "]" * 3000 + "} {}",
    }
    source = tmp_path / "source.mlir"
    source.write_text(sources[case])
    code = """import sys,json,resource
from pathlib import Path
sys.path[:0]=json.loads(sys.argv[1])
from merlin.targetgen.contract.emitted_control_flow import observe_emitted_control_flow
from merlin.targetgen.contract.emitted_dataflow import DataflowUnavailable
resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
resource.setrlimit(resource.RLIMIT_CPU,(5,5))
try:
 observe_emitted_control_flow(Path(sys.argv[2]).read_text(),entry_symbol='entry',pointer_bits=64,max_blocks=1,max_operations=1)
except DataflowUnavailable as error:
 print(json.dumps({'outcome':'typed_refusal','reason':str(error)}))
else:
 raise AssertionError('unsupported source was observed')
"""
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    roots = [str(Path(path).parent) for path in merlin.__path__]
    result = invocation_record.run(
        (sys.executable, "-I", "-B", "-c", code, json.dumps(roots), str(source)),
        directory=tmp_path,
        stage="bounded_cfg_parser_refusal",
        inputs=(source,),
        dependencies=(Path(sys.executable),),
        outputs=(),
        env=environment,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout)["outcome"] == "typed_refusal"
    for path in tmp_path.rglob("invocation.json"):
        invocation_record.require_environment(path, environment=environment)
