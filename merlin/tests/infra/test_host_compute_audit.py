"""An accelerator group whose host code works at element scale is vetoed; command-scale work is not."""

from __future__ import annotations

import pytest
from xdsl.context import Context
from xdsl.dialects import builtin, func, llvm
from xdsl.parser import Parser

from merlin.verify import host_compute_audit as HA

_ELEMENTS = 1024


def _loop(name: str, trips: int, body: str, *, bound: str | None = None) -> str:
    """One counted loop over ``trips`` iterations running ``body`` (which may use %i, %in, %out)."""
    limit = bound or f'%n = "llvm.mlir.constant"() <{{value = {trips} : i64}}> : () -> i64'
    return f'''
  "llvm.func"() <{{sym_name = "{name}", function_type = !llvm.func<void (!llvm.ptr, !llvm.ptr, i64)>}}> ({{
  ^entry(%in: !llvm.ptr, %out: !llvm.ptr, %dyn: i64):
    %c0 = "llvm.mlir.constant"() <{{value = 0 : i64}}> : () -> i64
    %c1 = "llvm.mlir.constant"() <{{value = 1 : i64}}> : () -> i64
    %scale = "llvm.mlir.constant"() <{{value = 5.000000e-01 : f32}}> : () -> f32
    {limit}
    "llvm.br"(%c0)[^header] : (i64) -> ()
  ^header(%i: i64):
    %cond = "llvm.icmp"(%i, %n) <{{predicate = 2 : i64}}> : (i64, i64) -> i1
    "llvm.cond_br"(%cond)[^body, ^exit] <{{operandSegmentSizes = array<i32: 1, 0, 0>}}> : (i1) -> ()
  ^body:
{body}
    %next = "llvm.add"(%i, %c1) : (i64, i64) -> i64
    "llvm.br"(%next)[^header] : (i64) -> ()
  ^exit:
    "llvm.return"() : () -> ()
  }}) : () -> ()'''


# The accelerator's work done on the host: load the accumulator, scale it in float, narrow, store.
_REQUANT = """    %src = "llvm.getelementptr"(%in, %i) <{elem_type = i32, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr
    %acc = "llvm.load"(%src) : (!llvm.ptr) -> i32
    %f = "llvm.sitofp"(%acc) : (i32) -> f32
    %s = "llvm.fmul"(%f, %scale) : (f32, f32) -> f32
    %q = "llvm.fptosi"(%s) : (f32) -> i8
    %dst = "llvm.getelementptr"(%out, %i) <{elem_type = i8, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr
    "llvm.store"(%q, %dst) : (i8, !llvm.ptr) -> ()"""
# Command issue: one tile per trip, address arithmetic only, the command itself opaque.
_ISSUE = """    %off = "llvm.mul"(%i, %c1) : (i64, i64) -> i64
    %addr = "llvm.add"(%off, %c1) : (i64, i64) -> i64
    "llvm.inline_asm"(%addr) <{asm_string = ".insn r 0x7b, 3, 2, x0, a0, a1", constraints = "r", has_side_effects}> : (i64) -> ()"""


def _module(*functions: str):
    ctx = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, llvm.LLVM, func.Func):
        ctx.load_dialect(dialect)
    text = '"builtin.module"() ({' + "".join(functions) + "\n}) : () -> ()"
    return Parser(ctx, text).parse_module()


def _site(symbol: str, placement: str = "unit0", elements: int = _ELEMENTS) -> HA.GroupSite:
    return HA.GroupSite(
        group=0,
        placement=placement,
        symbol=symbol,
        elements=elements,
        element_bytes=1,
        stages=("contraction", "quantize"),
    )


def test_command_issue_at_tile_scale_is_clean() -> None:
    report = HA.audit(_module(_loop("issue", 4, _ISSUE)), [_site("issue")])
    (row,) = report["groups"]
    assert row["verdict"] == HA.CLEAN, row
    assert row["arithmetic_per_element"] < 0.05 and row["host_value_arithmetic"] == 0
    assert (report["vetoed"], report["proven_clean"]) == (False, True)
    HA.require_clean(report)


def test_the_accelerators_arithmetic_done_on_the_host_is_vetoed_and_named() -> None:
    # The mutation: the group still says it is on the unit, and its epilogue runs in a host loop.
    report = HA.audit(_module(_loop("requant", _ELEMENTS, _REQUANT)), [_site("requant")])
    (row,) = report["groups"]
    assert row["verdict"] == HA.HOST_COMPUTE
    assert row["value_arithmetic_per_element"] == pytest.approx(3.0)  # sitofp fmul fptosi
    assert row["payload_ratio"] == pytest.approx(5.0)  # 4 bytes in, 1 out
    assert "sitofp" in row["dominant_block"]["signature"] and "unit0" in row["why"]
    assert report["vetoed"] and not report["proven_clean"]
    with pytest.raises(HA.HostComputeVeto, match="compute on the host"):
        HA.require_clean(report)


def test_the_same_code_in_a_declared_host_group_is_reported_not_judged() -> None:
    report = HA.audit(_module(_loop("requant", _ELEMENTS, _REQUANT)), [_site("requant", placement=HA.HOST)])
    (row,) = report["groups"]
    assert row["verdict"] == HA.DECLARED_HOST and row["host_value_arithmetic"] == 3 * _ELEMENTS
    assert not report["vetoed"] and report["accelerator_groups"] == 0


def test_work_that_cannot_be_counted_is_unknown_never_clean() -> None:
    dynamic = '%n = "llvm.add"(%dyn, %c0) : (i64, i64) -> i64'
    report = HA.audit(_module(_loop("opaque", 0, _REQUANT, bound=dynamic)), [_site("opaque")])
    (row,) = report["groups"]
    assert row["verdict"] == HA.UNKNOWN and "could not be counted" in row["why"]
    assert (report["vetoed"], report["proven_clean"]) == (False, False)
    missing = HA.audit(_module(_loop("issue", 4, _ISSUE)), [_site("absent")])
    assert missing["groups"][0]["verdict"] == HA.NOT_EMITTED and not missing["proven_clean"]


def test_the_budget_is_data_and_a_tighter_one_changes_the_verdict() -> None:
    module = _module(_loop("issue", 512, _ISSUE))  # a command every two elements
    assert HA.audit(module, [_site("issue")])["groups"][0]["verdict"] == HA.HOST_COMPUTE
    loose = HA.Budget(arithmetic_per_element=2.0)
    assert HA.audit(module, [_site("issue")], loose)["groups"][0]["verdict"] == HA.CLEAN


def test_sites_come_from_the_outliners_dispatch_table() -> None:
    from merlin.xdsl_dialects.lowering.outline import DispatchInfo

    rows = [
        DispatchInfo(
            index=0,
            symbol="forward$kernel_0__rconv_1",
            root_op="linalg.generic",
            n_operands=3,
            result_types=["tensor<1x64x56x56xi8>"],
            group=7,
            placement="unit0",
            stages=["contraction", "relu"],
        ),
        DispatchInfo(
            index=1, symbol="forward$kernel_1", root_op="linalg.generic", n_operands=1, result_types=["tensor<4xf32>"]
        ),
    ]
    (site,) = HA.sites_from_outline(rows)
    assert (site.group, site.placement, site.elements, site.element_bytes) == (7, "unit0", 64 * 56 * 56, 1)


def test_an_emitted_program_is_audited_from_its_own_buffer_and_artifact() -> None:
    from merlin.targetgen import offload_census as OC

    buffer = {
        "tensors": {"acc": {"shape": [32, 32], "dtype": "i32"}, "y": {"shape": [32, 32], "dtype": "i8"}},
        "kernel_abi": {"kind": "whole_program", "args": [], "outputs": ["y"]},
        "commands": [{"opcode": "COMMIT", "operands": {"src": "acc", "dst": "y"}}],
    }
    text = '"builtin.module"() ({' + _loop("entry", _ELEMENTS, _REQUANT) + "\n}) : () -> ()"
    vetoed = OC.host_compute_row("p", buffer, text, "offloaded")
    assert vetoed["verdict"] == HA.HOST_COMPUTE and vetoed["elements"] == _ELEMENTS
    # The same program, having SAID it runs on the host, is reported and not judged.
    assert OC.host_compute_row("p", buffer, text, "host_only")["verdict"] == HA.DECLARED_HOST
    # A device program is not host code; it is unknown to this audit, never clean.
    assert OC.host_compute_row("p", buffer, ".word 0x0000007b", "offloaded")["verdict"] == HA.UNKNOWN


def _graded_run(tmp_path, name: str, artifact: str, buffer: dict):
    import json

    generated = tmp_path / name / "generated"
    generated.mkdir(parents=True)
    (generated / "command_buffer.json").write_text(json.dumps(buffer), encoding="utf-8")
    (generated / "lowered.llvm.mlir").write_text(artifact, encoding="utf-8")
    # What the runner leaves beside them: the file every later reader takes the verdict from.
    status = "fail" if name == "already_failing" else "pass"
    (tmp_path / name / "capsule_result.json").write_text(
        json.dumps({"capsule": name, "status": status}), encoding="utf-8"
    )


def test_a_pass_that_does_the_units_work_on_the_host_fails_the_grade(tmp_path, monkeypatch) -> None:
    from merlin.targetgen import offload_census as OC

    buffer = {
        "tensors": {"acc": {"shape": [32, 32], "dtype": "i32"}, "y": {"shape": [32, 32], "dtype": "i8"}},
        "kernel_abi": {"kind": "whole_program", "args": [], "outputs": ["y"]},
        "commands": [{"opcode": "COMMIT", "operands": {"src": "acc", "dst": "y"}}],
    }
    host_loop = '"builtin.module"() ({' + _loop("entry", _ELEMENTS, _REQUANT) + "\n}) : () -> ()"
    for name in ("does_it_on_the_host", "already_failing", "whole_model"):
        _graded_run(tmp_path, name, host_loop, buffer)
    results = [
        {"capsule": "does_it_on_the_host", "kind": "layer", "status": "pass"},
        {
            "capsule": "already_failing",
            "kind": "layer",
            "status": "fail",
            "failure": {"category": "functional_mismatch"},
        },
        {"capsule": "whole_model", "kind": "model", "status": "pass"},
        {"capsule": "left_no_artifacts", "kind": "layer", "status": "pass"},
    ]
    vetoed = OC.apply_host_compute_veto(results, tmp_path)
    assert [row["program"] for row in vetoed] == ["does_it_on_the_host"]
    first, failing, model, bare = results
    assert (first["status"], first["failure"]["category"]) == ("fail", OC.VETO_CATEGORY)
    # The run's own result file carries it, because the row an agent reads is rebuilt from there.
    import json

    stored = json.loads((tmp_path / "does_it_on_the_host" / "capsule_result.json").read_text())
    assert (stored["status"], stored["status_before_host_compute_veto"]) == ("fail", "pass")
    assert stored["failure"]["plane"] == "host_compute" and stored["host_compute"]["verdict"]
    assert "reported, not failed" in stored["failure"]["detail"][:800]  # the tail survives the verdict's cap
    untouched = json.loads((tmp_path / "already_failing" / "capsule_result.json").read_text())
    assert "status_before_host_compute_veto" not in untouched
    # It can fail a pass and nothing else: an existing failure keeps its own reason, a whole model
    # owns host regions legitimately, and a run with no artifacts is left as the grade found it.
    assert failing["failure"] == {"category": "functional_mismatch"}
    assert model["status"] == "pass" and bare["status"] == "pass"

    # The switch reports without failing, and says that it did.
    monkeypatch.setenv(OC.VETO_ENV, "0")
    again = [{"capsule": "does_it_on_the_host", "kind": "layer", "status": "pass"}]
    assert OC.apply_host_compute_veto(again, tmp_path) and again[0]["status"] == "pass"
    assert again[0]["host_compute"]["veto_suppressed_by"] == OC.VETO_ENV


def test_the_grade_is_where_the_veto_is_applied() -> None:
    # Held by source: promotion and grading paths swallow exceptions by design, so an unwired veto
    # would look exactly like a clean corpus.
    from merlin.common.paths import merlin_dir, module_source_path

    text = module_source_path("merlin.targetgen.capsule_grade").read_text(encoding="utf-8")
    assert "apply_host_compute_veto(results, rr)" in text and 'score["host_compute_vetoed"]' in text


def test_a_finding_quotes_an_unrolled_loop_in_a_line_not_a_page() -> None:
    from merlin.verify.host_compute_audit import compact_signature

    unrolled = "ptrtoint ptrtoint mlir.constant " + "getelementptr store " * 400 + "br"
    assert compact_signature(unrolled) == "(ptrtoint) x2 mlir.constant (getelementptr store) x400 br"
    assert compact_signature("load add store") == "load add store" and compact_signature(None) == ""
    assert len(compact_signature(" ".join(f"op{i}" for i in range(200)))) <= 160
