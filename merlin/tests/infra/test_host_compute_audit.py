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


def test_unknown_memory_payload_is_not_zero_bytes_or_clean() -> None:
    # A pointer load has no derived byte width in this accounting instrument.
    # Known trip counts must not turn its unknown payload into zero traffic.
    body = '    %p = "llvm.load"(%in) : (!llvm.ptr) -> !llvm.ptr'
    report = HA.audit(_module(_loop("pointer_load", 4, body)), [_site("pointer_load")])
    assert report["groups"][0]["verdict"] == HA.UNKNOWN
    assert not report["proven_clean"]


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


@pytest.mark.parametrize("loaded_offset", [False, True])
def test_pointer_setup_is_not_payload_compute_but_loaded_offsets_are(loaded_offset: bool) -> None:
    """Replay the census/grading seam for a narrow output with many device commands.

    Pointer-plus-constant setup can exceed an output-normalized arithmetic budget
    without computing even one tensor element. A loaded offset must NOT receive
    that exemption: its value is tensor data, even if later used as an address.
    """
    from merlin.targetgen import offload_census as OC

    offset = (
        '%offset = "llvm.load"(%in) : (!llvm.ptr) -> i64'
        if loaded_offset
        else '%offset = "llvm.mlir.constant"() <{value = 16 : i64}> : () -> i64'
    )
    setup = [
        '%base = "llvm.ptrtoint"(%in) : (!llvm.ptr) -> i64',
        offset,
    ]
    for i in range(8):
        setup.extend(
            (
                f'%addr{i} = "llvm.add"(%base, %offset) : (i64, i64) -> i64',
                f'"llvm.inline_asm"(%addr{i}) <{{asm_string = "", constraints = "r", has_side_effects}}> : (i64) -> ()',
            )
        )
    text = (
        '"builtin.module"() ({ "llvm.func"() <{sym_name = "entry", '
        "function_type = !llvm.func<void (!llvm.ptr)>}> ({ ^entry(%in: !llvm.ptr):\n"
        + "\n".join(setup)
        + '\n "llvm.return"() : () -> () }) : () -> () }) : () -> ()'
    )
    buffer = {
        "tensors": {"y": {"shape": [2], "dtype": "i8"}},
        "kernel_abi": {"kind": "whole_program", "args": [], "outputs": ["y"]},
        "commands": [{"opcode": "COMMIT", "operands": {"dst": "y"}}],
    }
    row = OC.host_compute_row("neutral", buffer, text, "offloaded")
    assert row["verdict"] == (HA.HOST_COMPUTE if loaded_offset else HA.CLEAN), row
    assert row["arithmetic_per_element"] == (4.0 if loaded_offset else 0.0)
    assert row["host_total_arithmetic"] == 8
    assert row["host_address_arithmetic"] == (0 if loaded_offset else 8)
    assert row["arithmetic_basis"] == "pointer_offset_provenance_v1"


@pytest.mark.parametrize("sink", ["return", "store"])
def test_pointer_derived_scalar_values_do_not_get_an_address_exemption(sink: str) -> None:
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.parser import Parser

    end = "llvm.return %value : i64" if sink == "return" else "llvm.store %value, %in : i64, !llvm.ptr\n llvm.return"
    result_type = " -> i64" if sink == "return" else ""
    text = f"""module {{
      llvm.func @entry(%in: !llvm.ptr){result_type} {{
        %base = llvm.ptrtoint %in : !llvm.ptr to i64
        %one = llvm.mlir.constant(1 : i64) : i64
        %next = llvm.add %base, %one : i64
        %value = llvm.add %next, %one : i64
        {end}
      }}
    }}"""
    ctx = Context()
    for dialect in (builtin.Builtin, llvm.LLVM):
        ctx.load_dialect(dialect)
    module = Parser(ctx, text).parse_module()
    module.verify()
    (row,) = HA.audit(module, [_site("entry", elements=1)])["groups"]
    assert row["verdict"] == HA.HOST_COMPUTE
    assert row["host_address_arithmetic"] == 0
    assert row["arithmetic_per_element"] == 2.0


def test_pointer_only_target_commands_are_orchestration_but_unknown_asm_is_not(monkeypatch) -> None:
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.parser import Parser

    from merlin.targetgen.rocc import decode

    monkeypatch.setattr(decode, "isa_constants", lambda target: {"CUSTOM_OPCODE": 123, "FUNCT_CLASS": {2: "MVIN"}})

    text = """module {
      llvm.func @entry(%input: !llvm.ptr, %output: !llvm.ptr) {
        %base = llvm.ptrtoint %input : !llvm.ptr to i64
        %one = llvm.mlir.constant(1 : i64) : i64
        %first = llvm.udiv %one, %one : i64
        %remainder = llvm.urem %one, %one : i64
        %offset = llvm.add %first, %remainder : i64
        %second = llvm.add %base, %offset : i64
        "llvm.inline_asm"(%second, %one) <{asm_string = ".insn r 0x7b, 0x3, 0x2, x0, $0, $1", constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()
        llvm.return
      }
    }"""

    def judge(source: str):
        ctx = Context()
        for dialect in (builtin.Builtin, llvm.LLVM):
            ctx.load_dialect(dialect)
        module = Parser(ctx, source).parse_module()
        module.verify()
        return HA.audit(module, [_site("entry", elements=1)], target="fixture")["groups"][0]

    accepted = judge(text)
    assert accepted["verdict"] == HA.CLEAN
    assert accepted["arithmetic_basis"] == "pointer_only_target_commands_v1"
    refused = judge(text.replace(".insn r 0x7b", ".insn r 0x0b"))
    assert refused["verdict"] == HA.HOST_COMPUTE
    # A command with literal x0 sources (a flush) takes no operands and is still orchestration.
    flush = '"llvm.inline_asm"() <{asm_string = ".insn r 0x7b, 0x3, 0x2, x0, x0, x0", constraints = "~{memory}", has_side_effects}> : () -> ()\n        llvm.return'
    assert judge(text.replace("llvm.return", flush, 1))["verdict"] == HA.CLEAN
    # A source register that is neither x0 nor one of the asm's operands is refused.
    stray = flush.replace("x0, x0, x0", "x0, x5, x0")
    assert judge(text.replace("llvm.return", stray, 1))["verdict"] == HA.HOST_COMPUTE
    # An unused stack slot is not host computation; a store into it still is.
    scratch = "%count = llvm.mlir.constant(4 : i32) : i32\n        %slot = llvm.alloca %count x i32 : (i32) -> !llvm.ptr\n        "
    with_slot = text.replace("%base = llvm.ptrtoint", scratch + "%base = llvm.ptrtoint")
    assert judge(with_slot)["verdict"] == HA.CLEAN
    stored = with_slot.replace("llvm.return", "llvm.store %count, %slot : i32, !llvm.ptr\n        llvm.return")
    assert judge(stored)["verdict"] == HA.HOST_COMPUTE


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


def test_the_veto_is_evaluated_by_a_fresh_child_with_identical_results(tmp_path, monkeypatch) -> None:
    """A long-lived grader must apply the audit on disk, not the one it imported at launch."""
    import json

    from merlin.targetgen import host_compute_veto_process
    from merlin.targetgen import offload_census as OC

    buffer = {
        "tensors": {"acc": {"shape": [32, 32], "dtype": "i32"}, "y": {"shape": [32, 32], "dtype": "i8"}},
        "kernel_abi": {"kind": "whole_program", "args": [], "outputs": ["y"]},
        "commands": [{"opcode": "COMMIT", "operands": {"src": "acc", "dst": "y"}}],
    }
    host_loop = '"builtin.module"() ({' + _loop("entry", _ELEMENTS, _REQUANT) + "\n}) : () -> ()"
    graded = {}
    for mode in ("child", "here"):
        _graded_run(tmp_path / mode, "on_host", host_loop, buffer)
        rows = [{"capsule": "on_host", "kind": "layer", "status": "pass"}]
        monkeypatch.setenv(host_compute_veto_process.ENV, "1" if mode == "child" else "0")
        vetoed = OC.apply_host_compute_veto(rows, tmp_path / mode)
        stored = json.loads((tmp_path / mode / "on_host" / "capsule_result.json").read_text())
        graded[mode] = (vetoed, rows, stored)
    assert graded["child"] == graded["here"] and graded["child"][1][0]["status"] == "fail"

    # The audit this process holds does not decide: a stale (here, stubbed) module in the grader
    # changes nothing, because the child imports the module again.
    def stale(site, function, budget=HA.Budget(), **kwargs):
        return {"group": site.group, "placement": site.placement, "symbol": site.symbol, "verdict": HA.CLEAN}

    monkeypatch.setattr(HA, "audit_group", stale)
    for mode, expected in (("child", "fail"), ("here", "pass")):
        _graded_run(tmp_path / ("stale_" + mode), "on_host", host_loop, buffer)
        rows = [{"capsule": "on_host", "kind": "layer", "status": "pass"}]
        monkeypatch.setenv(host_compute_veto_process.ENV, "1" if mode == "child" else "0")
        OC.apply_host_compute_veto(rows, tmp_path / ("stale_" + mode))
        assert rows[0]["status"] == expected, mode


def test_the_grade_is_where_the_veto_is_applied() -> None:
    # Held by source: promotion and grading paths swallow exceptions by design, so an unwired veto
    # would look exactly like a clean corpus.
    from merlin.common.paths import module_source_path

    text = module_source_path("merlin.targetgen.capsule_grade").read_text(encoding="utf-8")
    assert "apply_host_compute_veto(results, rr, capsules=caps, target=target)" in text
    assert 'score["host_compute_vetoed"]' in text


def test_a_finding_quotes_an_unrolled_loop_in_a_line_not_a_page() -> None:
    from merlin.verify.host_compute_audit import compact_signature

    unrolled = "ptrtoint ptrtoint mlir.constant " + "getelementptr store " * 400 + "br"
    assert compact_signature(unrolled) == "(ptrtoint) x2 mlir.constant (getelementptr store) x400 br"
    assert compact_signature("load add store") == "load add store" and compact_signature(None) == ""
    assert len(compact_signature(" ".join(f"op{i}" for i in range(200)))) <= 160


def _tile_remainder_kernel(rows: int, tile: int, *, store: bool) -> str:
    """Per-row transfer commands under a ``min(tile, rows - i)`` row loop, as a tiled im2col emits."""
    write = (
        """
        %dst = llvm.getelementptr %output[%row] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %byte = llvm.trunc %row : i64 to i8
        llvm.store %byte, %dst : i8, !llvm.ptr"""
        if store
        else ""
    )
    return f"""module {{
      llvm.func @entry(%input: !llvm.ptr, %output: !llvm.ptr) {{
        %zero = llvm.mlir.constant(0 : i64) : i64
        %one = llvm.mlir.constant(1 : i64) : i64
        %rows = llvm.mlir.constant({rows} : i64) : i64
        %tile = llvm.mlir.constant({tile} : i64) : i64
        %width = llvm.mlir.constant(112 : i64) : i64
        llvm.br ^outer(%zero : i64)
      ^outer(%i: i64):
        %more = llvm.icmp "ult" %i, %rows : i64
        llvm.cond_br %more, ^tile, ^done
      ^tile:
        %left = llvm.sub %rows, %i : i64
        %short = llvm.icmp "ult" %left, %tile : i64
        %count = llvm.select %short, %left, %tile : i1, i64
        llvm.br ^inner(%zero : i64)
      ^inner(%j: i64):
        %again = llvm.icmp "ult" %j, %count : i64
        llvm.cond_br %again, ^row, ^latch
      ^row:
        %row = llvm.add %i, %j : i64
        %x = llvm.urem %row, %width : i64
        %y = llvm.udiv %row, %width : i64
        %offset = llvm.mul %y, %x : i64
        %src = llvm.getelementptr %input[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %addr = llvm.ptrtoint %src : !llvm.ptr to i64{write}
        "llvm.inline_asm"(%addr, %j) <{{asm_string = ".insn r 0x7b, 0x3, 0x2, x0, $0, $1", constraints = "r,r,~{{memory}}", has_side_effects}}> : (i64, i64) -> ()
        %next = llvm.add %j, %one : i64
        llvm.br ^inner(%next : i64)
      ^latch:
        %following = llvm.add %i, %tile : i64
        llvm.br ^outer(%following : i64)
      ^done:
        llvm.return
      }}
    }}"""


@pytest.mark.parametrize("store", [False, True])
def test_remainder_row_loops_past_the_stepping_budget_are_counted_and_judged(monkeypatch, store) -> None:
    """A row loop bounded by ``min(tile, rows - i)`` is decided, not UNKNOWN, at whole-model scale.

    Stepped one block at a time this needs about 13M block visits, past the exact trace's budget,
    which used to leave a legitimate tiled kernel UNKNOWN. Counting it must decide both ways:
    pointer-only command issue is clean; the same loop storing on the host is vetoed.
    """
    from merlin.targetgen.rocc import decode

    monkeypatch.setattr(decode, "isa_constants", lambda target: {"CUSTOM_OPCODE": 123, "FUNCT_CLASS": {2: "MVIN"}})
    rows, tile = 64 * 100_000, 100_000
    ctx = Context()
    for dialect in (builtin.Builtin, llvm.LLVM):
        ctx.load_dialect(dialect)
    module = Parser(ctx, _tile_remainder_kernel(rows, tile, store=store)).parse_module()
    module.verify()
    (row,) = HA.audit(module, [_site("entry", elements=rows)], target="fixture")["groups"]
    assert row["activity_status"] == "derived"
    if store:
        assert row["verdict"] == HA.HOST_COMPUTE
    else:
        assert row["verdict"] == HA.CLEAN
        assert row["arithmetic_basis"] == "pointer_only_target_commands_v1"
        assert row["host_command_orchestration_arithmetic"] > rows


@pytest.mark.parametrize("result_type,clean", [("!llvm.void", True), ("i64", False)])
def test_imported_unused_void_command_result_is_not_a_host_value(monkeypatch, result_type, clean):
    from merlin.targetgen.rocc import decode

    monkeypatch.setattr(decode, "isa_constants", lambda target: {"CUSTOM_OPCODE": 123, "FUNCT_CLASS": {2: "MVIN"}})
    text = _tile_remainder_kernel(97, 16, store=False)
    text = text.replace('"llvm.inline_asm"(%addr', '%observed = "llvm.inline_asm"(%addr')
    text = text.replace(': (i64, i64) -> ()', f': (i64, i64) -> {result_type}')
    ctx = Context()
    for dialect in (builtin.Builtin, llvm.LLVM):
        ctx.load_dialect(dialect)
    module = Parser(ctx, text).parse_module()
    (row,) = HA.audit(module, [_site("entry", elements=97)], target="fixture")["groups"]
    assert row["verdict"] == (HA.CLEAN if clean else HA.HOST_COMPUTE), row


def test_command_orchestration_exemption_refuses_unmodeled_effects(monkeypatch):
    from merlin.targetgen.rocc import decode

    monkeypatch.setattr(decode, "isa_constants", lambda target: {"CUSTOM_OPCODE": 123, "FUNCT_CLASS": {2: "MVIN"}})
    text = _tile_remainder_kernel(97, 16, store=False).replace(
        '%addr = llvm.ptrtoint %src',
        '"llvm.intr.memcpy"(%input, %output, %one) : (!llvm.ptr, !llvm.ptr, i64) -> ()\n'
        ' %addr = llvm.ptrtoint %src',
    )
    ctx = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, llvm.LLVM):
        ctx.load_dialect(dialect)
    module = Parser(ctx, text).parse_module()
    (row,) = HA.audit(module, [_site("entry", elements=97)], target="fixture")["groups"]
    assert row["verdict"] == HA.HOST_COMPUTE, row
