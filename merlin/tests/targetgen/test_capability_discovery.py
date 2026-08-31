"""Tests for :mod:`merlin.targetgen.capability_discovery`.

The module's whole value is that a capability is only reported when something evidences it, so the tests
are mostly about the three states and about the parser being structural rather than pattern-shaped: a
reader that only recognizes one spelling of a define silently drops the others, which is the failure mode
this repo has paid for repeatedly in its trace decoder.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import capability_discovery as CD

# --- a synthetic header. Deliberately NOT any shipped target's file: it exercises the parser's shapes
# (continuations, both comment forms, a bit-layout comment, a macro-written enum, marker defines) without
# tying the test to one vendor's spelling.
SYNTHETIC = """
// See LICENSE
#ifndef SYNTH_H
#define SYNTH_H
#include "synth_params.h"
#include <stdint.h>

#define K_CONFIG 0
#define K_LOAD 1
#define K_STORE 2

#define NO_ACTIVATION 0
#define RELU 1
#define LAYERNORM 2
#define IGELU 3
#define SOFTMAX 4

// RS1: [63:32] acc_scale | [31:16] a_stride | [9] b_transpose | [8] a_transpose | [4:3] activation
#define synth_config_ex(dataflow, sys_act, sys_shift, A_transpose, B_transpose) \\
    INSTR(((uint64_t)(B_transpose) << 9) | \\
          ((uint64_t)(A_transpose) << 8) | \\
          ((uint64_t)(sys_act) << 3), K_CONFIG)

#define synth_config_st(stride, acc_act, acc_scale, pool_stride, pool_size, upad, lpad) \\
    INSTR(stride, K_STORE)

#define STRING_WITH_SLASHES "a // not a comment /* either */"
#define HAS_SCALE
#endif
"""

PARAMS = """
#define DIM 16
typedef int8_t elem_t;
typedef int32_t acc_t;
typedef int64_t full_t;
typedef float acc_scale_t;
typedef uint32_t acc_scale_t_bits;
#define ACC_SCALE_T_IS_FLOAT
#define ACC_SCALE_EXP_BITS 8
#define ACC_SCALE_SIG_BITS 24
#define ROUND_NEAR_EVEN(x) ((x) + 1)
"""


@pytest.fixture()
def header(tmp_path):
    (tmp_path / "synth_params.h").write_text(PARAMS, encoding="utf-8")
    p = tmp_path / "synth.h"
    p.write_text(SYNTHETIC, encoding="utf-8")
    return p


# --------------------------------------------------------------------------------------------------
# the structural reader
# --------------------------------------------------------------------------------------------------


def test_parses_object_and_function_macros(header):
    hm = CD.parse_c_header(header)
    names = {m.name for m in hm.macros}
    assert {"K_CONFIG", "NO_ACTIVATION", "SOFTMAX", "HAS_SCALE"} <= names
    cfg = hm.macro("synth_config_ex")
    assert cfg is not None and cfg.is_function
    # every parameter survives the line continuations
    assert cfg.params == ("dataflow", "sys_act", "sys_shift", "A_transpose", "B_transpose")


def test_line_numbers_are_recorded_for_evidence(header):
    hm = CD.parse_c_header(header)
    m = hm.macro("SOFTMAX")
    assert m is not None and m.int_value == 4
    assert SYNTHETIC.splitlines()[m.line - 1].strip() == "#define SOFTMAX 4"


def test_comment_forms_and_string_literals_do_not_confuse_the_scanner(header):
    hm = CD.parse_c_header(header)
    m = hm.macro("STRING_WITH_SLASHES")
    # the `//` and `/* */` INSIDE the string literal must not have been eaten as comments
    assert m is not None and m.body.endswith('either */"')


def test_bit_layout_comment_is_parsed_structurally(header):
    hm = CD.parse_c_header(header)
    fields = {b.name: b for b in hm.bitfields}
    assert set(fields) == {"acc_scale", "a_stride", "b_transpose", "a_transpose", "activation"}
    assert (fields["a_transpose"].hi, fields["a_transpose"].lo) == (8, 8)
    assert (fields["activation"].hi, fields["activation"].lo) == (4, 3)
    assert fields["acc_scale"].register == "RS1"


def test_local_includes_are_recorded(header):
    hm = CD.parse_c_header(header)
    # angle-bracket system includes are NOT local sources and must not be followed
    assert hm.includes == ("synth_params.h",)


def test_typedefs_are_recorded(tmp_path):
    p = tmp_path / "p.h"
    p.write_text(PARAMS, encoding="utf-8")
    hm = CD.parse_c_header(p)
    assert ("elem_t", "int8_t") in {(a, u) for a, u, _l in hm.typedefs}


def test_no_regex_import_in_the_module():
    src = (repo_root() / "merlin" / "python" / "merlin" / "targetgen"
           / "capability_discovery.py").read_text(encoding="utf-8")
    assert "import re\n" not in src and "import re " not in src


# --------------------------------------------------------------------------------------------------
# derivation
# --------------------------------------------------------------------------------------------------


def test_activation_modes_are_derived_with_encodings_and_families(header):
    hm = CD.parse_c_header(header)
    out: list[CD.Finding] = []
    CD._activation_modes([hm], out)
    modes = {f.name: f for f in out if f.axis == "activation_mode"}
    assert {f.name: f.value for f in modes.values()} == {
        "NO_ACTIVATION": 0, "RELU": 1, "LAYERNORM": 2, "IGELU": 3, "SOFTMAX": 4}
    # the point of the whole module: these two are NOT elementwise activations
    assert modes["LAYERNORM"].family == "normalization"
    assert modes["SOFTMAX"].family == "softmax"
    assert modes["RELU"].family == "elementwise_map"
    # an integer-approximation spelling still resolves, through the substring rule, and says so
    assert modes["IGELU"].family == "elementwise_map"
    assert "substring" in (modes["IGELU"].family_basis or "")
    # the identity mode licenses nothing
    assert modes["NO_ACTIVATION"].family is None
    for f in modes.values():
        assert f.evidence and f.evidence[0].line is not None


def test_enum_grouping_does_not_need_a_known_name(header):
    hm = CD.parse_c_header(header)
    groups = CD._enum_groups(hm.macros)
    got = [[m.name for m in g] for g in groups]
    assert ["K_CONFIG", "K_LOAD", "K_STORE"] in got
    assert ["NO_ACTIVATION", "RELU", "LAYERNORM", "IGELU", "SOFTMAX"] in got


def test_feature_axes_report_parameters_and_which_operand(header):
    hm = CD.parse_c_header(header)
    out: list[CD.Finding] = []
    CD._feature_axes([hm], out)
    by_axis = {f.axis: f for f in out if f.axis != "family"}
    assert by_axis["transpose"].state == CD.PRESENT
    assert "A_transpose" in by_axis["transpose"].value["parameters"]
    # the operand each transpose applies to is derivable from the identifier's remaining tokens
    assert {"a", "b"} <= set(by_axis["transpose"].value["qualifiers"])
    assert by_axis["pooling"].state == CD.PRESENT
    assert {"pool_size", "pool_stride"} <= set(by_axis["pooling"].value["parameters"])
    assert by_axis["padding"].state == CD.PRESENT
    # this header has none of these, and the header WAS read -> absent, with the evidence of silence
    assert by_axis["residual_add"].state == CD.ABSENT
    assert by_axis["residual_add"].evidence


def test_a_valued_macro_is_a_datum_not_a_feature_marker(tmp_path):
    """A performance-counter index named for a unit must not evidence that unit's feature."""
    p = tmp_path / "counters.h"
    p.write_text("#define TRANSPOSE_UNROLLER_ACTIVE_CYCLES 44\n", encoding="utf-8")
    out: list[CD.Finding] = []
    CD._feature_axes([CD.parse_c_header(p)], out)
    assert {f.axis: f.state for f in out if f.axis == "transpose"} == {"transpose": CD.ABSENT}


def test_dtype_findings_are_scoped_to_a_datapath_and_never_collapsed(tmp_path):
    p = tmp_path / "p.h"
    p.write_text(PARAMS, encoding="utf-8")
    out: list[CD.Finding] = []
    CD._dtype_axes([CD.parse_c_header(p)], out)
    roles = {}
    for f in out:
        if f.axis == "datapath_dtype":
            roles.setdefault(f.datapath, set()).add(str(f.value))
    assert roles["operand"] == {"int8_t"}
    assert roles["accumulate"] == {"int32_t"}
    assert roles["scale"] == {"float"}
    # THE claim this module exists to keep straight: fp32 is true of the scale path and false of the
    # operand path, and nothing may report a bare "supports fp32".
    assert "float" not in roles["operand"]
    # the float markers ride along as evidence on the scale finding
    scale = [f for f in out if f.axis == "datapath_dtype" and f.datapath == "scale"][0]
    observed = " ".join(e.observed for e in scale.evidence)
    assert "ACC_SCALE_EXP_BITS 8" in observed and "ACC_SCALE_SIG_BITS 24" in observed


def test_bit_pun_aliases_are_not_reported_as_datapaths(tmp_path):
    p = tmp_path / "p.h"
    p.write_text(PARAMS, encoding="utf-8")
    out: list[CD.Finding] = []
    CD._dtype_axes([CD.parse_c_header(p)], out)
    assert not [f for f in out if "acc_scale_t_bits" in str(f.name)]


def test_scale_rounding_is_reported_from_the_headers_own_macro(tmp_path):
    p = tmp_path / "p.h"
    p.write_text(PARAMS, encoding="utf-8")
    out: list[CD.Finding] = []
    CD._dtype_axes([CD.parse_c_header(p)], out)
    r = [f for f in out if f.axis == "scale_rounding"][0]
    assert r.state == CD.PRESENT
    assert "ROUND_NEAR_EVEN" in r.value["rounding_macros"]


def test_axes_fold_across_sources_rather_than_contradicting_each_other(header, tmp_path):
    """A parameter header silent about pooling must not make pooling ABSENT for the target."""
    params = CD.parse_c_header(tmp_path / "synth_params.h")
    main = CD.parse_c_header(header)
    out: list[CD.Finding] = []
    CD._feature_axes([main, params], out)
    states = {f.state for f in out if f.axis == "pooling"}
    assert states == {CD.PRESENT}


# --------------------------------------------------------------------------------------------------
# three states, and the refusal to diff what could not be read
# --------------------------------------------------------------------------------------------------


def test_absent_and_undeterminable_are_never_collapsed(monkeypatch):
    """No readable ISA source is UNDETERMINABLE for every header-grounded axis, not ABSENT."""
    monkeypatch.setattr(CD, "isa_sources", lambda *a, **k: [])
    monkeypatch.setattr(CD, "_facts_if_present", lambda t: (None, ""))
    monkeypatch.setattr(CD, "_pins_for", lambda t: [])
    surf = CD.discover("unused_name_resolved_at_runtime")
    assert surf.findings == []
    assert surf.rungs_ran == ()
    assert "activation_mode" in surf.undeterminable_axes()
    assert any("UNDETERMINABLE" in n or "undeterminable" in n for n in surf.notes)


def test_a_family_whose_deciding_rung_never_ran_is_undeterminable_not_over_declared(monkeypatch):
    """The direction that deletes a real capability from a manifest, guarded explicitly."""
    dec = CD.CapabilitySurface(target="t", origin="declared")
    dec.findings.append(CD.Finding(axis="family", name="softmax", state=CD.PRESENT, family="softmax",
                                   evidence=(CD.Evidence(rung="contract", locator="c", observed="x"),)))
    disc = CD.CapabilitySurface(target="t", origin="discovered", rungs_ran=("rtl_facts",))
    monkeypatch.setattr(CD, "discover", lambda t, **k: disc)
    monkeypatch.setattr(CD, "declared", lambda t: dec)
    d = CD.delta("t")
    assert d["over_declared"] == []
    assert any(u["kind"] == "family" and u["name"] == "softmax" for u in d["undeterminable"])


def test_a_family_whose_deciding_rung_did_run_is_over_declared(monkeypatch):
    dec = CD.CapabilitySurface(target="t", origin="declared")
    dec.findings.append(CD.Finding(axis="family", name="softmax", state=CD.PRESENT, family="softmax",
                                   evidence=(CD.Evidence(rung="contract", locator="c", observed="x"),)))
    disc = CD.CapabilitySurface(target="t", origin="discovered",
                                rungs_ran=("rtl_facts", "isa_header"))
    monkeypatch.setattr(CD, "discover", lambda t, **k: disc)
    monkeypatch.setattr(CD, "declared", lambda t: dec)
    d = CD.delta("t")
    assert [o["name"] for o in d["over_declared"]] == ["softmax"]


def test_delta_refuses_to_diff_an_unreadable_declaration(monkeypatch):
    dec = CD.CapabilitySurface(target="t", origin="declared", resolved=False)
    dec.notes.append("no target contract resolved")
    disc = CD.CapabilitySurface(target="t", origin="discovered", rungs_ran=("rtl_facts",))
    disc.findings.append(CD.Finding(axis="family", name="contraction", state=CD.PRESENT,
                                    family="contraction",
                                    evidence=(CD.Evidence(rung="rtl_facts", locator="f",
                                                          observed="mesh"),)))
    monkeypatch.setattr(CD, "discover", lambda t, **k: disc)
    monkeypatch.setattr(CD, "declared", lambda t: dec)
    d = CD.delta("t")
    assert d["status"] == "no_readable_declaration"
    assert d["under_declared"] == [] and d["over_declared"] == []


def test_discover_refuses_when_a_pin_does_not_verify(monkeypatch):
    from merlin.common import provenance as P
    bad = P.Verification(pin="p", observed=P.Observation(path="/nowhere", present=False),
                         drift=("commit: declared abc, observed def",))
    monkeypatch.setattr(CD, "_pins_for", lambda t: ["p"])
    monkeypatch.setattr(P, "verify", lambda name, **k: bad)
    with pytest.raises(CD.ProvenanceRefused):
        CD.discover("unused_name_resolved_at_runtime")
    # ...and records rather than raises when the caller explicitly opts out
    monkeypatch.setattr(CD, "isa_sources", lambda *a, **k: [])
    monkeypatch.setattr(CD, "_facts_if_present", lambda t: (None, ""))
    surf = CD.discover("unused_name_resolved_at_runtime", require_pin=False)
    assert any("WITHOUT PIN VERIFICATION" in n for n in surf.notes)


def test_op_class_family_is_never_guessed_from_an_rtl_module_name(monkeypatch):
    """A funct name the contract does not classify yields no family, on purpose."""
    facts = {"facts": {"interfaces": [{"name": "funct_decode_table",
                                       "names": {"3": "SOME_VENDOR_MATMUL_CMD"}}]}}
    monkeypatch.setattr(CD, "_facts_if_present", lambda t: (facts, "test"))
    monkeypatch.setattr(CD, "_target_contract", lambda t: ({}, None))
    out: list[CD.Finding] = []
    assert CD._from_facts("t", out, [])
    f = [x for x in out if x.axis == "op_class"][0]
    assert f.state == CD.PRESENT and f.family is None
    assert "UNDETERMINABLE" in f.detail


# --------------------------------------------------------------------------------------------------
# against the real tree
# --------------------------------------------------------------------------------------------------


def test_target_discovery_types_no_name():
    """The target set is discovered from the target homes; the test asserts only its shape."""
    ts = CD.targets_with_facts()
    assert isinstance(ts, list) and ts == sorted(ts)


@pytest.mark.parametrize("target", CD.targets_with_facts())
def test_every_target_with_facts_yields_a_surface_without_raising(target):
    d = CD.delta(target, require_pin=False)
    assert d["target"] == target
    # both directions are lists, and nothing lands in two of them at once
    under = {(u["kind"], u["name"]) for u in d["under_declared"]}
    over = {(o["kind"], o["name"]) for o in d["over_declared"]}
    assert not (under & over)
    for f in d["under_declared"] + d["over_declared"]:
        assert f.get("evidence") or f.get("detail")


@pytest.mark.parametrize("target", CD.targets_with_facts())
def test_every_finding_carries_evidence(target):
    surf = CD.discover(target, require_pin=False)
    for f in surf.findings:
        assert f.evidence, f"{target}: {f.key} reported {f.state} with no evidence"


def test_a_target_whose_header_resolves_reports_the_readout_activation_modes():
    """On whichever target actually ships a readable C ISA header, the modes must come back named.

    Skipped rather than pinned to one target: the point is that the derivation works wherever a header
    resolves, not that a particular accelerator is present in this checkout.
    """
    for target in CD.targets_with_facts():
        try:
            surf = CD.discover(target, require_pin=False)
        except CD.ProvenanceRefused:
            continue
        if "isa_header" not in surf.rungs_ran:
            continue
        modes = surf.by_axis("activation_mode")
        if not [m for m in modes if m.state == CD.PRESENT]:
            continue
        for m in modes:
            assert m.evidence[0].locator and m.evidence[0].line
        return
    pytest.skip("no target in this checkout resolves a readable ISA header with activation modes")


# --------------------------------------------------------------------------------------------------
# the deciding rung: what the design was BUILT with, not what its ISA can ENCODE
# --------------------------------------------------------------------------------------------------

SYNTH_CONFIG = """
package synth

object SynthConfigs {
  val defaultConfig = SynthArrayConfig[SInt, Float](
    inputType = SInt(8.W),
    meshRows = 16,
    meshColumns = 16,
    tileRows = 1,
    sp_capacity = CapacityInKilobytes(256),
    acc_capacity = CapacityInKilobytes(64),
    has_max_pool = true,
    has_nonlinear_activations = true,
    mvin_scale_shared = false
  )
}

class DefaultSynthConfig(
  synthConfig: SynthArrayConfig = SynthConfigs.defaultConfig
) extends Config((site, here, up) => { case BuildRoCC => Nil })
"""

SYNTH_CASECLASS = """
package synth

case class SynthArrayConfig[T <: Data, U <: Data](
  inputType: T,
  meshRows: Int = 8,
  meshColumns: Int = 8,
  tileRows: Int = 1,
  sp_capacity: SynthMemCapacity = CapacityInKilobytes(64),
  acc_capacity: SynthMemCapacity = CapacityInKilobytes(16),
  has_max_pool: Boolean = true,
  has_nonlinear_activations: Boolean = true,
  has_normalizations: Boolean = false,
  mvin_scale_shared: Boolean = false
)
"""

SYNTH_HW = """
package synth

class SynthAccumulatorScale {
  val e_act = MuxCase(e, Seq(
    (has_nonlinear_activations.B && act === Activation.RELU) -> e.relu,
    (has_nonlinear_activations.B && has_normalizations.B && act === Activation.LAYERNORM) ->
      (e - io.in.bits.mean),
    (has_nonlinear_activations.B && has_normalizations.B && act === Activation.SOFTMAX) ->
      AccumulatorScale.iexp(e),
  ))
  assert(has_normalizations.B || (act =/= Activation.LAYERNORM && act =/= Activation.SOFTMAX))
}
"""


@pytest.fixture()
def scala_tree(tmp_path, monkeypatch):
    root = tmp_path / "gen"
    root.mkdir()
    (root / "Configs.scala").write_text(SYNTH_CONFIG, encoding="utf-8")
    (root / "ArrayConfig.scala").write_text(SYNTH_CASECLASS, encoding="utf-8")
    (root / "AccumulatorScale.scala").write_text(SYNTH_HW, encoding="utf-8")
    (root / "build" / "generated").mkdir(parents=True)
    (root / "build" / "generated" / "Decoy.scala").write_text(SYNTH_CASECLASS, encoding="utf-8")
    sources, _t = None, None

    def _sources(_target):
        out = []
        for fp in sorted(root.glob("*.scala")):
            code, _c = CD._split_code_and_comments(fp.read_text(encoding="utf-8"))
            out.append((fp, "\n".join(code)))
        return out, False

    monkeypatch.setattr(CD, "_config_sources", _sources)
    return root


def test_config_resolves_through_a_defaulted_parameter(scala_tree):
    facts = {"facts": {"source": {"config": "DefaultSynthConfig"}}}
    cfg = CD.elaborated_config("t", facts)
    assert cfg is not None and cfg.instantiated == "SynthArrayConfig"
    # a config commonly hands its payload over as a DEFAULTED argument; a resolver that only followed
    # explicit references stops one hop short of every field
    assert cfg.fields["meshRows"].value == 16
    assert cfg.fields["meshRows"].origin == "set"


def test_unset_fields_take_the_classes_own_declared_default(scala_tree):
    cfg = CD.elaborated_config("t", {"facts": {"source": {"config": "DefaultSynthConfig"}}})
    f = cfg.fields["has_normalizations"]
    assert f.value is False and f.origin == "declared_default"
    assert f.locator.endswith("ArrayConfig.scala") and f.line > 0


def test_capacity_units_are_read_from_the_wrappers_own_name(scala_tree):
    cfg = CD.elaborated_config("t", {"facts": {"source": {"config": "DefaultSynthConfig"}}})
    assert cfg.fields["sp_capacity"].scaled == 256 * 1024
    assert cfg.fields["acc_capacity"].scaled == 64 * 1024


def test_config_and_rtl_geometry_are_cross_checked(scala_tree):
    cfg = CD.elaborated_config("t", {"facts": {"source": {"config": "DefaultSynthConfig"}}})
    body = {"arrays": [{"name": "mesh", "rows": 16, "cols": 16}],
            "memories": [{"name": "scratchpad", "bytes": 256 * 1024},
                         {"name": "accumulator", "bytes": 64 * 1024}]}
    out: list[CD.Finding] = []
    notes: list[str] = []
    CD._corroborate_config(cfg, body, out, notes)
    got = {f.name for f in out}
    assert {"array.rows=16", "array.cols=16",
            "memory.scratchpad=262144", "memory.accumulator=65536"} <= got
    for f in out:
        assert f.evidence and f.evidence[0].line


def test_a_mode_gated_off_is_encodable_not_built_and_licenses_nothing(scala_tree, header):
    """The finding this rung exists for: an encoding with no unit behind it."""
    cfg = CD.elaborated_config("t", {"facts": {"source": {"config": "DefaultSynthConfig"}}})
    findings: list[CD.Finding] = []
    CD._activation_modes([CD.parse_c_header(header)], findings)
    assert {f.name for f in findings if f.axis == "family"} != set()
    CD._apply_build_gates("t", cfg, findings, [])
    by = {f.name: f for f in findings if f.axis == "activation_mode"}
    assert by["LAYERNORM"].state == CD.ENCODABLE_NOT_BUILT
    assert by["SOFTMAX"].state == CD.ENCODABLE_NOT_BUILT
    assert by["LAYERNORM"].gate["off"] == ["has_normalizations"]
    # ...and the family licence it issued is withdrawn with it
    fams = {f.family for f in findings if f.axis == "family"}
    assert "normalization" not in fams and "softmax" not in fams


def test_a_sibling_modes_licence_survives_its_neighbours_being_gated_off(scala_tree, header):
    """RELU is gated on a field that IS true; withdrawing LAYERNORM must not take it down too."""
    cfg = CD.elaborated_config("t", {"facts": {"source": {"config": "DefaultSynthConfig"}}})
    findings: list[CD.Finding] = []
    CD._activation_modes([CD.parse_c_header(header)], findings)
    CD._apply_build_gates("t", cfg, findings, [])
    by = {f.name: f for f in findings if f.axis == "activation_mode"}
    assert by["RELU"].state == CD.PRESENT
    assert by["RELU"].gate == {"status": "built",
                               "fields": {"has_nonlinear_activations": True},
                               "config": "DefaultSynthConfig"}
    assert "elementwise_map" in {f.family for f in findings if f.axis == "family"}


def test_the_identity_mode_is_not_gated(scala_tree, header):
    cfg = CD.elaborated_config("t", {"facts": {"source": {"config": "DefaultSynthConfig"}}})
    findings: list[CD.Finding] = []
    CD._activation_modes([CD.parse_c_header(header)], findings)
    CD._apply_build_gates("t", cfg, findings, [])
    no_act = [f for f in findings if f.name == "NO_ACTIVATION"][0]
    assert no_act.state == CD.PRESENT and no_act.gate == {"status": "identity_mode"}


def test_a_feature_gate_must_be_a_presence_switch_not_any_shared_word(scala_tree, header):
    """`mvin_scale_shared` is a sharing option; reading it as the requant enable is the mirror bug."""
    cfg = CD.elaborated_config("t", {"facts": {"source": {"config": "DefaultSynthConfig"}}})
    findings: list[CD.Finding] = []
    CD._feature_axes([CD.parse_c_header(header)], findings)
    CD._apply_build_gates("t", cfg, findings, [])
    by = {f.axis: f for f in findings if f.axis in CD._FEATURE_STEMS}
    assert by["pooling"].state == CD.PRESENT
    assert by["pooling"].gate["fields"] == {"has_max_pool": True}
    assert by["transpose"].state == CD.PRESENT
    assert by["transpose"].gate["status"] == "ungated"


def test_no_gate_is_inferred_for_a_token_the_sources_never_mention(scala_tree, tmp_path):
    cfg = CD.elaborated_config("t", {"facts": {"source": {"config": "DefaultSynthConfig"}}})
    p = tmp_path / "odd.h"
    p.write_text("#define NO_ACTIVATION 0\n#define RELU 1\n#define QUANTUMFOLD 2\n", encoding="utf-8")
    findings: list[CD.Finding] = []
    CD._activation_modes([CD.parse_c_header(p)], findings)
    CD._apply_build_gates("t", cfg, findings, [])
    odd = [f for f in findings if f.name == "QUANTUMFOLD"][0]
    assert odd.state == CD.UNDETERMINABLE
    assert odd.gate["status"] == "no_gate_found"


def test_encodable_not_built_makes_a_declared_family_over_declared(monkeypatch):
    dec = CD.CapabilitySurface(target="t", origin="declared")
    dec.findings.append(CD.Finding(axis="family", name="softmax", state=CD.PRESENT, family="softmax",
                                   evidence=(CD.Evidence(rung="contract", locator="c", observed="x"),)))
    disc = CD.CapabilitySurface(target="t", origin="discovered",
                                rungs_ran=("rtl_facts", "isa_header", "build_config"))
    disc.findings.append(CD.Finding(
        axis="activation_mode", name="SOFTMAX", state=CD.ENCODABLE_NOT_BUILT, family="softmax",
        gate={"status": "not_built", "off": ["has_normalizations"], "config": "C"},
        evidence=(CD.Evidence(rung="build_config", locator="C.scala", observed="false", line=9),)))
    monkeypatch.setattr(CD, "discover", lambda t, **k: disc)
    monkeypatch.setattr(CD, "declared", lambda t: dec)
    d = CD.delta("t")
    assert [o["name"] for o in d["over_declared"]] == ["softmax"]
    assert "NOT BUILT" in d["over_declared"][0]["detail"]
    assert d["under_declared"] == []


# --------------------------------------------------------------------------------------------------
# a pinned claim needs pinned bytes
# --------------------------------------------------------------------------------------------------


def test_pin_status_reports_a_nested_checkout_off_its_recorded_gitlink(tmp_path, monkeypatch):
    import subprocess

    def git(cwd, *args):
        subprocess.run(("git", "-C", str(cwd)) + args, check=True, capture_output=True,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})

    inner = tmp_path / "outer" / "nested"
    inner.mkdir(parents=True)
    git(tmp_path, "init", "-q", "outer")
    git(inner.parent, "init", "-q", "nested")
    (inner / "h.h").write_text("#define A 1\n", encoding="utf-8")
    git(inner, "add", "h.h")
    git(inner, "commit", "-qm", "one")
    first = subprocess.run(("git", "-C", str(inner), "rev-parse", "HEAD"),
                           capture_output=True, text=True).stdout.strip()
    git(tmp_path / "outer", "add", "nested")
    git(tmp_path / "outer", "commit", "-qm", "record gitlink")
    # the nested tree then moves on, exactly like a submodule left on a newer revision
    (inner / "h.h").write_text("#define A 2\n", encoding="utf-8")
    git(inner, "commit", "-qam", "two")

    from merlin.common import provenance as P
    monkeypatch.setattr(P, "pin", lambda name: P.Pin(name=name, commit="x", root_env="E",
                                                     path="") if False else _FakePin(tmp_path / "outer"))
    st = CD._pin_status(inner / "h.h", "somepin")
    assert st["status"] == "off_pin"
    assert st["superproject_records"] == first
    assert st["checkout_commit"] != first


class _FakePin:
    def __init__(self, path):
        self._p = path

    def checkout(self):
        return self._p


def test_a_header_claim_from_an_unpinned_file_is_not_reported_as_pinned():
    """On this checkout the shipped headers are off-pin; the surface must say so, not imply otherwise."""
    for target in CD.targets_with_facts():
        try:
            surf = CD.discover(target, require_pin=False)
        except CD.ProvenanceRefused:
            continue
        hdr = [f for f in surf.findings
               if f.evidence and f.evidence[0].rung == "isa_header"]
        if not hdr:
            continue
        for f in hdr:
            assert f.pin_status is not None
            if f.pin_status not in CD._PIN_OK:
                assert "NOT A PINNED CLAIM" in f.detail
        return
    pytest.skip("no target in this checkout resolves a readable ISA header")
