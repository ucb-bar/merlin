"""The external-kernel bench must refuse the three ways a kernel comparison quietly lies.

Each test here is a failure mode that was FOUND in a real external bundle on this host, not a
hypothetical: a kernel that is declared but never called, a compile-time guard that substitutes the
stock library under the tuned kernel's name, and a measured window that contains host work on one
side and not the other. A bench that cannot detect these produces numbers that look fine and are not
about what the row says they are about.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys

import pytest

from merlin.common.paths import repo_root

SCRIPTS = repo_root() / "merlin" / "experiments" / "gemmini_perf_bench" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

external_kernel_bundle = pytest.importorskip("external_kernel_bundle")
external_kernel_table = pytest.importorskip("external_kernel_table")


GUARDED = """
#define ACME_K0_INPUT_ROWS   5184
#define ACME_K0_WEIGHT_ROWS  2304
#define ACME_K0_FITS (SPAD >= 7488)

static void acme_kernel0_conv(int batch, const elem_t *in, elem_t *out)
{
#if !ACME_K0_FITS
    stock_tiled_conv_auto(batch, in, out);
#else
    /* the tuned body */
#endif
}

static void acme_kernel1_conv(int batch, const elem_t *in, elem_t *out)
{
    /* no guard at all: this kernel's fit is not asserted by anything */
    tuned_body(batch, in, out);
}

#define ACME_K2_INPUT_ROWS   4032
#define ACME_K2_WEIGHT_ROWS  9216
#define ACME_K2_FITS (SPAD >= 13248)

static void acme_kernel2_conv(int batch, const elem_t *in, elem_t *out)
{
#if !ACME_K2_FITS
    stock_tiled_conv_auto(batch, in, out);
#else
    /* the tuned body */
#endif
}
"""

GEOMETRY = {"DIM": 16, "BANK_NUM": 4, "BANK_ROWS": 4096, "ACC_ROWS": 1024, "MAX_BYTES": 64}

DESCRIPTOR_ROWS = [
    {
        "kernel": "acme_kernel0_conv",
        "op": "conv2d",
        "filter": "3x3",
        "in_ch": "64",
        "out_ch": "64",
        "stride": "1",
        "spatial": "56x56",
        "call_sites": "3",
        "spad_rows_needed": "7488",
        "status": "used",
    },
    {
        "kernel": "acme_kernel1_conv",
        "op": "conv2d",
        "filter": "3x3",
        "in_ch": "128",
        "out_ch": "128",
        "stride": "1",
        "spatial": "28x28",
        "call_sites": "0",
        "spad_rows_needed": "",
        "status": "dead_code_never_called",
    },
    {
        "kernel": "acme_kernel2_conv",
        "op": "conv2d",
        "filter": "3x3",
        "in_ch": "256",
        "out_ch": "256",
        "stride": "1",
        "spatial": "14x14",
        "call_sites": "5",
        "spad_rows_needed": "13248",
        "status": "used",
    },
]


def _capture(layers):
    """A model capture in the shape ``layer_library_table.unique_layers`` reads."""
    nodes = []
    for ci, co, dim, count in layers:
        for i in range(count):
            nodes.append(
                {
                    "op": "Conv",
                    "name": f"/conv_{ci}_{dim}_{i}",
                    "weight_shape": [co, ci, 3, 3],
                    "in_shape": [1, ci, dim, dim],
                    "out_shape": [1, co, dim, dim],
                    "attributes": {"strides": [1, 1], "pads": [1, 1, 1, 1]},
                    "relu": True,
                    "fused_requant_scale": 0.00390625,
                }
            )
    return {"nodes": nodes}


def _bundle(tmp_path, *, rows=None, header=GUARDED, manifest=True):
    root = tmp_path / "bundle"
    (root / "descriptors").mkdir(parents=True)
    (root / "src").mkdir(parents=True)
    kernels_h = root / "src" / "acme_kernels.h"
    kernels_h.write_text(header, encoding="utf-8")
    descriptor = root / "descriptors" / "kernels.csv"
    rows = DESCRIPTOR_ROWS if rows is None else rows
    with descriptor.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if manifest:
        lines = []
        for path in (kernels_h, descriptor):
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}")
        (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


# --- the parser the whole bench rests on -------------------------------------------------------


def test_int_defines_reads_plain_integers_and_leaves_expressions_absent():
    """A define whose body is an expression is NOT half-understood: it is simply not in the result, so
    a caller that needs it fails closed instead of reading a guessed value."""
    got = external_kernel_bundle.int_defines(
        "#define BANK_NUM 4\n#define SPAD (BANK_NUM * BANK_ROWS)\n#define NEG -3\n#define BANK_ROWS 4096\n"
    )
    assert got == {"BANK_NUM": 4, "BANK_ROWS": 4096, "NEG": -3}
    assert "SPAD" not in got


def test_find_header_locates_the_geometry_by_content(tmp_path):
    """Which header carries the design geometry is a property of the include path, not of a file name."""
    root = tmp_path / "inc"
    (root / "nested").mkdir(parents=True)
    (root / "unrelated.h").write_text("#define DIM 16\n", encoding="utf-8")
    (root / "nested" / "renamed_params.h").write_text(
        "\n".join(f"#define {k} {v}" for k, v in GEOMETRY.items()), encoding="utf-8"
    )
    path, defines = external_kernel_bundle.find_header([root], external_kernel_bundle.GEOMETRY_MACROS)
    assert path.name == "renamed_params.h"
    assert {k: defines[k] for k in GEOMETRY} == GEOMETRY


def test_find_header_refuses_when_no_header_carries_the_geometry(tmp_path):
    (tmp_path / "only.h").write_text("#define DIM 16\n", encoding="utf-8")
    with pytest.raises(LookupError):
        external_kernel_bundle.find_header([tmp_path], external_kernel_bundle.GEOMETRY_MACROS)


# --- trap 2: the silent fallback ----------------------------------------------------------------


def test_fit_guard_and_its_fallback_are_read_off_the_kernel_that_carries_them():
    defines = external_kernel_bundle.int_defines(GUARDED)
    guard = external_kernel_bundle.fit_predicate(GUARDED, "acme_kernel0_conv", defines)
    assert guard.macro == "ACME_K0_FITS"
    assert guard.fallback_symbol == "stock_tiled_conv_auto"
    assert guard.footprint_rows == 7488


def test_an_unguarded_kernel_does_not_inherit_the_next_kernel_s_guard():
    """The scan is bounded at the next definition. Unbounded, it walks out of a kernel that has no
    guard and reports the FOLLOWING kernel's -- which reads as "fit asserted" for a kernel nothing
    asserted, and is exactly how a stock-library fallback gets measured under a tuned kernel's name."""
    defines = external_kernel_bundle.int_defines(GUARDED)
    with pytest.raises(LookupError):
        external_kernel_bundle.fit_predicate(GUARDED, "acme_kernel1_conv", defines)


def test_a_kernel_without_a_guard_is_carried_as_unbenchable_not_as_fitting(tmp_path):
    bundle = external_kernel_bundle.load_bundle(_bundle(tmp_path))
    by_name = {k.name: k for k in bundle.kernels}
    assert by_name["acme_kernel1_conv"].guard is None
    assert by_name["acme_kernel1_conv"].guard_error
    assert by_name["acme_kernel1_conv"].benchable is False
    evidence = external_kernel_bundle.fit_evidence(by_name["acme_kernel1_conv"], GEOMETRY)
    assert evidence["fits"] is None
    assert evidence["asserted_at_compile_time"] is False


def test_fit_needs_two_agreeing_sources(tmp_path):
    """The descriptor's footprint and the kernel source's own constants must agree. One of them alone
    is a claim; together they are a cross-check, and a disagreement stops the row."""
    rows = [dict(r) for r in DESCRIPTOR_ROWS]
    rows[0]["spad_rows_needed"] = "7000"
    bundle = external_kernel_bundle.load_bundle(_bundle(tmp_path, rows=rows))
    kernel = next(k for k in bundle.kernels if k.name == "acme_kernel0_conv")
    evidence = external_kernel_bundle.fit_evidence(kernel, GEOMETRY)
    assert evidence["sources_agree"] is False
    assert evidence["spad_rows_declared_by_descriptor"] == 7000
    assert evidence["spad_rows_derived_from_source"] == 7488


def test_a_kernel_too_big_for_the_design_does_not_read_as_fitting(tmp_path):
    bundle = external_kernel_bundle.load_bundle(_bundle(tmp_path))
    kernel = next(k for k in bundle.kernels if k.name == "acme_kernel2_conv")
    small = {**GEOMETRY, "BANK_ROWS": 2048}
    assert external_kernel_bundle.fit_evidence(kernel, small)["fits"] is False
    assert external_kernel_bundle.fit_evidence(kernel, GEOMETRY)["fits"] is True


def test_the_rendered_program_asserts_the_fit_macro_and_names_the_refused_fallback(tmp_path):
    """The compiler, not the harness, is what makes the fallback impossible."""
    bundle = external_kernel_bundle.load_bundle(_bundle(tmp_path))
    kernel = next(k for k in bundle.kernels if k.name == "acme_kernel0_conv")
    source = external_kernel_table.external_kernel_c(kernel, {"batch": 1, "relu": True, "scale": 0.00390625})
    assert "_Static_assert(ACME_K0_FITS" in source
    assert "stock_tiled_conv_auto" in source
    assert f"{kernel.name}(1, input, weights, bias, output" in source
    assert external_kernel_table.EXTERNAL_SYMBOL in source


def test_the_requant_scale_is_emitted_as_an_exact_float32_literal():
    """A decimal literal would be read back as a double and re-round; the digest check would then fail
    for a kernel that is actually correct, or pass one that is not."""
    literal = external_kernel_table.c_float32(0.00397842036530451)
    assert literal.startswith("0x") and literal.endswith("f")
    assert float.fromhex(literal[:-1]) == float(__import__("numpy").float32(0.00397842036530451))


# --- trap 1: isolated flatters embedded ----------------------------------------------------------


def test_the_one_time_cost_is_inside_the_cold_window_and_outside_the_warm_one():
    cold = external_kernel_table.window_for("external", "cold_single")
    warm = external_kernel_table.window_for("external", "warm_then_measured")
    assert "host_pad_buffer_zero_once" in cold
    assert "host_pad_buffer_zero_once" not in warm
    # everything else is in both: the per-call host padding does NOT amortize away
    assert "host_zero_pad_copy" in cold and "host_zero_pad_copy" in warm


def test_the_library_window_never_contains_host_padding():
    """The external kernels zero-pad on the host; the library pads in hardware. Printing both under
    one 'cycles' column without saying so is how a schedule comparison becomes a padding comparison."""
    for protocol in external_kernel_table.PROTOCOLS:
        assert external_kernel_table.window_for("library", protocol) == ["accelerator_conv"]


def test_per_call_site_is_undefined_without_both_windows():
    assert external_kernel_table.per_call_site(None, 100, 3) is None
    assert external_kernel_table.per_call_site(100, None, 3) is None
    assert external_kernel_table.per_call_site(100, 100, 0) is None


def test_per_call_site_charges_the_one_time_cost_once():
    """Three call sites, a first invocation costing 400 and later ones 100: 200 per site, not 400."""
    assert external_kernel_table.per_call_site(400, 100, 3) == pytest.approx(200.0)
    assert external_kernel_table.per_call_site(400, 100, 1) == pytest.approx(400.0)


# --- trap 3 and the selection rules ---------------------------------------------------------------


def test_a_declared_but_never_called_kernel_is_refused_with_its_reason(tmp_path):
    bundle = external_kernel_bundle.load_bundle(_bundle(tmp_path))
    capture = _capture([(64, 64, 56, 3), (128, 128, 28, 3), (256, 256, 14, 5)])
    paired, refused = external_kernel_table.select_layers(bundle, capture)
    assert [k.name for k, _, _ in paired] == ["acme_kernel0_conv", "acme_kernel2_conv"]
    dead = [r for r in refused if r["kernel"] == "acme_kernel1_conv"]
    assert dead and "never called" in dead[0]["why"]
    # the capture DOES contain that shape, three times over -- being present in the model is not
    # being used by the bundle, and the bench must not confuse the two
    assert any(n["in_shape"][1] == 128 for n in capture["nodes"])


def test_a_call_site_count_the_capture_contradicts_stops_the_row(tmp_path):
    """The per-call-site figure is only meaningful if the descriptor and the model agree on how many
    call sites there are."""
    rows = [dict(r) for r in DESCRIPTOR_ROWS]
    rows[0]["call_sites"] = "9"
    bundle = external_kernel_bundle.load_bundle(_bundle(tmp_path, rows=rows))
    capture = _capture([(64, 64, 56, 3), (256, 256, 14, 5)])
    paired, refused = external_kernel_table.select_layers(bundle, capture)
    assert [k.name for k, _, _ in paired] == ["acme_kernel2_conv"]
    stopped = [r for r in refused if r["kernel"] == "acme_kernel0_conv"]
    assert stopped and stopped[0]["descriptor_call_sites"] == 9 and stopped[0]["capture_multiplicity"] == 3


def test_a_shape_the_capture_does_not_hold_is_refused_rather_than_invented(tmp_path):
    bundle = external_kernel_bundle.load_bundle(_bundle(tmp_path))
    paired, refused = external_kernel_table.select_layers(bundle, _capture([(256, 256, 14, 5)]))
    assert [k.name for k, _, _ in paired] == ["acme_kernel2_conv"]
    assert any("capture holds 0 layers" in r["why"] for r in refused)


def test_a_bundle_whose_bytes_moved_is_not_usable(tmp_path):
    root = _bundle(tmp_path)
    (root / "src" / "acme_kernels.h").write_text(GUARDED + "\n/* edited */\n", encoding="utf-8")
    with pytest.raises(ValueError):
        external_kernel_bundle.load_bundle(root)


def test_a_directory_without_a_descriptor_is_not_a_bundle(tmp_path):
    with pytest.raises(FileNotFoundError):
        external_kernel_bundle.load_bundle(tmp_path)


# --- what the bench refuses to compare -------------------------------------------------------------


def test_the_routes_that_cannot_express_a_padded_conv_say_so():
    """Two of the three routes this repo can lower through cannot express a 3x3 stride-1 PADDED conv
    at all. That is a finding about the compiler, so the bench records it instead of omitting the row
    and leaving the reader to infer that only the package route was tried."""
    spec = {
        "op": "conv2d",
        "batch": 1,
        "in_dim": 56,
        "in_channels": 64,
        "out_channels": 64,
        "kernel": 3,
        "stride": 1,
        "padding": 1,
        "relu": True,
    }
    routes = external_kernel_table.route_reachability(spec)
    assert set(routes) == set(external_kernel_table.OUR_ROUTES)
    assert routes["schedule"]["reachable"] is False and routes["schedule"]["why"]
    for name, verdict in routes.items():
        assert verdict["reachable"] is True or verdict["why"]


def test_the_schedule_route_does_admit_the_shape_it_was_built_for():
    """The refusal above must be about THIS shape, not about the probe always saying no."""
    spec = {
        "op": "conv2d",
        "batch": 1,
        "in_dim": 56,
        "in_channels": 64,
        "out_channels": 64,
        "kernel": 1,
        "stride": 1,
        "padding": 0,
        "relu": True,
    }
    assert external_kernel_table.route_reachability(spec)["schedule"]["reachable"] is True


def test_the_bundle_s_own_parameter_header_is_compared_with_the_design_s(tmp_path):
    """The kernels are compiled against THIS design's header, so whether the bundle's own header
    describes the same machine is a fact the product carries -- the kernels' scratchpad addresses are
    absolute literals searched against whatever machine the bundle's header described."""
    root = _bundle(tmp_path)
    (root / "src" / "their_params.h").write_text(
        "\n".join(f"#define {k} {v}" for k, v in {**GEOMETRY, "BANK_ROWS": 2048}.items()), encoding="utf-8"
    )
    (root / "MANIFEST.sha256").unlink()
    bundle = external_kernel_bundle.load_bundle(root)
    agreement = external_kernel_bundle.geometry_agreement(bundle, GEOMETRY)
    assert agreement["bundle_ships_a_parameter_header"] is True
    assert agreement["geometry_matches"] is False
    assert agreement["differing_macros"]["BANK_ROWS"] == (2048, 4096)


def test_a_bundle_shipping_no_parameter_header_says_so(tmp_path):
    bundle = external_kernel_bundle.load_bundle(_bundle(tmp_path))
    assert external_kernel_bundle.geometry_agreement(bundle, GEOMETRY) == {"bundle_ships_a_parameter_header": False}


def test_each_row_keeps_the_program_that_produced_it(tmp_path):
    """A build tree is scratch and gets pruned. A row whose C nobody can read again is a number with
    no way back to what it measured, so the source moves out of the row and into the product."""
    rows = [
        {"arm": "external", "kernel": "k0", "protocol": "cold_single", "source": "int main(void){}\n"},
        {"arm": "package", "kernel": "k0", "protocol": "warm_then_measured"},  # adopted: carries none
    ]
    written = external_kernel_table.write_programs(tmp_path / "programs", rows)
    assert written == {"external_k0_cold_single": "programs/external_k0_cold_single.c"}
    assert (tmp_path / "programs" / "external_k0_cold_single.c").read_text() == "int main(void){}\n"
    assert rows[0]["program"] == "programs/external_k0_cold_single.c"
    assert rows[0]["program_sha256"] == hashlib.sha256(b"int main(void){}\n").hexdigest()
    assert "source" not in rows[0]  # the text lives in the file, not twice in the JSON
    assert "program" not in rows[1]  # and is not invented for a row that never had one


def test_a_route_that_admits_a_shape_but_fails_to_lower_it_is_not_left_reading_as_working():
    """Admitting a layer and producing code for it are different claims. Leaving the optimistic one
    standing beside a failed row is how a table says a route works when the run says otherwise."""
    routes = {"k": {"package": {"reachable": True, "why": "admits the shape; whether it lowers is measured"}}}
    external_kernel_table.apply_package_outcome(
        routes, [{"kernel": "k", "status": "CodegenError: command timed out after 900s"}]
    )
    assert routes["k"]["package"]["reachable"] is False
    assert "did not produce code" in routes["k"]["package"]["why"]

    ok = {"k": {"package": {"reachable": True, "why": "admits the shape; whether it lowers is measured"}}}
    external_kernel_table.apply_package_outcome(ok, [{"kernel": "k", "status": "ok", "cycles": 4242}])
    assert ok["k"]["package"]["reachable"] is True
    assert "4,242" in ok["k"]["package"]["why"]


def test_a_window_that_was_never_run_does_not_read_as_a_failure():
    """The package arm is measured under one protocol only. A blank that looks like FAILED is how a
    route gets blamed for a run nobody attempted."""
    assert external_kernel_table._cyc(None, measured=False) == "not run"
    assert external_kernel_table._cyc(None, measured=True) == "FAILED"
    assert external_kernel_table._cyc(1234) == "1,234"


def test_the_refused_comparisons_are_stated_in_the_product_not_left_to_the_reader():
    """The bundle quotes its own cycle figures for these same kernels a few lines from the kernels.
    A reader who does not find the refusal written down will make the comparison themselves."""
    reasons = {n["comparison"]: n["why"] for n in external_kernel_table.NOT_COMPARED}
    assert any("quotes" in c for c in reasons)
    assert any("whole-model" in c for c in reasons)
    assert all(why.strip() for why in reasons.values())


def test_an_adopted_package_row_must_share_the_numerics_contract(tmp_path):
    """A package row measured under a different contract is not comparable; it is dropped by the
    caller on the digest it carries, so the digest has to survive the fold."""
    table = tmp_path / "layer_package_table.json"
    table.write_text(
        json.dumps(
            {
                "contract_digest": "deadbeef",
                "package": "/somewhere/submission",
                "package_digest": "abc123",
                "rows": [{"sig": "SIG", "cycles": 1234, "numerics": "exact"}],
            }
        ),
        encoding="utf-8",
    )
    rows = external_kernel_table.package_rows(table, {"SIG": "acme_kernel0_conv"})
    assert len(rows) == 1
    assert rows[0]["contract_digest"] == "deadbeef"
    assert rows[0]["arm"] == "package"
    assert rows[0]["window"] == external_kernel_table.window_for("package", "warm_then_measured")
    assert "bias is zero" in rows[0]["operand_note"]
    # a layer this bench did not pair is not folded in
    assert external_kernel_table.package_rows(table, {"OTHER": "x"}) == []
