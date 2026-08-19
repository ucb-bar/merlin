"""What a matrix-unit delivery may claim about itself.

A matrix-routed image is the one row in a package whose evidence is split, and every way of getting that
wrong has already happened once: a README that told the recipient a stand-in twin "grades bit-exact" in a
package where the twin was never run, a section describing the shipped binaries as containing "no vector
instructions at all" because a matrix image is not an RVV one, and a manifest that named a unit and a
configuration without ever saying which tile edge the kernel was compiled for. Each is a sentence a
reader cannot check, so each is pinned here.
"""
from __future__ import annotations

import importlib.util
import sys

from merlin.common.paths import repo_root
from merlin.runtime import boards


def _load_packager():
    p = repo_root() / "build_tools" / "scripts" / "make_delivery.py"
    spec = importlib.util.spec_from_file_location("make_delivery", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _matrix_binary(**over):
    matrix = {"unit": "a_unit", "config": "SomeConfig", "unit_instruction_counts": {"ACC": 3},
              "tile_edge": 64, "alignment_bytes": 32, "scratch_bytes": 8192,
              "parallel_tiles": True, "twin_simulated": False, "lowering_digest": "ld",
              "twin_build_hash": "tb", "twin_spike_cycles": None, "gated_via": "..."}
    matrix.update(over.pop("matrix", {}))
    b = {"model": "m", "elf": "m_h2_matrix.elf", "harts": 2, "dtype": "int8", "backend": "matrix",
         "build_hash": "z", "ram_bytes": 268435456, "spike_cycles": None, "gate_ok": False,
         "tier_ok": None, "cos": None, "rel": None, "upload_estimate_s": 100, "matrix": matrix}
    b.update(over)
    return b


def _manifest(binaries, **over):
    brd = boards.board("kodiak_opu_2core")
    man = {"board": {"name": brd.name, "dram_bytes": brd.dram_bytes, "harts": brd.harts,
                     "vlen": brd.vlen, "console": brd.console, "notes": brd.notes},
           "notes": [], "binaries": binaries, "problems": [], "vector_probe": None,
           "firesim_evidence": None, "merlin_commit": "abc", "validated_on": "x"}
    man.update(over)
    return brd, man


def test_the_readme_states_the_geometry_the_kernel_was_compiled_for():
    """`unit` and `config` do not tell anyone what the kernel assumes. The tile edge is the number that
    has to match the design, and a kernel built for the wrong one links, runs and computes garbage."""
    md = _load_packager()
    brd, man = _manifest([_matrix_binary()])
    txt = md._readme(brd, man)
    assert "SomeConfig" in txt
    assert "tile edge **64**" in txt


def test_an_unrun_twin_is_not_described_as_graded():
    md = _load_packager()
    brd, man = _manifest([_matrix_binary()])
    txt = md._readme(brd, man)
    assert "grades bit-exact" not in txt
    assert "we did **not** run it" in txt
    # And nothing points the reader at a reference file the package does not contain.
    assert "expected_console.txt` beside these ELFs is the twin's" not in txt


def test_a_run_twin_is_described_as_graded():
    md = _load_packager()
    brd, man = _manifest([_matrix_binary(matrix={"twin_simulated": True,
                                                 "twin_spike_cycles": 1234})])
    txt = md._readme(brd, man)
    assert "grades bit-exact" in txt


def test_a_matrix_image_does_not_pull_in_the_scalar_section():
    """The section says the binaries it describes contain no vector instructions at all. A matrix image is
    full of them, and selecting it with `backend != "rvv"` said otherwise."""
    md = _load_packager()
    brd, man = _manifest([_matrix_binary()])
    txt = md._readme(brd, man)
    assert "how to use every core" not in txt


def test_package_notes_are_rendered_where_they_will_be_read():
    md = _load_packager()
    brd, man = _manifest([_matrix_binary()], notes=["nothing here ran on two-core hardware"])
    txt = md._readme(brd, man)
    assert "nothing here ran on two-core hardware" in txt
    # Before the binaries, not in a footnote after them.
    assert txt.index("nothing here ran on two-core hardware") < txt.index("## The binaries")


def test_the_provenance_line_names_a_backend_exception():
    md = _load_packager()
    line = md._provenance(False, set(), {"matrix"})
    assert "spike" in line and "matrix" in line and "not\nsimulated" not in line
    assert "EXCEPT" in line


def test_matrix_facts_keep_the_revision_and_drop_our_directory_layout():
    """A delivery leaves this host. The build's own record carries absolute paths into our checkouts; what
    the recipient needs is the geometry and which revision the encodings came from."""
    md = _load_packager()
    got = md._matrix_facts({
        "object": "/somewhere/private/shim.o", "source": "/somewhere/private/shim.c",
        "tile_edge": 64, "alignment_bytes": 32, "scratch_bytes": 8192, "parallel_tiles": True,
        "scalar_tile": False, "gaps": ["pin drifted"],
        "provenance": {"hardware_pins": {"a_pin": {
            "pin": "a_pin", "ok": False,
            "observed": {"path": "/somewhere/private/checkout", "commit": "f" * 40,
                         "branch": "a-branch", "remote": "https://example.invalid/x.git",
                         "dirty_files": 9, "dirty_paths": ["private/file"]},
            "drift": ["a source moved"]}},
            "source_digest": "abcd", "sources": ["/somewhere/private/Consts.scala"]}})
    assert got["tile_edge"] == 64 and got["parallel_tiles"] is True
    assert got["unit_revision"]["commit"] == "f" * 40
    assert got["unit_revision"]["verified"] is False
    assert got["unit_revision"]["drift"] == ["a source moved"]
    assert got["unit_source_digest"] == "abcd"
    assert got["shim_gaps"] == ["pin drifted"]
    flat = repr(got)
    assert "/somewhere/private" not in flat
    assert "dirty_paths" not in flat


def test_matrix_facts_of_a_non_matrix_build_are_empty():
    md = _load_packager()
    assert md._matrix_facts(None) == {}


def test_a_configurations_loader_instructions_come_from_the_port_it_is_built_against():
    """A second configuration of a known board is loaded the same way: the loader belongs to the port and
    the link, not to how many cores the design has."""
    md = _load_packager()
    brd, man = _manifest([_matrix_binary()])
    txt = md._readme(brd, man)
    assert md.LOADER_DOC[brd.zephyr_board].split("\n")[1] in txt


def test_a_configuration_does_not_inherit_another_boards_bring_up_history():
    """That text addresses a conversation ("your table", "your trap logs") about runs a different
    configuration's owner did. Attributing them to whoever receives this package is putting words in
    their mouth."""
    md = _load_packager()
    brd, man = _manifest([_matrix_binary()])
    txt = md._readme(brd, man)
    assert "Your trap logs" not in txt
    assert md.HISTORY_DOC["chipyard_kodiak"] not in txt


def test_the_diagnostic_build_of_a_matrix_image_is_audited_and_named():
    """The `_debug` build is the one someone falls back to when the plain image misbehaves. A diagnostic
    build that lost the routing would run, print a clean trace, grade correctly on the host core and prove
    nothing about the unit -- so its counts are recorded separately from the shipped image's and stated."""
    md = _load_packager()
    ship = _matrix_binary(matrix={"debug_unit_instruction_counts": {"ACC": 3}})
    dbg = _matrix_binary(elf="m_h2_matrix_debug.elf", debug=True)
    dbg.pop("matrix")
    brd, man = _manifest([ship, dbg])
    txt = md._readme(brd, man)
    assert "m_h2_matrix_debug.elf" in txt
    assert "refuses to ship a diagnostic build that lost the routing" in txt


def test_the_first_binary_to_run_is_named_from_this_package():
    """The recommendation used to name a model by hand, which is wrong for every package that does not
    contain it. It is a property of the set: the cheapest instrumented image to get on the wire."""
    md = _load_packager()
    big = _matrix_binary(elf="big_debug.elf", debug=True, upload_estimate_s=9000)
    small = _matrix_binary(elf="small_debug.elf", debug=True, upload_estimate_s=60)
    brd, man = _manifest([_matrix_binary(), big, small])
    txt = md._readme(brd, man)
    assert "Start with `small_debug.elf`" in txt
    assert "deepjscc" not in txt


def test_a_host_assisted_console_says_to_stay_attached_where_the_debug_lines_are_explained():
    """HTIF is not a self-contained UART: the loader is the host that services it. A detached run of the
    image that exists to explain itself prints nothing at all -- no banner, no STAGE, no fault -- which is
    indistinguishable from a dead image on silicon where "dead image" is the first hypothesis."""
    md = _load_packager()
    dbg = _matrix_binary(elf="m_h2_matrix_debug.elf", debug=True)
    dbg.pop("matrix")
    brd, man = _manifest([_matrix_binary(), dbg])
    assert brd.console == md.CONSOLE_HTIF
    txt = md._readme(brd, man)
    assert "Stay attached until `DONE`" in txt


def test_the_loader_section_does_not_name_models_a_package_may_not_ship():
    """It documents the LINK, and it is now shared by more than one configuration of the same port. Naming
    one package's models there recommends a binary the recipient does not have."""
    md = _load_packager()
    doc = md.LOADER_DOC["chipyard_kodiak"]
    for model in md.STATUS:
        assert model not in doc, f"the loader doc names {model}"


# --------------------------------------------------------------------------- memory and bundles
def test_an_image_whose_arena_is_short_is_reported_as_short():
    """The arena is the leftover of the linked region after the image, so the only way to know it is big
    enough is to subtract. An image that is short does not ship: it would run, print its stages, and die in
    its tail on a machine somebody waited hours for."""
    md = _load_packager()

    class _Rep:
        facts = {"image_memsz_mb": 1500.0}

    short = md._memory_facts({"ram_bytes": 3000 * 2**20, "allocation_bytes_total": 2000 * 2**20}, _Rep())
    assert short["arena_mb"] == 1500.0
    assert short["arena_short_mb"] == 500.0
    ok = md._memory_facts({"ram_bytes": 3600 * 2**20, "allocation_bytes_total": 2000 * 2**20}, _Rep())
    assert "arena_short_mb" not in ok


def test_a_bundle_directory_can_be_named_explicitly():
    """A bundle that is not the whole model is a legitimate deliverable, and its directory should say which
    section while the label stays readable in a filename. Guessing between the two conventions is how a
    package ships someone else's weights under this model's name."""
    md = _load_packager()
    label, where = md._bundle_spec("a_label:a_directory", "int8")
    assert label == "a_label"
    assert where.name == "a_directory"
    label, where = md._bundle_spec("plain", "int8")
    assert (label, where.name) == ("plain", "plain_int8_full")


def test_the_readme_states_what_actually_went_to_the_unit():
    """"The matrix unit is used" is compatible with one small GEMM out of a hundred going through it."""
    md = _load_packager()
    b = _matrix_binary(matrix={"routing": {
        "routed_contractions": 110, "distinct_signatures": 9, "skipped": [],
        "macs_routed": 195_890_000_000,
        "widest_routed": {"fqn": "lm_head", "shape": [1, 128, 256000, 2304], "macs": 7.5e10}}})
    brd, man = _manifest([b])
    txt = md._readme(brd, man)
    assert "110 contraction(s)" in txt
    assert "9 distinct signature(s)" in txt
    assert "195.89e9 MACs" in txt
    assert "nothing was left behind" in txt


def test_a_skipped_contraction_is_named_rather_than_averaged_away():
    md = _load_packager()
    b = _matrix_binary(matrix={"routing": {
        "routed_contractions": 3, "distinct_signatures": 1,
        "skipped": ["sym_x: init is not provably zero"], "macs_routed": 10}})
    brd, man = _manifest([b])
    txt = md._readme(brd, man)
    assert "NOT everything was routed" in txt
    assert "init is not provably zero" in txt


def test_a_multi_hour_upload_is_stated_in_the_run_instructions():
    """A 1.5 GB image on a 57600-baud link is days of wire time, and the loader polls with no timeout, so
    from outside it looks exactly like a hang. That belongs where someone decides whether to start, not in
    a table column and a warning field."""
    md = _load_packager()
    b = _matrix_binary(upload_estimate_s=266_220, upload_bytes=1533 * 2**20)
    brd, man = _manifest([b])
    txt = md._readme(brd, man)
    assert "the upload is the long pole" in txt
    assert "74." in txt, "the hours must be stated as a number"
    assert "are not\npractical" in txt, "past a day of wire time the verdict is not 'budget for it'"
    # And it appears before the loader command, not after it.
    assert txt.index("the upload is the long pole") < txt.index("## Run one")


def test_a_long_but_workable_upload_is_not_called_impractical():
    """A 40-minute upload is a wait to budget for. Calling it impractical spends the phrase where it is not
    true, and then it carries no weight where it is."""
    md = _load_packager()
    brd, man = _manifest([_matrix_binary(upload_estimate_s=2400, upload_bytes=13 * 2**20)])
    txt = md._readme(brd, man)
    assert "the upload is the long pole" in txt
    assert "not\npractical" not in txt
    assert "40 min" in txt
    # And a sub-hour alternative reads as minutes, not as "0.0 h".
    assert "0.0 h" not in txt


def test_a_short_upload_does_not_get_its_own_section():
    md = _load_packager()
    brd, man = _manifest([_matrix_binary(upload_estimate_s=120)])
    txt = md._readme(brd, man)
    assert "the upload is the long pole" not in txt


def test_a_wrapper_that_hand_picks_its_return_still_carries_the_memory_demand():
    """The build-only wrapper dropped these, so the arena check downstream compared against zero and
    passed an image it never examined. An unchecked arena is worse than an unsized one: it reads as
    verified."""
    md = _load_packager()
    kept = md._keep_memory({"elf": "x", "allocation_bytes_total": 7, "allocation_dynamic_calls": 1,
                            "activation_peak_bytes": None})
    assert kept == {"allocation_bytes_total": 7, "allocation_dynamic_calls": 1}


def test_no_start_small_advice_when_every_image_costs_the_same_days():
    """"Start with the smallest" is advice when the smallest is cheap and noise when it is 73.9 hours
    against 74.0. Advice that carries nothing teaches a reader to skip the paragraph."""
    md = _load_packager()
    a = _matrix_binary(elf="a.elf", upload_estimate_s=266_000, upload_bytes=1533 * 2**20)
    b = _matrix_binary(elf="b.elf", upload_estimate_s=266_220, upload_bytes=1533 * 2**20)
    brd, man = _manifest([a, b])
    txt = md._readme(brd, man)
    assert "start with `a.elf`" not in txt
