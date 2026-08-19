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
