"""The Gemmini timing space and probe backend: decode and emission from the RTL facts, and their pins.

Fixtures are a subset of the selected GemminiGsimFireSimAlignedConfig facts and the annotated-Spike
profile's price roster (``merlin/tests/data/timing_probes``). Nothing here runs a simulator.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from merlin_experiments.timing_probes import backends, decode, space
from merlin_experiments.timing_probes.selection import Unrealizable

from merlin.common.paths import repo_root

DATA = repo_root() / "merlin/tests/data/timing_probes"
PROBES = repo_root() / "examples/gemmini/verification/timing_probes"


@pytest.fixture(scope="module")
def facts():
    return json.loads((DATA / "gemmini_facts.json").read_text())


@pytest.fixture(scope="module")
def timing_space(facts):
    profile = json.loads((DATA / "gemmini_timing_profile.json").read_text())
    return space.load_space(PROBES / "timing_space.yaml", facts=facts, profile=profile, target="gemmini")


@pytest.fixture(scope="module")
def backend(timing_space, facts):
    return backends.load_backend(timing_space, facts=facts, contract={})


def test_class_status_follows_the_selected_facts(timing_space):
    status = {c.name: c.status for c in timing_space.classes}
    for name in ("store_act2", "store_act3", "store_act4"):
        assert status[name] == "support_unknown"  # the facts do not state the normalizer
    assert {status[f"normalize{k}"] for k in range(1, 8)} == {"undecodable"}  # no LocalAddr layout
    assert status["counter"] == "out_of_domain"
    assert {status[n] for n in ("compute_os", "compute_os_relu", "compute_ws_relu", "pool_store_act1")} == {"required"}


def _interface(facts, name):
    return next(row for row in facts["facts"]["interfaces"] if row["name"] == name)


def test_every_selector_and_field_comes_from_the_facts(timing_space, facts):
    layouts = _interface(facts, "register_bundle_layouts")["bundles"]
    assert timing_space.selector.offset == layouts["ConfigMvoutRs1"]["fields"]["cmd_type"]["offset"]
    ex = timing_space.states["EX"]["slots"]
    assert (ex["dataflow"].offset, ex["dataflow"].width) == (2, 1)
    assert set(timing_space.funct_names.values()) >= {"LOAD2_CMD", "LOAD3_CMD", "COMPUTE_AND_STAY_CMD"}


@pytest.mark.parametrize(
    "name",
    [
        "compute_ws",
        "compute_ws_relu",
        "compute_os",
        "compute_os_relu",
        "load",
        "store_act1",
        "pool_store",
        "pool_store_act1",
    ],
)
@pytest.mark.parametrize("shape", ["full", "partial"])
def test_a_realized_probe_decodes_to_the_class_it_was_asked_for(timing_space, backend, name, shape):
    cls = timing_space.by_name()[name]
    request = {"class": name, "instructions": list(cls.instructions), "state": cls.state, "shape": shape}
    decoded = decode.decode(timing_space, backend.realize(request, timing_space).commands)
    assert decoded.counts[name] >= 1
    assert not decoded.unclassified
    if len(cls.instructions) > 1 and (shape == "full" or name == "load"):
        assert {i for (c, i) in decoded.variants if c == name} == set(cls.instructions)


def test_partial_weight_stationary_shape_issues_computes_without_preload(timing_space, backend):
    cls = timing_space.by_name()["compute_ws"]
    request = {"class": "compute_ws", "instructions": list(cls.instructions), "state": cls.state, "shape": "partial"}
    decoded = decode.decode(timing_space, backend.realize(request, timing_space).commands)
    assert decoded.counts["compute_ws"] > decoded.counts["preload"]


def test_requests_outside_the_pipeline_are_refused(timing_space, backend):
    counter = timing_space.by_name()["counter"]
    with pytest.raises(Unrealizable):
        backend.realize(
            {"class": "counter", "instructions": list(counter.instructions), "state": {}, "shape": "full"}, timing_space
        )
    act4 = timing_space.by_name()["store_act4"]
    with pytest.raises(Unrealizable, match="normalization"):
        backend.realize(
            {"class": "store_act4", "instructions": list(act4.instructions), "state": act4.state, "shape": "full"},
            timing_space,
        )


def test_rendered_program_uses_the_facts_opcode_and_the_reviewed_timing_window(timing_space, backend, facts):
    class Timing:
        def source_marker(self):
            return "/* merlin.kernel_call_timing.v1 marker */"

        def declarations(self):
            return ["static unsigned long counter(void){ return 0; }"]

        def begin(self):
            return ["  /* begin */"]

        def end(self):
            return ["  /* end */"]

    cls = timing_space.by_name()["compute_os"]
    realized = backend.realize(
        {"class": "compute_os", "instructions": list(cls.instructions), "state": cls.state, "shape": "partial"},
        timing_space,
    )
    source = backend.render(realized, Timing(), probe_id="p")
    opcode = _interface(facts, "funct_decode_table")["custom_opcode"]
    assert source.count(f".insn r {opcode}, ") == sum(c.name != "FENCE" for c in realized.commands)
    assert source.index("/* begin */") < source.index("kernel_entry();") < source.index("kernel_complete();")
    assert source.index("kernel_complete();") < source.index("/* end */")


def test_protocol_codes_match_their_pinned_rtl_sources(timing_space):
    """Each evidence pin names the selected Gemmini source; its declarations carry the codes used."""
    root = os.environ.get("MERLIN_CHIPYARD") or os.environ.get("MERLIN_EXT_CHIPYARD")
    if not root or not Path(root, "generators/gemmini").is_dir():
        pytest.skip("selected Chipyard checkout not available")
    texts = {}
    for pin in timing_space.document["evidence"]:
        path = Path(root, pin["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"], pin["path"]
        texts[path.name] = path.read_text()
    declared = {}
    for line in texts["GemminiISA.scala"].splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0] == "val" and fields[2] == "=" and fields[3].endswith(".U"):
            declared[fields[1]] = fields[3].removesuffix(".U")
    codes = {name: entry["code"] for name, entry in timing_space.states.items()}
    assert {
        k: int(declared[f"CONFIG_{n}"]) for k, n in (("EX", "EX"), ("LD", "LOAD"), ("ST", "STORE"), ("NORM", "NORM"))
    } == codes
    assert "val OS, WS, BOTH = Value" in texts["Dataflow.scala"]
    assert "val RELU = 1.U" in texts["Activation.scala"]
    assert "(cmd.bits.cmd.inst.funct === LOAD2_CMD) -> 1.U" in texts["LoadController.scala"]
