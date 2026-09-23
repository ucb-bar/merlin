"""The one structural input of the roofline that no target publishes, and what it takes to close it.

:mod:`merlin.perf.derived_bound` derives an achievable rate from a target's own RTL facts. On every
target in this repo it comes back UNKNOWN for exactly one reason: nothing states the machine's
TRANSFER WIDTH, so the traffic term cannot be built and the bound falls back to a compute-only
``partial_rate``. That is the honest behaviour -- dropping a term loosens the bound and the module
says so -- but a refusal that has stood on every target since the module was written needs to be a
work item rather than a sentence.

So two things are pinned here:

* the refusal NAMES the fact class to extract (:data:`~merlin.perf.derived_bound.MOVEMENT_WIDTH_FACTS`),
  so the gap is actionable from the artifact alone;
* the loop CLOSES the moment a target publishes one of those facts -- proved against a synthetic facts
  document, because none of this repo's targets publishes it yet and a test that waited for one would
  be a test that never ran.

And the thing that must NOT happen: a width that means something else being read as a transfer rate.
A PE operand port, a scratchpad row and a register bundle are all widths in bits on these targets and
none of them is a bandwidth; substituting one would put a fabricated number under a bound people
quote. The last test is the one that catches that.
"""

from __future__ import annotations

from merlin.perf import derived_bound as DB
from merlin.perf.decompose import is_unknown

#: A facts body with everything the bound needs EXCEPT a movement width. Synthetic on purpose: the
#: point is what the reader does with a shape, and a real target's document would also be asserting
#: that target's numbers.
_BODY: dict = {
    "arrays": [{"name": "mesh", "rows": 8, "cols": 8, "mac_idiom": {"muls": 1}}],
    "datapaths": [
        {"name": "input", "dtype": "i8", "bits": 8},
        {"name": "accumulator", "dtype": "i32", "bits": 32},
    ],
    "memories": [
        {"name": "scratchpad", "bytes": 4096, "depth": 32},
        {"name": "accumulator", "bytes": 2048, "depth": 8},
    ],
    "interfaces": [{"name": "readout", "offchip_dtypes": ["i8"]}],
}


def _machine(body: dict) -> DB.Machine:
    return DB.machine_from_facts("synthetic", facts={"facts": body}, measure_fill=False)


def test_without_the_fact_the_term_is_unknown_and_the_refusal_says_what_to_extract() -> None:
    machine = _machine(_BODY)
    assert is_unknown(machine.dram_bytes_per_cycle)
    reason = machine.refusals["dram_bytes_per_cycle"]
    for path, _what in DB.MOVEMENT_WIDTH_FACTS:
        assert path in reason, f"the refusal does not name {path}, so nobody can act on it"
    assert "looser, never tighter" in reason, "dropping a term must state which way the bound moves"


def test_an_interface_that_states_its_transfer_width_closes_the_loop() -> None:
    """The whole point. One fact, published by the target's own extractor, and the bound resolves --
    with no change to this module, which is what makes it a fact class rather than a special case."""
    body = {**_BODY, "interfaces": [*_BODY["interfaces"], {"name": "dma", "transfer_bits": 128}]}
    machine = _machine(body)
    assert machine.dram_bytes_per_cycle == 16.0
    assert "dram_bytes_per_cycle" not in machine.refusals
    assert "128" in machine.provenance["dram_bytes_per_cycle"]


def test_a_movement_datapath_states_it_the_other_way_and_is_read_the_same() -> None:
    body = {**_BODY, "datapaths": [*_BODY["datapaths"], {"name": "dma", "role": "movement", "bits": 256}]}
    assert _machine(body).dram_bytes_per_cycle == 32.0


def test_beats_per_cycle_multiplies_the_beat_width() -> None:
    body = {**_BODY, "interfaces": [*_BODY["interfaces"], {"name": "dma", "beat_bits": 64, "beats_per_cycle": 2}]}
    machine = _machine(body)
    assert machine.dram_bytes_per_cycle == 16.0
    assert "2 beat(s)/cycle" in machine.provenance["dram_bytes_per_cycle"]


def test_the_widest_evidenced_path_wins() -> None:
    """A machine whose load path is wider than its store path moves traffic faster than the narrow
    figure says, so the wide one keeps this an upper bound rather than a cap a schedule could beat."""
    body = {
        **_BODY,
        "interfaces": [
            *_BODY["interfaces"],
            {"name": "narrow", "transfer_bits": 64},
            {"name": "wide", "transfer_bits": 512},
        ],
    }
    assert _machine(body).dram_bytes_per_cycle == 64.0


def test_a_width_that_is_not_a_transfer_width_is_never_read_as_one() -> None:
    """The failure this reader is shaped to avoid. Every block below states a width in bits and none
    of them states a bandwidth: an operand element port, an accumulator port, a memory row, a register
    bundle. Reading any of them would claim a DRAM rate this design never published -- and on one real
    target here the operand port is 8 bits, i.e. a claimed 1 byte/cycle, 16x wrong in the direction
    that makes every schedule look memory-bound."""
    body = {
        **_BODY,
        "memories": [*_BODY["memories"], {"name": "vmem", "row_bits_rtl": 256, "bytes": 1024, "depth": 8}],
        "interfaces": [
            *_BODY["interfaces"],
            {"name": "register_bundle_layouts", "bundles": {"Rs1": {"width": 64}}},
        ],
    }
    machine = _machine(body)
    assert is_unknown(machine.dram_bytes_per_cycle), machine.provenance.get("dram_bytes_per_cycle")


def test_the_resolved_term_actually_produces_a_bound_rather_than_a_partial_one() -> None:
    """Resolving the fact is only worth something if the bound stops being UNKNOWN. ``rate`` is
    UNKNOWN whenever ANY term is unresolved, so this is the end-to-end statement that the traffic term
    was the one holding it back."""
    body = {**_BODY, "interfaces": [*_BODY["interfaces"], {"name": "dma", "transfer_bits": 128}]}
    before = DB.achievable_bound(DB.Gemm(32, 32, 32), _machine(_BODY))
    after = DB.achievable_bound(DB.Gemm(32, 32, 32), _machine(body))
    assert DB.TRAFFIC_TERM in before.unresolved and not before.known
    assert after.known and not after.unresolved, after.reasons
    assert after.limiter in (DB.COMPUTE_TERM, DB.TRAFFIC_TERM)
