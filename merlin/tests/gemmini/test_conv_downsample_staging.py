"""The strided-load eligibility predicate, and what the mode removes from the input staging.

No accelerator execution: everything here is read out of the target's own convolution header and the
RTL facts the schedule already uses.
"""

import importlib.util
import json

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import target_registry


def _support_root():
    selected = target_registry.explicit_targets().get("gemmini")
    if selected is None:
        pytest.skip("requires explicit Gemmini support on MERLIN_TARGET_PATH", allow_module_level=True)
    info = target_registry.resolve("gemmini")
    assert info.base.resolve() == selected.resolve(), "selected support resolution drifted"
    return info.base


ROOT = repo_root()
SPEC = importlib.util.spec_from_file_location("conv_downsample", _support_root() / "backend/gemmini_conv_downsample.py")
DS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DS)

INCLUDE = ROOT / "examples/gemmini/phase1/contracts/harness_curated/gemmini-rocc-tests/include"
HEADER = INCLUDE / "gemmini.h"
FACTS = ROOT / "merlin/targets/gemmini/contracts/rtl_facts/facts.json"


@pytest.fixture(scope="module")
def header_text():
    if not HEADER.is_file():
        pytest.skip("the target's convolution header is unavailable; never substitute a remembered predicate")
    return HEADER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def facts():
    """The three RTL numbers the staging count needs, each from its own source.

    The mesh width comes from the elaborated facts, the element width from the datapath those facts
    record, and the DMA's transfer size from the params header. ``MAX_BLOCK_LEN`` itself is written
    there as an expression over the other two, so it is computed the same way rather than read as a
    literal that is not one.
    """
    if not (INCLUDE / "gemmini_params.h").is_file():
        pytest.skip("the target's params header is unavailable")
    raw = json.loads(FACTS.read_text())["facts"]
    mesh = next(row for row in raw["arrays"] if row.get("name") == "mesh")
    datapath = next(row for row in raw["datapaths"] if row.get("name") == "input")
    bits = int(str(datapath["dtype"]).lstrip("iu"))
    values = {}
    for line in (INCLUDE / "gemmini_params.h").read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text.startswith("#define "):
            continue
        name, _, value = text[len("#define ") :].strip().partition(" ")
        values[name] = value.strip()
    dim = int(mesh["rows"])
    elem_bytes = bits // 8
    return {
        "dim": dim,
        "elem_bytes": elem_bytes,
        "max_block_len": int(values["MAX_BYTES"]) // (dim * elem_bytes),
    }


def test_header_predicate_is_the_one_this_module_implements(header_text):
    """The device's own clause set, not a remembered one.

    A clause the header grew that this module does not check would let an ineligible layer stage a
    quarter of the input it needs -- a wrong output, not a slow one.
    """
    DS.predicate_is_implemented(header_text)


def test_the_library_states_the_predicate_more_than_once(header_text):
    """And not identically, which is why the contract is containment and not equality.

    The tile-search entry point omits the two clauses it has already pinned; taking either declaration
    as THE predicate would be wrong in one direction or the other.
    """
    declarations = DS.header_downsample_clauses(header_text)
    assert len(declarations) > 1
    assert len({frozenset(row) for row in declarations}) > 1
    assert set().union(*(set(row) for row in declarations)) == set(DS.IMPLEMENTED_CLAUSES)


def test_predicate_reads_structurally_not_by_spelling():
    """Whitespace and line wrapping are not part of the predicate."""
    text = "x;\nconst   bool downsample =\n   stride == 2\n   &&  kernel_dim == 1 ;\nrest"
    assert DS.header_downsample_clauses(text) == (("stride == 2", "kernel_dim == 1"),)


def test_a_missing_declaration_is_unknown_not_false():
    with pytest.raises(DS.DownsampleUnknown):
        DS.header_downsample_clauses("nothing about convolutions here")


def test_an_unrecognised_clause_is_unknown_not_ignored():
    text = "const bool downsample = stride == 2 && something_new;"
    with pytest.raises(DS.DownsampleUnknown):
        DS.predicate_is_implemented(text)


def test_the_three_projection_shortcuts_are_eligible(header_text):
    """ResNet-50's three 1x1 stride-2 downsamples, by their own geometry."""
    for in_dim in (56, 28, 14):
        assert (
            DS.downsample_flag(
                kernel=1,
                stride=2,
                padding=0,
                in_rows=in_dim,
                in_cols=in_dim,
                pooled=False,
                header_text=header_text,
            )
            == 1
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kernel=3, stride=2, padding=1, in_rows=14, in_cols=14, pooled=False),  # 3x3 stride 2
        dict(kernel=1, stride=1, padding=0, in_rows=28, in_cols=28, pooled=False),  # unit stride
        dict(kernel=1, stride=2, padding=0, in_rows=7, in_cols=7, pooled=False),  # odd extent
        dict(kernel=1, stride=2, padding=0, in_rows=56, in_cols=56, pooled=True),  # pooled readout
    ],
)
def test_everything_else_is_refused(kwargs, header_text):
    assert DS.downsample_flag(header_text=header_text, **kwargs) == 0


def test_the_mode_quarters_the_staged_window(facts):
    """At stride 2 in both dimensions the loader stages a quarter of the pixels.

    Stated over the tiling the schedule actually picks for the largest of the three shortcuts, so the
    number is the one that layer pays and not a limit it approaches.
    """
    tile = {"batches": 1, "porows": 7, "pocols": 17, "pochs": 32, "kchs": 256}
    waste = DS.staging_waste(
        batch=1,
        out_dim=28,
        out_channels=512,
        kernel=1,
        stride=2,
        in_channels=256,
        tile=tile,
        facts=facts,
        downsample=1,
    )
    assert waste["unstrided"] == 4 * waste["strided"]
    assert waste["discarded"] == 3 * waste["strided"]


def test_an_ineligible_layer_pays_nothing_and_saves_nothing(facts):
    tile = {"batches": 1, "porows": 22, "pocols": 23, "pochs": 16, "kchs": 128}
    waste = DS.staging_waste(
        batch=1,
        out_dim=28,
        out_channels=128,
        kernel=3,
        stride=1,
        in_channels=128,
        tile=tile,
        facts=facts,
        downsample=0,
    )
    assert waste["discarded"] == 0
    assert waste["unstrided"] == waste["strided"]


def test_working_rows_quarters_with_the_mode(facts):
    """The capacity check is the reason the tile is small; it must see the same quarter.

    A capacity model that prices the unstrided window refuses output tiles the device would hold, and
    the smaller tile it settles on is what re-reads the input once per output-channel tile.
    """
    shape = dict(
        stride=2,
        batches=1,
        porows=7,
        pocols=17,
        pochs=32,
        krows=1,
        kcols=1,
        kchs=256,
        pool_size=1,
        pool_stride=1,
        dim=facts["dim"],
    )
    off = DS.working_rows(acc=False, downsample=0, **shape)
    on = DS.working_rows(acc=False, downsample=1, **shape)
    weights = 2 * 1 * 1 * 256  # ceil(pochs/dim) * kcols * krows * kchs, unchanged by the mode
    assert (off - weights) == 4 * (on - weights)
    # The accumulator holds outputs, which the mode does not touch.
    assert DS.working_rows(acc=True, downsample=0, **shape) == DS.working_rows(acc=True, downsample=1, **shape)


def test_the_two_descriptor_fields_move_together():
    """`downsample` without `A_stride` is a different convolution, so they are one answer."""
    assert DS.descriptor_overrides(1, stride=2) == {"downsample": 1, "A_stride": 1}
    assert DS.descriptor_overrides(0, stride=2) == {"downsample": 0, "A_stride": 2}


def test_eligible_layers_selects_only_the_shortcuts(header_text):
    layers = [
        {"name": "a", "kernel": 1, "stride": 2, "padding": 0, "in_rows": 56, "in_cols": 56},
        {"name": "b", "kernel": 3, "stride": 1, "padding": 1, "in_rows": 28, "in_cols": 28},
        {"name": "c", "kernel": 1, "stride": 2, "padding": 0, "in_rows": 14, "in_cols": 14},
    ]
    picked = DS.eligible_layers(layers, header_text=header_text)
    assert [row["name"] for row in picked] == ["a", "c"]
    assert all(row["downsample"] == 1 for row in picked)
