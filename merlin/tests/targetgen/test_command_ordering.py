"""Command-ordering facts are OBSERVED on a dispatch unit, never assumed.

Three levels: the cycle evaluator on a small synthetic HW module; the derivation driving a behavioural
dispatch model whose queues, release points and span formulas the test controls (so every derived
fact is checked against a known answer); and, when the pinned elaboration is on this machine, the
derivation on the real dispatch module.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from merlin.targetgen.rtl import command_ordering as CO
from merlin.targetgen.rtl.hw_cycle_eval import CycleModel
from merlin.targetgen.rtl.hw_design import load_design


# ---------------------------------------------------------------------- the cycle evaluator
def _module() -> str:
    ports = "input clock : !seq.clock, input reset : i1, input en : i1, input d : i8, input sel : i1, output q : i8, output s : i8, output pick : i8"
    body = [
        '%zero = "hw.constant"() {value = 0 : i8} : () -> i8',
        '%seven = "hw.constant"() {value = 7 : i8} : () -> i8',
        # graph region: the mux that feeds the register is defined after its use
        '%r = "seq.firreg"(%n, %clock, %reset, %zero) <{name = "r"}> : (i8, !seq.clock, i1, i8) -> i8',
        '%sum = "comb.add"(%r, %d) : (i8, i8) -> i8',
        '%n = "comb.mux"(%en, %sum, %r) : (i1, i8, i8) -> i8',
        '%arr = "hw.array_create"(%seven, %r) : (i8, i8) -> !hw.array<2xi8>',
        '%pick = "hw.array_get"(%arr, %sel) : (!hw.array<2xi8>, i1) -> i8',
    ]
    return (
        '"builtin.module"() ({\n  "hw.module"() ({\n'
        "  ^bb0(%clock: !seq.clock, %reset: i1, %en: i1, %d: i8, %sel: i1):\n"
        + "".join(f"    {line}\n" for line in body)
        + '    "hw.output"(%r, %sum, %pick) : (i8, i8, i8) -> ()\n'
        f'  }}) {{module_type = !hw.modty<{ports}>, parameters = [], sym_name = "Acc"}} : () -> ()\n'
        "}) : () -> ()\n"
    )


def test_cycle_model_registers_reset_mux_and_arrays():
    model = CycleModel(load_design(_module()), "Acc")
    assert model.step({"reset": 1})["q"] == 0
    assert model.step({"en": 1, "d": 5})["q"] == 0  # outputs are seen before the edge
    assert model.step({"en": 1, "d": 250})["q"] == 5
    out = model.peek({"sel": 0})
    assert out["q"] == 255 and out["pick"] == 255  # element 0 is the LAST array_create operand
    assert model.peek({"sel": 1})["pick"] == 7
    model.step({"reset": 1})
    assert model.peek({})["q"] == 0


# ---------------------------------------------------------------------- a behavioural dispatch unit
class FakeDispatch:
    """Load queue releases at issue, store queue at completion; cross-queue overlaps with a writer wait
    for completion (start-inclusive ranges); configurations complete at issue; a load's span is
    ``(ceil(cols/16) - 1) * stride[state] + rows`` with the stride latched per load state."""

    inputs = ["reset", "a_valid", "a_funct", "a_rs1", "a_rs2", "ld_ready", "st_ready", "ex_ready", "c_valid", "c_id"]
    outputs = ["a_ready"] + [f"{q}_{p}" for q in ("ld", "st", "ex") for p in ("valid", "id", "funct", "rs1", "rs2")]
    unreset_registers = 0

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self._s = {"entries": [], "next": 0, "strides": [0, 0, 0, 0]}

    @property
    def state(self):
        return [copy.deepcopy(self._s)]

    @state.setter
    def state(self, value):
        self._s = copy.deepcopy(value[0])

    @staticmethod
    def queue(funct, rs1):
        if funct in (1, 2):
            return "ld"
        if funct == 3:
            return "st"
        return {0: "ex", 1: "ld"}.get(rs1 & 3, "st")

    def span(self, funct, rs2):
        if funct not in (1, 2, 3):
            return None
        rows, cols, addr = (rs2 >> 48) & 0xFFFF, (rs2 >> 32) & 0xFFFF, rs2 & 0xFFFF_FFFF
        blocks = max(1, -(-cols // 16))
        stride = 16 if funct == 3 else self._s["strides"][0 if funct == 2 else 1]
        start = addr & 0x3FFF
        return (addr >> 31, start, start + (blocks - 1) * stride + rows, funct != 3)

    @staticmethod
    def overlap(a, b):
        if a is None or b is None or a[0] != b[0] or not (a[3] or b[3]):
            return False
        return (b[1] <= a[1] < b[2]) or (a[1] <= b[1] < a[2])

    def _ready_entry(self, q):
        live = {e["id"]: e for e in self._s["entries"]}
        for e in self._s["entries"]:
            if e["q"] != q or e["issued"]:
                continue
            ok = all(i not in live for i in e["deps"]) and all(
                live.get(i, {"issued": True})["issued"] for i in e["issue_deps"]
            )
            return e if ok else None
        return None

    def step(self, inputs):
        get = lambda k: inputs.get(k, 0)  # noqa: E731
        out = {name: 0 for name in self.outputs}
        heads = {q: self._ready_entry(q) for q in ("ld", "st", "ex")}
        for q, e in heads.items():
            if e is not None:
                out.update(
                    {
                        f"{q}_valid": 1,
                        f"{q}_id": e["id"],
                        f"{q}_funct": e["funct"],
                        f"{q}_rs1": e["rs1"],
                        f"{q}_rs2": e["rs2"],
                    }
                )
        out["a_ready"] = get("a_valid")
        if get("reset"):
            self.reset_state()
            return out
        for q, e in heads.items():
            if e is not None and get(f"{q}_ready"):
                e["issued"] = True
                if e["funct"] == 0:
                    self._s["entries"].remove(e)
        if get("c_valid"):
            self._s["entries"] = [e for e in self._s["entries"] if e["id"] != get("c_id")]
        if get("a_valid"):
            funct, rs1, rs2 = get("a_funct"), get("a_rs1"), get("a_rs2")
            q = self.queue(funct, rs1)
            if funct == 0 and rs1 & 3 == 1:
                self._s["strides"][(rs1 >> 3) & 3] = (rs1 >> 16) & 0xFFFF
            mine = self.span(funct, rs2)
            entry = {
                "id": self._s["next"],
                "q": q,
                "funct": funct,
                "rs1": rs1,
                "rs2": rs2,
                "issued": False,
                "span": mine,
                "deps": [],
                "issue_deps": [],
            }
            for e in self._s["entries"]:
                if e["q"] == q:
                    (entry["deps"] if q == "st" else entry["issue_deps"]).append(e["id"])
                elif self.overlap(mine, e["span"]):
                    entry["deps"].append(e["id"])
            self._s["next"] += 1
            self._s["entries"].append(entry)
        return out


CONTRACT = {
    "encoding": {
        "semantic_class": {0: "CONFIG", 1: "MVIN2", 2: "MVIN", 3: "MVOUT"},
        "config_subtype": {0: "CONFIG_EX", 1: "CONFIG_LD", 2: "CONFIG_ST"},
        "addr_len": 32,
    },
    "rocc_operand_roles": {
        "local_address_flags": {"accumulator_select": {"msb_offset": 0}},
        "config": {"class": "CONFIG", "selector": {"operand": "rs1", "bundle": "CfgRs1", "field": "cmd_type"}},
        "classes": {
            "MVIN": {"fields": [{"name": "spad_addr", "operand": "rs2", "bundle": "MvRs2", "field": "local_addr"}]},
            "MVIN2": {"fields": [{"name": "spad_addr", "operand": "rs2", "bundle": "MvRs2", "field": "local_addr"}]},
            "MVOUT": {"fields": [{"name": "acc_addr", "operand": "rs2", "bundle": "MvRs2", "field": "local_addr"}]},
        },
    },
    "rtl_checks": {
        "class_roles": {"MVIN": ["movement_in"], "MVIN2": ["movement_in"], "MVOUT": ["movement_out"]},
        "local_addresses": [
            {"role": "movement_in", "address": "spad_addr", "rows": "rows"},
            {"role": "movement_out", "address": "acc_addr"},
        ],
    },
}
DISPATCH = {
    "module": "FakeDispatch",
    "classes": {"load": "ld_", "store": "st_", "execute": "ex_"},
    "ordering": {
        "reset": "reset",
        "allocate": {"valid": "a_valid", "ready": "a_ready", "funct": "a_funct", "rs1": "a_rs1", "rs2": "a_rs2"},
        "issue": {"valid": "valid", "ready": "ready", "id": "id", "funct": "funct", "rs1": "rs1", "rs2": "rs2"},
        "completion": {"valid": "c_valid", "id": "c_id"},
        "config_fields": [{"bundle": "CfgRs1", "operand": "rs1"}],
    },
}
BODY = {
    "interfaces": [
        {"name": "funct_decode_table", "legal_funct": [0, 1, 2, 3], "width": 7},
        {
            "name": "register_bundle_layouts",
            "bundles": {
                "MvRs2": {
                    "fields": {
                        "local_addr": {"offset": 0, "width": 32},
                        "num_cols": {"offset": 32, "width": 16},
                        "num_rows": {"offset": 48, "width": 16},
                    }
                },
                "CfgRs1": {
                    "fields": {
                        "cmd_type": {"offset": 0, "width": 2},
                        "state_id": {"offset": 3, "width": 2},
                        "stride": {"offset": 16, "width": 16},
                    }
                },
            },
        },
    ],
    "timing_structure": {
        "command_classes": {
            "classes": {
                "load": {"downstream_buffers": [{"instance": "U/reader/tracker", "entries": [16]}]},
                "store": {"downstream_buffers": [{"instance": "U/writeq", "entries": [2]}]},
            }
        },
        "memory_clients": {
            "clients": {"dma_read": {"instance": "U/reader", "id_space": 16, "in_flight": {"value": 16}}}
        },
    },
}


@pytest.fixture(scope="module")
def derived():
    view = CO.isa_view(CONTRACT, BODY, DISPATCH)
    deriver = CO.Deriver(CO.Dispatch(FakeDispatch(), CO._ports(DISPATCH)), view)
    queues = deriver.funct_queue()
    same = deriver.same_queue()
    cross = deriver.cross_queue()
    return {
        "queues": queues,
        "same": same,
        "cross": cross,
        "extents": deriver.extents(cross),
        "order": CO.completion_order(BODY["timing_structure"], same),
        "unknown": deriver.unknown,
    }


def test_queue_of_every_funct_and_configuration_code(derived):
    rows = {row["funct"]: row for row in derived["queues"]}
    assert rows[2]["queue"] == "load" and rows[1]["queue"] == "load" and rows[3]["queue"] == "store"
    assert rows[0]["by_code"] == {"0": "execute", "1": "load", "2": "store", "3": "store"}


def test_release_points_are_observed(derived):
    assert derived["same"]["load"]["release"] == "issue"
    assert derived["same"]["store"]["release"] == "completion"
    assert derived["same"]["execute"]["release"] == "unknown"  # no local operand enters it: said, not assumed


def test_cross_queue_waits_only_when_overlapping(derived):
    rows = {(r["older"], r["younger"], r["space"]): r for r in derived["cross"]}
    load_then_store = rows[("MVIN.spad_addr", "MVOUT.acc_addr", "scratchpad")]
    assert load_then_store["overlap"] == "completion" and load_then_store["tracked"] is True
    assert rows[("MVOUT.acc_addr", "MVIN.spad_addr", "accumulator")]["tracked"] is True


def test_span_formulas_with_state_keyed_block_stride(derived):
    mvin = derived["extents"]["MVIN.spad_addr"]
    assert mvin["status"] == "derived" and mvin["rows"]["name"] == "num_rows" and mvin["scale"] is None
    stride = mvin["blocks"]["stride"]
    assert mvin["blocks"]["block"] == 16 and (stride["offset"], stride["width"]) == (16, 16)
    assert stride["state"]["offset"] == 3 and stride["state"]["value"] == 0
    assert derived["extents"]["MVIN2.spad_addr"]["blocks"]["stride"]["state"]["value"] == 1
    assert derived["extents"]["MVOUT.acc_addr"]["blocks"]["stride"] == {"constant": 16}


def test_completion_order_from_the_memory_client_facts(derived):
    assert derived["order"]["load"]["order"] == "out_of_order"
    assert derived["order"]["store"]["order"] == "in_order"
    missing = CO.completion_order({}, {"load": {"release": "issue"}})
    assert missing["load"]["order"] == "unknown"


def test_a_dispatch_without_the_declared_ports_is_refused():
    broken = copy.deepcopy(DISPATCH)
    broken["ordering"]["completion"]["id"] = "no_such_port"
    with pytest.raises(ValueError, match="no port"):
        CO.Dispatch(FakeDispatch(), CO._ports(broken))
    del broken["ordering"]["allocate"]["rs2"]
    with pytest.raises(ValueError, match="allocate"):
        CO._ports(broken)


# ---------------------------------------------------------------------- the pinned elaboration
_HW = Path(
    "/work/merlin/tasks/phase1-harness-v2/merlin/out/artifacts/probes/gemmini/fire-aligned-30mhz/"
    "firesim-target/source-bundle/core.hw.generic.mlir"
)


@pytest.mark.skipif(not _HW.is_file(), reason="the pinned Gemmini elaboration is not on this machine")
def test_real_dispatch_module_releases_loads_at_issue():
    import json

    import yaml

    from merlin.common.paths import merlin_dir, repo_root

    contract = yaml.safe_load(
        (Path(repo_root()) / "examples/gemmini/target/contracts/target_contract.yaml").read_text()
    )
    dispatch = contract["rtl_extraction"]["timing_structure"]["dispatch"]
    recorded = json.loads((merlin_dir() / "tests/data/completion_order/gemmini_command_ordering.json").read_text())
    body = json.loads((merlin_dir() / "tests/data/completion_order/gemmini_interfaces.json").read_text())
    view = CO.isa_view(contract, body, dispatch)
    model = CycleModel(load_design(_HW.read_text()), dispatch["module"])
    deriver = CO.Deriver(CO.Dispatch(model, CO._ports(dispatch)), view)
    queues = {row["funct"]: row for row in deriver.funct_queue()}
    assert queues[2]["queue"] == "load" and queues[3]["queue"] == "store"
    same = deriver.same_queue()
    assert same["load"]["release"] == "issue" and same["store"]["release"] == "completion"
    assert same == recorded["same_queue"]
