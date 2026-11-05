"""Frozen-panel custody/measurement checks; no cloud access or FPGA is used."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from merlin.common.tree_hash import hash_tree


def module():
    path = Path(__file__).with_name("physical_panel.py")
    spec = importlib.util.spec_from_file_location("physical_panel_under_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(tmp_path, *, names=("baseline", "candidate"), count=2):
    api = module()
    def dump(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return api.pin(path)
    def artifact(name, data=b"artifact"):
        path = tmp_path / name
        path.write_bytes(data)
        return api.pin(path)
    subjects = []
    for name in names:
        source = tmp_path / name
        source.mkdir()
        (source / "compiler.py").write_text(name)
        subjects.append({"id": name, "role": "initial_baseline" if name == "baseline" else "final_compiler",
                         "source": {"path": str(source), **hash_tree(source)},
                         "freeze": {"requirements_complete": True, "frozen_unix": 1,
                                    "rules": artifact(name + "-rules.txt"), "settings": artifact(name + "-settings.json")}})
    cases, compiled = [], []
    cb = {"tensors": {"A": {"shape": [1, 2], "dtype": "i8", "role": "input"},
                      "W": {"shape": [2, 1], "dtype": "i8", "role": "weight"},
                      "Y": {"shape": [1, 1], "dtype": "i32", "role": "output"}},
          "kernel_abi": {"args": [{"tensor": x} for x in ["W", "Y", "A"]]}}
    cb_pin = dump("cb.json", cb)
    obj = artifact("kernel.o")
    for i in range(count):
        name = "case" + str(i)
        ip, op = tmp_path / (name + "-inputs.npz"), tmp_path / (name + "-outputs.npz")
        np.savez(ip, A=np.array([[2, 3]], dtype=np.int8), W=np.array([[4], [5]], dtype=np.int8))
        np.savez(op, Y=np.array([[23]], dtype=np.int32))
        source = artifact(name + ".mlir", name.encode())
        cases.append({"case": name, "source": source, "fixtures": {"inputs": api.pin(ip), "outputs": api.pin(op)}})
        for subject in subjects:
            receipt = dump(subject["id"] + name + "-receipt.json", {
                "compiler_sha256": subject["source"]["sha256"], "case": name, "source_sha256": source["sha256"],
                "object_sha256": obj["sha256"], "command_buffer_sha256": cb_pin["sha256"]})
            compiled.append({"subject": subject["id"], "case": name,
                             "compiler_sha256": subject["source"]["sha256"], "object": obj,
                             "command_buffer": cb_pin, "build_receipt": receipt})
    plan_pin = dump("evaluation.json", {"subjects": subjects, "cases": cases})
    panel_pin = dump("panel.json", {"schema": "merlin.study.transfer_panel.v1", "cases": cases})
    tool = artifact("tool")
    header = artifact("runtime.h", b"/* callback header */")
    selection = {"schema": api.SCHEMA, "evaluation_plan": plan_pin, "panel": panel_pin,
                 "subjects": list(names), "snapshot": "final", "compiled": compiled,
                 "domain": {"agfi": "agfi-test", "target_clock_hz": 30000000,
                            "roi": "complete_call_including_completion_v1", "cache_policy": "sweep_not_cold",
                            "preparation_policy": "materialized_fixtures_then_poison_outputs",
                            "runtime_header_sha256": header["sha256"]},
                 "toolchain": {"compiler": tool, "nm": tool, "objcopy": tool,
                               "runtime_header": header, "linker_script": tool, "support_objects": [],
                               "cflags": []}}
    return api, selection, dump


def prepared_fixture(tmp_path):
    api, selection, dump = fixture(tmp_path)
    selection_pin = dump("selection.json", selection)
    prepared = api.prepare(selection_pin["path"], tmp_path / "private")
    return api, selection, dump, prepared


def observation_fixture(tmp_path, *, missing=False, mismatch=False, order_bad=False):
    api, selection, dump, prepared = prepared_fixture(tmp_path)
    elf = tmp_path / "panel.elf"
    elf.write_bytes(b"\x7fELF-test")
    build_pin = dump("build.json", {"prepared": api.pin(tmp_path / "private/prepared.json"), "elf": api.pin(elf)})
    boot = {"agfi": "agfi-test", "working_directory": "/run", "simulator_argv": ["driver", "+prog0=/run/panel.elf"],
            "consumed_elf": {"path": "/run/panel.elf", "sha256": api.pin(elf)["sha256"]},
            "ssm_command_id": "ssm-test"}
    script = tmp_path / "sim-run.sh"
    script.write_text("driver +prog0=/run/panel.elf\n")
    uart = tmp_path / "uartlog"
    rows = ["PANEL_BEGIN"]
    for slot in prepared["slots"]:
        if missing and slot["slot"] == 3:
            continue
        cycles = 110 if order_bad and slot["round"] == "order_control" else 100
        rows.append(f"PANEL {slot['slot']} {cycles} {1 if mismatch and slot['slot']==3 else 0}")
    rows.append("PANEL_DONE")
    uart.write_text("\n".join(rows))
    observation = {"schema": api.OBSERVATION, "domain_sha256": prepared["domain_sha256"],
                   "booted_elf": dump("boot.json", boot), "sim_run": api.pin(script),
                   "ssm_invocation": dump("ssm.json", {"CommandId": "ssm-test", "Status": "Success", "ResponseCode": 0}),
                   "uploaded_elf": api.pin(elf), "uart": api.pin(uart),
                   "runtime_domain": dump("domain.json", {"ssm_command_id": "ssm-test", "domain": prepared["domain"]})}
    return api, dump, prepared, build_pin, dump("observation.json", observation)


def test_full_declared_roster_three_balanced_rounds_and_repeat(tmp_path):
    api, selection, dump, prepared = prepared_fixture(tmp_path)
    assert len(prepared["slots"]) == 13
    assert prepared["slots"][-1]["repeats"] == 0
    for subject in selection["subjects"]:
        for case in ("case0", "case1"):
            assert [s["round"] for s in prepared["slots"][:-1] if s["subject"] == subject and s["case"] == case] == [0, 1, 2]
    assert prepared["paid_jobs_launched"] == 0


def test_render_exact_abi_poison_every_byte_compare_after_complete(tmp_path):
    api, selection, dump, prepared = prepared_fixture(tmp_path)
    path, blobs = api.render(prepared, tmp_path / "private")
    text = path.read_text()
    assert "K0_entry(panel_tensor_W, panel_tensor_Y, panel_tensor_A); K0_complete();" in text
    assert text.index("K0_complete();") < text.index("const uint64_t end") < text.index("mismatches +=")
    assert "(unsigned char)~_binary_panel_blob_" in text
    assert "for (size_t j=0;j<4;j++) mismatches +=" in text
    assert len(blobs) == 3  # shared across all subjects, cases and repeated rounds


@pytest.mark.parametrize("fault", ["source", "object", "receipt", "fixture", "domain", "abi"])
def test_preparation_rejects_stale_or_inconsistent_evidence(tmp_path, fault):
    api, selection, dump = fixture(tmp_path)
    row = selection["compiled"][0]
    if fault == "source":
        (tmp_path / "baseline/compiler.py").write_text("changed")
    elif fault == "object":
        Path(row["object"]["path"]).write_bytes(b"changed")
    elif fault == "receipt":
        value = api.read(row["build_receipt"])
        value["compiler_sha256"] = "wrong"
        row["build_receipt"] = dump("bad-receipt.json", value)
    elif fault == "fixture":
        Path(selection["panel"]["path"]).write_text("{}")
    elif fault == "domain":
        selection["domain"]["runtime_header_sha256"] = "wrong"
    else:
        value = api.read(row["command_buffer"])
        value["kernel_abi"]["args"][0]["tensor"] = "Y"
        row["command_buffer"] = dump("bad-cb.json", value)
        receipt = api.read(row["build_receipt"])
        receipt["command_buffer_sha256"] = row["command_buffer"]["sha256"]
        row["build_receipt"] = dump("bad-cb-receipt.json", receipt)
    with pytest.raises(ValueError):
        api.prepare(dump("selection.json", selection)["path"], tmp_path / "private")


def test_uncompiled_cells_remain_in_denominator(tmp_path):
    api, selection, dump = fixture(tmp_path)
    selection["compiled"].pop(1)
    prepared = api.prepare(dump("selection.json", selection)["path"], tmp_path / "private")
    assert len(prepared["cells"]) == 4
    assert any(c.get("state") == "missing" for c in prepared["cells"])


def test_successful_observation_requires_every_byte_and_exact_custody(tmp_path):
    api, dump, prepared, build_pin, observation_pin = observation_fixture(tmp_path)
    result = api.collect(build_pin["path"], observation_pin["path"])
    assert result["complete"] and all(c["state"] == "pass" for c in result["cells"])
    assert all(c["ordered_samples"] == [100, 100, 100] for c in result["cells"])
    assert result["scientific_qualification"] is False


@pytest.mark.parametrize("fault", ["missing", "mismatch", "order_bad"])
def test_physical_missing_wrong_and_position_effect_never_rank_as_pass(tmp_path, fault):
    api, dump, prepared, build_pin, observation_pin = observation_fixture(tmp_path, **{fault: True})
    result = api.collect(build_pin["path"], observation_pin["path"])
    expected = {"missing": "missing", "mismatch": "fail", "order_bad": "unqualified"}[fault]
    assert any(c["state"] == expected and c["cycles_median"] is None for c in result["cells"])
    assert len(result["cells"]) == 4


@pytest.mark.parametrize("fault", ["elf", "command", "agfi", "runtime_domain", "uart_changed"])
def test_observation_refuses_unjoined_domain_boot_or_uart(tmp_path, fault):
    api, dump, prepared, build_pin, observation_pin = observation_fixture(tmp_path)
    observation = api.read(observation_pin)
    if fault == "uart_changed":
        Path(observation["uart"]["path"]).write_text("forged")
    else:
        key = "booted_elf" if fault in ("elf", "agfi", "command") else "runtime_domain"
        value = api.read(observation[key])
        if fault == "elf":
            value["consumed_elf"]["sha256"] = "wrong"
        elif fault == "command":
            value["ssm_command_id"] = "wrong"
        elif fault == "agfi":
            value["agfi"] = "wrong"
        else:
            value["domain"]["cache_policy"] = "different"
        observation[key] = dump("bad-domain.json", value)
        observation_pin = dump("bad-observation.json", observation)
    with pytest.raises(ValueError):
        api.collect(build_pin["path"], observation_pin["path"])


def test_physical_import_replays_custody_and_refuses_changed_source(tmp_path):
    api, dump, prepared, build_pin, observation_pin = observation_fixture(tmp_path)
    result = api.collect(build_pin["path"], observation_pin["path"])
    result_pin = dump("collected.json", result)
    plan = api.read(prepared["evaluation_plan"])
    subjects = {s["id"]: s for s in plan["subjects"]}
    collection = {"plan": prepared["evaluation_plan"], "cells": [
        {"subject": c["subject"], "case": c["case"], "source": subjects[c["subject"]]["source"],
         "state": "pass", "split": "heldout", "baseline_subject": "baseline"} for c in result["cells"]]}
    final = api.apply_physical(collection, [result_pin["path"]])
    assert final["physical_measurements_imported"] == 4
    assert all(c["physical_comparison"]["candidate_to_baseline_ratio"] == 1 for c in final["cells"] if c["subject"] == "candidate")
    result["cells"][0]["cycles_median"] = 1
    with pytest.raises(ValueError, match="replayed"):
        api.apply_physical(collection, [dump("forged.json", result)["path"]])


def test_legacy_runtime_emits_real_retained_htif_adapter(tmp_path):
    api = module()
    config = {}
    for key in ("timing_header", "accelerator_header", "common_c"):
        p = tmp_path / key
        p.write_text("/* pinned */")
        config[key] = api.pin(p)
    config["sweep_bytes"] = 2097152
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config))
    result = api.legacy_runtime(p, tmp_path / "runtime.h")
    text = Path(result["header"]["path"]).read_text()
    assert "console_init(); gemmini_flush(0);" in text
    assert "l2_sweep();" in text and "htif_exit(status);" in text
    assert result["cache_claim"] == "declared sweep; cold cache is not established"


def test_every_qk_output_is_checked(tmp_path):
    api, selection, dump = fixture(tmp_path, names=("candidate",), count=1)
    row = selection["compiled"][0]
    cb = api.read(row["command_buffer"])
    cb["tensors"].pop("Y")
    for i in range(32):
        cb["tensors"][f"S{i}"] = {"shape": [1, 1], "dtype": "i32", "role": "output"}
    cb["kernel_abi"]["args"] = [{"tensor": name} for name in cb["tensors"]]
    row["command_buffer"] = dump("qk-cb.json", cb)
    receipt = api.read(row["build_receipt"])
    receipt["command_buffer_sha256"] = row["command_buffer"]["sha256"]
    row["build_receipt"] = dump("qk-receipt.json", receipt)
    panel = api.read(selection["panel"])
    path = tmp_path / "qk-outputs.npz"
    np.savez(path, **{f"S{i}": np.array([[i]], dtype=np.int32) for i in range(32)})
    panel["cases"][0]["fixtures"]["outputs"] = api.pin(path)
    selection["panel"] = dump("qk-panel.json", panel)
    prepared = api.prepare(dump("selection.json", selection)["path"], tmp_path / "private")
    harness, _ = api.render(prepared, tmp_path / "private")
    assert harness.read_text().count("mismatches +=") == 32 * 4
