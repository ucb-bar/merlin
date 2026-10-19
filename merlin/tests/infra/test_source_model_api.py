"""Public saved-model execution discovers shared dense rules and judges all state.

Neutral primitives establish this compiler/execution interface, not accelerator
support, arbitrary machine-code equivalence, or latency. Actual native and Spike
primitive arguments are independently interpreted by the existing private fixture.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_dense_ordinary_model import capture, observe_console
from test_dense_source_catalog import compilation_memory
from test_dense_source_provider import DenseProvider
from test_selected_dense_contracts import inputs as dense_inputs

from merlin.common.digest import sha256_file
from merlin.compile.baremetal_model import BaremetalModelError, compile_saved_model
from merlin.model_check import full_build as FB
from merlin.runtime.backends import spike, spike_model
from merlin.targetgen.elf_lanes import read_program_section
from merlin.targetgen.rtl.facts import observed_facts

OUTPUT_NAMES = ("history", "state", "position", "input", "weight0", "weight1")


def setup(tmp_path, monkeypatch, shape, tile, width, shared, *, unavailable=False):
    from merlin.llvmlower import toolchain

    if not toolchain.available() or not spike.available():
        pytest.skip("native lowering and functional simulator unavailable")
    facts = tmp_path / "capabilities"
    facts.mkdir()

    def declare(facts, software, dense):
        facts["datapaths"] = [
            dict(name="input", dtype="int8" if width == 1 else "int16"),
            dict(name="accumulator", dtype="int32"),
        ]
        if unavailable:
            dense["support"]["partial_reduction_tiles"] = False

    model = dense_inputs(facts, tile=tile, widths=(width, width, 4), shared=shared, shape=shape, mutate=declare)
    bundle = tmp_path / "capture"
    expected, trace = capture(bundle, shape, width)
    np.savez(bundle / "goldens.npz", **dict(zip(OUTPUT_NAMES, expected, strict=True)))
    (bundle / "output_order.json").write_text(json.dumps(OUTPUT_NAMES))
    # Seal after all capture files, including every carried-state reference.
    artifacts = {p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p)) for p in bundle.iterdir()}
    (bundle / "capture_receipt.json").write_text(
        json.dumps(dict(schema="m2m.capture-receipt.v1", materialized_abi=dict(complete=True), artifacts=artifacts))
    )
    host = tmp_path / "host"
    host.mkdir()
    (host / "manifest.yaml").write_text(
        yaml.safe_dump(
            dict(
                target="host", run_id="shared-source-api", family="scalar_linalg",
                authoring=dict(mode="deterministic_generated_from_spec", generated_by_agent=False),
                outputs=dict(knobs="knobs.yaml"),
            )
        )
    )
    (host / "knobs.yaml").write_text(
        yaml.safe_dump(
            dict(
                backend="scalar", dtype_strategy="fp32",
                cflags=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
            )
        )
    )
    dts = tmp_path / "host.dts"
    dts.write_text('cpu@0 { device_type = "cpu"; riscv,isa = "rv64imafdc_zicsr_zifencei"; };\n')
    catalog = tmp_path / "boards.yaml"
    catalog.write_text(
        yaml.safe_dump(
            dict(schema_version=1, boards=dict(selected=dict(
                target=model[0].target, dram_bytes=64 << 20, dram_base=spike_model.DRAM_BASE,
                harts=1, console="htif", flow="baremetal", code_reserve=16 << 20,
                host_dts_sha256=sha256_file(dts),
            )))
        )
    )
    root = tmp_path / "out"
    root.mkdir()
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    options = dict(
        capture=bundle, package=host, board_catalog=catalog, board="selected", dts=dts,
        target=model[0].target, run="spike", arena_mb=1, timeout_s=30, readback="full",
        respect_captured_quantization_scope=True, host_llvm_bitcode=True,
    )
    return model, expected, trace, root, options


def execute(options, output, provider, expected):
    selected = provider.selected
    started = time.perf_counter()
    with observed_facts(selected.target, json.loads(selected.facts_bytes)), compilation_memory() as memory:
        receipt = compile_saved_model(**options, output=output, shared_source_lowerings=(provider.selection(),))
    elapsed = time.perf_counter() - started
    assert receipt["status"] == "verified_complete_output"
    assert receipt["inputs"]["shared_source_lowerings"] == ["SourceDenseLowering"]
    result = receipt["output"]
    assert result["readback"] == "full"
    judged = result["per_output"]
    assert [row["name"] for row in judged] == list(OUTPUT_NAMES)
    for row, reference in zip(judged, expected, strict=True):
        assert row["status"] == "compared" and row["passed"] is True
        assert row["mismatched_elements"] == 0 and row["of"] == reference.size
    console_path = Path(result["console"])
    assert result["console_sha256"] == sha256_file(console_path)
    console = console_path.read_bytes()
    actual, metadata = FB.read(console, options["capture"], result["build_hash"])
    assert len(actual) == len(expected) == 6
    with np.load(result["outputs_npz"]) as saved:
        assert set(saved.files) == {"out" + str(i) for i in range(6)}
        for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
            np.testing.assert_array_equal(got, want)
            np.testing.assert_array_equal(saved["out" + str(i)], want)
    preparation = result["shared_source_preparation"]
    assert preparation["sha256"] == sha256_file(Path(preparation["path"]))
    prepared = json.loads(Path(preparation["path"]).read_text())
    for prefix in ("input_snapshot", "source", "selected"):
        assert prepared[prefix + "_sha256"] == sha256_file(Path(prepared[prefix + "_path"]))
    source_path = Path(preparation["path"]).parent / "source_dense.json"
    source = json.loads(source_path.read_text())
    assert source["source_sha256"] == prepared["input_snapshot_sha256"]
    assert source["selected_sha256"] == prepared["selected_sha256"]
    elf = Path(result["elf"])
    assert result["elf_sha256"] == sha256_file(elf)
    observation = dict(
        receipt=receipt, source=source, compile_and_execute_seconds=elapsed,
        compilation_and_execution_memory=memory, elf_bytes=elf.stat().st_size,
        model_object_sha256=sha256_file(elf.parent / "model.o"),
        functional_forward_instructions=metadata["metrics"]["cycles"],
        scope="neutral host primitives; instrumented functional execution is not accelerator latency",
    )
    return observation, actual, console


@pytest.mark.parametrize("shape,tile,width,shared", [((3, 7, 5), (2, 3, 4), 1, False), ((4, 9, 7), (3, 2, 5), 2, True)])
def test_public_saved_model_runs_shared_dense_rules_and_complete_judging(
    tmp_path, monkeypatch, record_property, shape, tile, width, shared
):
    model, expected, trace, root, options = setup(tmp_path, monkeypatch, shape, tile, width, shared)
    observations, outputs = [], []
    for name, streaming in (("baseline", True), ("resource_chosen", False)):
        provider = DenseProvider(tmp_path / (name + "_provider"), model, trace, streaming=streaming)
        observation, actual, console = execute(options, root / name, provider, expected)
        witness = json.loads((provider.work / "native_qualification.json").read_text())
        observation["actual_calls"] = observe_console(console, provider, witness)
        observation["native_qualification"] = witness
        host = observation["receipt"]["output"]["host_provider"]
        assert host["status"] == "completed" and len(host["objects"]) == 2
        assert json.loads(Path(host["receipt_path"]).read_text()) == host
        for pin in (*host["source_model"], *(r[k] for r in host["objects"] for k in ("object", "llvm"))):
            assert pin["sha256"] == sha256_file(Path(pin["path"]))
        sections = []
        for item in provider.records:
            code, header = read_program_section(
                Path(observation["receipt"]["output"]["elf"]), item["section"], max_bytes=64 << 10
            )
            assert not header["writable"]
            sections.append(dict(**header, sha256=hashlib.sha256(code).hexdigest()))
            item["result"].revalidate()
        observation["actual_kernel_sections"] = sections
        observations.append(observation)
        outputs.append(actual)
    for before, after in zip(*outputs, strict=True):
        np.testing.assert_array_equal(before, after)
    before, after = observations
    assert before["model_object_sha256"] == after["model_object_sha256"]
    assert before["source"]["source_sha256"] == after["source"]["source_sha256"]
    assert before["source"]["selected_sha256"] == after["source"]["selected_sha256"]
    assert before["receipt"]["inputs"]["capture_tree"] == after["receipt"]["inputs"]["capture_tree"]
    for b, a in zip(before["source"]["bindings"], after["source"]["bindings"], strict=True):
        bp, ap = b["schedule"]["pipeline"], a["schedule"]["pipeline"]
        assert bp["library_sources_sha256"] == ap["library_sources_sha256"]
        assert bp["selected_contracts"] == ap["selected_contracts"]
        for key in ("hardware", "execution", "library", "numerical_policy"):
            assert bp["context"][key] == ap["context"][key]
    assert sum(c["effects"]["requested_read_bytes"] for c in after["actual_calls"]) < sum(
        c["effects"]["requested_read_bytes"] for c in before["actual_calls"]
    )
    (root / "pair.json").write_text(json.dumps(observations, indent=2))
    record_property("public_saved_model_dense_pair", json.dumps(observations))


def test_public_saved_model_falls_back_when_reduction_tails_are_unavailable(tmp_path, monkeypatch):
    model, expected, trace, root, options = setup(
        tmp_path, monkeypatch, (3, 7, 5), (2, 3, 4), 1, False, unavailable=True
    )
    provider = DenseProvider(tmp_path / "provider", model, trace)
    observation, _, console = execute(options, root / "fallback", provider, expected)
    assert not provider.records and b"MERLIN_DENSE_" not in console
    assert not (provider.work / "native_qualification.json").exists()
    assert "host_provider" not in observation["receipt"]["output"]
    source = observation["source"]
    assert source["bindings"] == [] and len(source["declined"]) == 2
    assert all("reduction" in reason for reason in source["declined"])
    assert source["source_sha256"] == source["selected_sha256"]


@pytest.mark.parametrize(
    "mutation,diagnostic",
    [
        ("missing", "shared source replacement lost its completed provider record"),
        ("other_build", "shared source provider belongs to a different ELF or prepared model"),
    ],
)
def test_public_saved_model_requires_its_actual_completed_provider_before_simulation(
    tmp_path, monkeypatch, mutation, diagnostic
):
    from merlin.runtime.host_provider import revalidate_host_provider

    model, _, trace, root, options = setup(tmp_path, monkeypatch, (3, 7, 5), (2, 3, 4), 1, False)
    simulator_calls = []

    def forbidden_simulator(*args, **kwargs):
        simulator_calls.append((args, kwargs))
        raise AssertionError("invalid provider custody reached the simulator")

    monkeypatch.setattr(spike_model, "run_raw", forbidden_simulator)
    donor = None
    if mutation == "other_build":
        provider = DenseProvider(tmp_path / "donor_provider", model, trace, streaming=True)
        with observed_facts(model[0].target, json.loads(model[0].facts_bytes)):
            receipt = compile_saved_model(
                **dict(options, run="none"), output=root / "donor", shared_source_lowerings=(provider.selection(),)
            )
        assert receipt["status"] == "compiled"
        donor = receipt["output"]["host_provider"]
        revalidate_host_provider(donor)

    actual_build = spike_model.build
    built_images = []

    def corrupt_producer_result(*args, **kwargs):
        built = actual_build(*args, **kwargs)
        elf = Path(built["elf"])
        preparation = json.loads((elf.parent / "prepared_model_transform/prepared_model_transform.json").read_text())
        assert preparation["selected_sha256"] != preparation["input_snapshot_sha256"]
        revalidate_host_provider(built["host_provider"])
        built_images.append(elf)
        if donor is None:
            del built["host_provider"]
        else:
            # The substituted record is internally valid and all of its files
            # remain unchanged. Its association with this ELF is the violation.
            revalidate_host_provider(donor)
            assert donor["executable"] != built["host_provider"]["executable"]
            built["host_provider"] = donor
        return built

    monkeypatch.setattr(spike_model, "build", corrupt_producer_result)
    provider = DenseProvider(tmp_path / "current_provider", model, trace)
    output = root / "invalid"
    with observed_facts(model[0].target, json.loads(model[0].facts_bytes)):
        with pytest.raises(BaremetalModelError, match=diagnostic):
            compile_saved_model(**options, output=output, shared_source_lowerings=(provider.selection(),))
    assert len(built_images) == 1 and built_images[0].is_file()
    assert not simulator_calls
    failed = json.loads((output / "baremetal_model.json").read_text())
    assert failed["status"] == "failed" and diagnostic in failed["failure"]
    assert not (output / "console.bin").exists() and not (output / "outputs.npz").exists()
    if donor is not None:
        revalidate_host_provider(donor)
