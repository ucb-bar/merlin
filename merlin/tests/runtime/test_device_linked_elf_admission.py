"""Actual ordinary build handoffs with substituted compiler/provider work.

These are source-only diagnostic transport controls. The owned artifact is not
an executable, and the selected marker policy is not an ISA/no-FSM issuer.
No model, target provider, compiler or simulator is launched.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.digest import sha256_file
from merlin.llvmlower.device_build import DeviceBuild, DeviceRouting
from merlin.llvmlower.device_offload import BY_GROUP, DeviceRewrite, Routed
from merlin.runtime.backends import spike_model
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService


class DiagnosticPolicy:
    def __init__(self, events, mode="accepted"):
        self.events, self.mode = events, mode

    def evaluate(self, *, elf, evidence_root):
        self.events.append(("selected audit", elf, elf.read_bytes()))
        evidence_root.mkdir(parents=True)
        report = evidence_root / "report.json"
        result = {"status": "refused" if self.mode == "refused" else "accepted", "elf_sha256": sha256_file(elf)}
        report.write_text(json.dumps(result))
        selected = {**result, "report_path": str(report), "report_sha256": sha256_file(report)}
        if self.mode == "missing_report":
            report.unlink()
        elif self.mode == "wrong_elf":
            selected["elf_sha256"] = "0" * 64
        elif self.mode == "changed_report":
            report.write_text("{}")
        elif self.mode == "changed_elf":
            elf.write_bytes(b"changed during selected evaluation")
        return selected

    def replacement(self, *, elf, evidence_root):
        raise AssertionError("replaced selected policy must never execute")


@pytest.fixture
def ordinary(tmp_path, monkeypatch):
    from merlin.llvmlower import device_build, qinner, weight_prepack
    from merlin.runtime import host_math

    events = []
    compiler, archive, source = (tmp_path / name for name in ("compiler", "libm.a", "model.ll"))
    compiler.write_text("owned diagnostic tool; never launched\n")
    compiler.chmod(0o700)
    linker = compiler.with_name("riscv64-unknown-elf-ld")
    linker.write_bytes(compiler.read_bytes())
    linker.chmod(0o700)
    archive.write_bytes(b"owned diagnostic archive")
    source.write_text("owned substituted lowering; not executable LLVM\n")
    runtime, harness, abi = (tmp_path / name for name in ("runtime", "harness", "abi"))
    for directory in (runtime, harness, abi):
        directory.mkdir()
    for directory, names in (
        (runtime, ("merlin_model.c",)),
        (harness, ("model_main.c", "crt.S", "htif.c", "libc_min.c", "merlin_malloc.c", "model_link.ld")),
        (abi, ("mlir_runtime.c",)),
    ):
        for name in names:
            (directory / name).write_text("owned diagnostic input; never compiled\n")
    layout = {"data_layout": "diagnostic-unavailable", "index_bits": 64, "effective_pipeline": "substituted"}
    observation = {**layout, "compiler_resolved": str(compiler), "compiler_sha256": sha256_file(compiler)}
    monkeypatch.setattr(weight_prepack, "prepare_build_bundle", lambda model, *_: model)
    monkeypatch.setattr(qinner, "plan_for_bundle", lambda *_: False)
    monkeypatch.setattr(spike_model._spike, "gcc_path", lambda: compiler)
    monkeypatch.setattr(spike_model, "_harness_dir", lambda: harness)
    monkeypatch.setattr(spike_model, "_c_runtime_dir", lambda: runtime)
    monkeypatch.setattr(spike_model, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(spike_model, "_mlir_runtime_compiler", lambda *_: [str(compiler)])
    monkeypatch.setattr(
        spike_model,
        "selected_model_compiler_plan",
        lambda **_: {"observation": observation, "gcc_cflags": [], "clang_cflags": [], "model_cflags": []},
    )
    monkeypatch.setattr(
        spike_model,
        "lower_model_file",
        lambda *_a, **_k: SimpleNamespace(ll_path=source, stats={"index_lowering": layout}),
    )
    monkeypatch.setattr(host_math, "build_host_math", lambda *_: ((), ()))
    monkeypatch.setattr(
        spike_model, "_selected_libm_archive", lambda *_: (archive, sha256_file(archive), sha256_file(compiler))
    )

    def generate(_model, output, *_args, **_kwargs):
        output.mkdir()
        (output / "weights.bin").write_bytes(b"owned diagnostic input")
        (output / "model_call.c").write_text("owned diagnostic call; never compiled\n")
        return {"out_dt": "i8", "weights_bytes": 1}

    def run(argv, **_kwargs):
        if "-o" in argv:
            output = Path(argv[argv.index("-o") + 1])
            output.write_bytes(
                b"owned diagnostic linked artifact" if output.name == "model.elf" else b"diagnostic object"
            )
            events.append(("link" if output.name == "model.elf" else "object", output))
        return subprocess.CompletedProcess(argv, 0, "substituted diagnostic compiler", "")

    def build_device(_target, signatures, *_args, workdir, **_kwargs):
        events.append(("device objects", tuple(signatures)))
        workdir.mkdir()
        obj, shim = workdir / "kernel.o", workdir / "shim.o"
        obj.write_bytes(b"owned diagnostic device object")
        shim.write_bytes(b"owned diagnostic shim")
        return DeviceBuild(
            device="fixture",
            kernels={symbol: symbol for symbol in signatures},
            objects=(obj, shim),
            shim_object=shim,
            built_from={symbol: "stated_group" for symbol in signatures},
            object_dedup={"unique_artifacts": 1},
        )

    monkeypatch.setattr(spike_model.c_runtime, "generate", generate)
    monkeypatch.setattr(spike_model, "_run", run)
    monkeypatch.setattr(device_build, "build_device_objects", build_device)

    def invoke(
        *, active=True, gate="accepted", legacy=None, selection_target="fixture", mutation=None, matrix_active=None
    ):
        work = tmp_path / "build"
        work.mkdir(exist_ok=True)
        if active:
            DeviceRewrite(
                device="fixture",
                granularity=BY_GROUP,
                signatures={"kernel": (3, 5, 2)},
                routed=(Routed("kernel", (3, 5), (2,), ("i8", "i8", "i8")),),
                entries={"kernel": {"op": "matmul"}},
            ).write_sidecar(work)
        service = (
            None
            if gate is None
            else LinkedElfAdmissionService(
                selection_target,
                DiagnosticPolicy(events, gate).evaluate,
                ((str(Path(__file__).resolve()), sha256_file(Path(__file__))),),
            )
        )
        if gate == "callback_only":

            def service():
                return None

        def legacy_audit(elf):
            if legacy is not None:
                legacy(elf)
            if mutation == "code":
                monkeypatch.setattr(service.evaluator.__func__, "__code__", DiagnosticPolicy.replacement.__code__)
            elif mutation == "owner":
                object.__setattr__(
                    route,
                    "linked_elf_admission",
                    LinkedElfAdmissionService("fixture", DiagnosticPolicy(events).evaluate, service.source_pins),
                )

        route = (
            DeviceRouting(
                "fixture",
                tmp_path / "unselected",
                "i8",
                "i8",
                granularity=BY_GROUP,
                final_elf_audit=legacy_audit if legacy is not None or mutation is not None else None,
                linked_elf_admission=service,
            )
            if matrix_active is None
            else None
        )
        matrix = None
        if matrix_active is not None:
            from merlin.runtime.backends import zephyr_model

            matrix = zephyr_model.MatrixRouting("fixture", "selected_unit", "selected_config")

            def matrix_object(*_a, **_k):
                events.append(("matrix object",))
                obj = tmp_path / "matrix.o"
                obj.write_bytes(b"owned substituted matrix object; never executable")
                return SimpleNamespace(
                    object_path=obj,
                    tile_edge=1,
                    scalar_tile=False,
                    scratch_bytes=0,
                    to_dict=lambda: {"scope": "substituted diagnostic object"},
                )

            provider = SimpleNamespace(build_object=matrix_object)
            monkeypatch.setattr(zephyr_model.MatrixRouting, "provider", lambda _: provider)
            monkeypatch.setattr(
                zephyr_model,
                "load_matrix_signatures",
                lambda _work, _matrix: {"matrix_kernel": (1, 1, 1)} if matrix_active else {},
            )
        result = spike_model.build(tmp_path / "source", work, backend="scalar", device=route, matrix=matrix)
        return result, service

    return invoke, events, tmp_path / "build"


def test_active_device_work_requires_policy_even_with_successful_legacy_callback(ordinary):
    invoke, events, work = ordinary
    with pytest.raises(ValueError, match="active device.*linked ELF admission"):
        invoke(gate=None, legacy=lambda elf: events.append(("legacy audit", elf)))
    assert not any(event[0] in {"device objects", "link", "legacy audit"} for event in events)
    assert json.loads((work / "compilation_recipe.json").read_text())["status"] != "completed"


def test_original_refusal_is_retained_without_completing_build(ordinary):
    invoke, events, work = ordinary
    with pytest.raises(ValueError, match="linked ELF.*refused"):
        invoke(gate="refused")
    assert any(event[0] == "selected audit" for event in events)
    assert json.loads((work / "compilation_recipe.json").read_text())["status"] != "completed"
    reports = list(work.rglob("report.json"))
    assert len(reports) == 1 and json.loads(reports[0].read_text())["status"] == "refused"


def test_selected_gate_observes_same_final_artifact_after_legacy_audit(ordinary):
    invoke, events, _work = ordinary
    result, service = invoke(legacy=lambda elf: events.append(("legacy audit", elf)))
    names = [event[0] for event in events]
    assert names.index("link") < names.index("legacy audit") < names.index("selected audit")
    observation = result["linked_elf_admission"]
    assert service.revalidate(elf=result["elf"], result=observation, target="fixture") == "accepted"
    assert observation["elf_sha256"] == sha256_file(result["elf"])


@pytest.mark.parametrize("mode", ["missing_report", "wrong_elf", "changed_report", "changed_elf"])
def test_changed_or_absent_selected_evidence_never_completes_build(ordinary, mode):
    invoke, _events, work = ordinary
    with pytest.raises(ValueError, match="linked ELF"):
        invoke(gate=mode)
    assert json.loads((work / "compilation_recipe.json").read_text())["status"] != "completed"


def test_legacy_audit_cannot_replace_the_actual_linked_image(ordinary):
    invoke, events, work = ordinary
    with pytest.raises(ValueError, match="linked ELF.*changed"):
        invoke(legacy=lambda elf: elf.write_bytes(b"legacy callback replaced linked image"))
    assert not any(event[0] == "selected audit" for event in events)
    assert json.loads((work / "compilation_recipe.json").read_text())["status"] != "completed"


def test_selected_target_mismatch_refuses_before_compiler_work(ordinary):
    invoke, events, _work = ordinary
    with pytest.raises(ValueError, match="exact target"):
        invoke(selection_target="different")
    assert events == []


def test_unrouted_host_work_does_not_need_device_policy(ordinary):
    invoke, events, work = ordinary
    result, _service = invoke(active=False, gate=None)
    assert not any(event[0] in {"device objects", "selected audit"} for event in events)
    assert "linked_elf_admission" not in result
    assert json.loads((work / "compilation_recipe.json").read_text())["status"] == "completed"


def test_legacy_callback_alone_is_not_an_admission_selection(ordinary):
    invoke, events, _work = ordinary
    with pytest.raises(ValueError, match="explicitly selected service"):
        invoke(gate="callback_only")
    assert events == []


def test_active_matrix_object_route_has_no_unqualified_link_bypass(ordinary):
    invoke, events, work = ordinary
    with pytest.raises(ValueError, match="active matrix route.*linked.ELF"):
        invoke(active=False, gate=None, matrix_active=True)
    assert not any(event[0] in {"matrix object", "link"} for event in events)
    assert json.loads((work / "compilation_recipe.json").read_text())["status"] != "completed"


def test_inert_matrix_route_preserves_host_build(ordinary):
    invoke, events, work = ordinary
    invoke(active=False, gate=None, matrix_active=False)
    assert not any(event[0] in {"matrix object", "selected audit"} for event in events)
    assert json.loads((work / "compilation_recipe.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("mutation", ["code", "owner"])
def test_selected_policy_cannot_drift_during_legacy_audit(ordinary, mutation):
    invoke, events, work = ordinary
    with pytest.raises(ValueError, match="linked ELF admission.*changed"):
        invoke(mutation=mutation)
    assert not any(event[0] == "selected audit" for event in events)
    assert json.loads((work / "compilation_recipe.json").read_text())["status"] != "completed"


def test_original_placement_routing_preserves_exact_selected_gate(tmp_path, monkeypatch):
    from merlin.llvmlower.device_build import routing_for_placement
    from merlin.system import place

    owner = Path(__file__).resolve()
    service = LinkedElfAdmissionService("fixture", DiagnosticPolicy([]).evaluate, ((str(owner), sha256_file(owner)),))
    placement = SimpleNamespace(
        placed=(SimpleNamespace(on_device=True, device="fixture", demand=SimpleNamespace(in_fmt="i8"), acc="i8"),)
    )
    monkeypatch.setattr(place, "device_selector", lambda *_: None)
    route = routing_for_placement(placement, "fixture", tmp_path, linked_elf_admission=service)
    assert route.linked_elf_admission is service


@pytest.fixture
def zephyr_preparation(tmp_path, monkeypatch):
    from merlin.llvmlower import weight_prepack
    from merlin.runtime import boards
    from merlin.runtime.backends import zephyr_model

    class LoweringReached(Exception):
        """Diagnostic boundary; no lowering or object compilation executes."""

    brd = SimpleNamespace(
        flow="zephyr",
        zephyr_default_ram_bytes=1,
        zephyr_link_limit_bytes=1,
        vlen=None,
        hart_ids_for=lambda _: (0,),
        name="fixture",
    )
    source = tmp_path / "prepared.mlir"
    source.write_text("owned diagnostic preparation; never lowered\n")
    monkeypatch.setattr(boards, "board", lambda *_a, **_k: brd)
    monkeypatch.setattr(zephyr_model, "build_available", lambda: True)
    monkeypatch.setattr(weight_prepack, "prepare_build_bundle", lambda model, *_: model)
    monkeypatch.setattr(zephyr_model._spike, "gcc_path", lambda: tmp_path / "not-launched")
    monkeypatch.setattr(zephyr_model.toolchain, "clang", lambda: tmp_path / "not-launched")

    def stop(*_a, **_k):
        raise LoweringReached

    monkeypatch.setattr(zephyr_model, "lower_model_file", stop)

    def invoke(*, active, selected, matrix_active=None):
        work = tmp_path / "work"
        route = DeviceRouting("fixture", tmp_path, "i8", "i8", select=lambda _: active) if selected else None
        matrix = None
        if matrix_active is not None:
            matrix = zephyr_model.MatrixRouting("fixture", "selected_unit", "selected_config")
            monkeypatch.setattr(zephyr_model.MatrixRouting, "provider", lambda _: SimpleNamespace())
            monkeypatch.setattr(
                zephyr_model,
                "load_matrix_signatures",
                lambda _work, _matrix: {"matrix_kernel": (1, 1, 1)} if matrix_active else {},
            )

        def prepare(_source, destination, **kwargs):
            assert kwargs["device"] is route
            DeviceRewrite(
                device="fixture",
                signatures={"kernel": (3, 5, 2)} if active else {},
                routed=(Routed("kernel", (3, 5), (2,), ("i8", "i8", "i8")),) if active else (),
            ).write_sidecar(destination)
            return source, frozenset()

        monkeypatch.setattr(zephyr_model, "prepare_for_lowering", prepare)
        return zephyr_model.build_app(tmp_path, work, board="fixture", backend="scalar", device=route, matrix=matrix)

    return invoke, LoweringReached, zephyr_model.ZephyrModelError


def test_zephyr_cannot_compile_device_calls_without_its_device_link_consumer(zephyr_preparation):
    invoke, _reached, error = zephyr_preparation
    with pytest.raises(error, match="active device route.*device-object/final-link"):
        invoke(active=True, selected=True)


@pytest.mark.parametrize("selected", [False, True])
def test_zephyr_host_or_inert_route_retains_lowering_boundary(zephyr_preparation, selected):
    invoke, reached, _error = zephyr_preparation
    with pytest.raises(reached):
        invoke(active=False, selected=selected)


def test_zephyr_active_matrix_route_cannot_compile_without_selected_final_policy(zephyr_preparation):
    invoke, _reached, error = zephyr_preparation
    with pytest.raises(error, match="active matrix route.*linked.ELF"):
        invoke(active=False, selected=False, matrix_active=True)


def test_zephyr_inert_matrix_route_retains_lowering_boundary(zephyr_preparation):
    invoke, reached, _error = zephyr_preparation
    with pytest.raises(reached):
        invoke(active=False, selected=False, matrix_active=False)
