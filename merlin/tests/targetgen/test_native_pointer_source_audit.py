"""Current pointer audit rejects forged custody without manufacturing a plan."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_native_pointer_source_binding import frozen  # noqa: F401
from test_native_pointer_source_execution import execution  # noqa: F401

from merlin.targetgen import native_pointer_source_audit as A
from merlin.targetgen import native_pointer_source_execution as E
from merlin.targetgen.native_model_policy import observed_execution


@pytest.fixture
def audit_case(request, monkeypatch, tmp_path):
    args, _, _, observe, contract = request.getfixturevalue("execution")
    service = E.original._build_service_for("neutral")
    service.source_pins = ((str(contract), hashlib.sha256(contract.read_bytes()).hexdigest()),)
    service.render = lambda *_, **__: "original selected logical harness"
    original_backend, citation, _ = E.original._functional_engine("neutral")
    # A genuine recorded neutral Python call exercises invocation custody only;
    # this fixture deliberately supplies no accelerator or target authority.

    class Backend:
        def run_elf(self, elf, *, simulator, timeout):
            assert simulator == "spike" and timeout > 0
            return observe(original_backend, elf, simulator=simulator)[0]

        def parse_output(self, console):
            return original_backend.parse_output(console)

    backend = Backend()
    monkeypatch.setattr(E.original, "_functional_engine", lambda _: (backend, citation, lambda: None))
    monkeypatch.setattr(E, "observed_execution", observed_execution)
    compiler = E.compile_lowered_to_elf

    def recorded_build(buffer, lowered, destination, **kwargs):
        from merlin.common import invocation_record as I

        elf = compiler(buffer, lowered, destination, **kwargs)
        typed = destination / "kernel.llvm.mlir"
        typed.write_text(lowered)
        previous = typed
        for stage, name in (("llvm_translation", "kernel.ll"), ("object", "kernel.o"), ("elf", elf.name)):
            product = destination / name
            # Actual neutral child observations join the exact file roles only;
            # they deliberately provide no real compiler or ISA qualification.
            I.run(
                [str(Path(sys.executable).resolve()), "-c",
                 "from pathlib import Path;import sys;Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes())",
                 str(previous), str(product)],
                directory=destination,
                stage=stage,
                inputs=(previous,),
                outputs=(product,),
                check=True,
            )
            previous = product
        return elf

    monkeypatch.setattr(E, "compile_lowered_to_elf", recorded_build)
    native = E.execute_candidate_pointer_source(**args)
    assert native.get("failure") is None
    generated = tmp_path / "emitted"
    generated.mkdir()
    command = generated / "command_buffer.json"
    lowered = generated / "lowered.llvm.mlir"
    command.write_text(json.dumps(args["command_buffer"]))
    lowered.write_text(args["lowered_mlir_text"])
    emission = dict(
        command_buffer=A._pin(command),
        lowered_mlir=A._pin(lowered),
        source_interface=A._pin(args["capsule_dir"] / "source.mlir"),
    )
    monkeypatch.setattr(
        "merlin.targetgen.native_instruction_audit.audit_native_instruction_tier",
        lambda *_, **__: dict(status="pass", scope="neutral control only"),
    )
    return emission, native, args, contract


def audit(case):
    emission, native, *_ = case
    return A.audit_pointer_source_tiers(emission, native, target="neutral", entry_symbol="selected_entry")


def test_same_elf_current_storage_and_full_values_reaudit_without_whole_program(audit_case):
    report = audit(audit_case)
    assert {tier: row["status"] for tier, row in report["tiers"].items()} == dict(
        L0="pass", L1="pass", L2="pass", L3="pending"
    )
    assert report["status"] == "unverified"
    assert report["missing_tiers"] == ["L3"]
    assert report["hardware_execution_qualified"] is False
    assert report["timing_authority"] == "NONE"
    assert report["source_effect_correspondence"] == "requires original live realization owner"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["binding"]["inputs"][0].update(compiled_name="counterfeit"),
        lambda row: row["pointer_execution"].update(entry_symbol="foreign_entry"),
        lambda row: row["pointer_execution"].update(completion_symbol="foreign_completion"),
        lambda row: row["candidate"].update(command_buffer_sha256="0" * 64),
        lambda row: row["candidate"].update(lowered_mlir_sha256="0" * 64),
        lambda row: row.update(source_execution_protocol="whole_program"),
        lambda row: row.update(oracle_tier_ceiling="L3"),
        lambda row: row.update(transport_implementation=[]),
        lambda row: row.update(host_build_sources=[]),
        lambda row: row["execution_invocations"]["L2"].update(sha256="0" * 64),
    ],
)
def test_changed_source_protocol_or_original_custody_never_passes(audit_case, mutation):
    mutation(audit_case[1])
    report = audit(audit_case)
    assert report["tiers"]["L0"]["status"] == "unverified"
    assert report["status"] != "pass"


def test_missing_build_leaf_is_not_ignored(audit_case):
    native = audit_case[1]
    leaf = next(key for key in native["build_artifacts"] if key.endswith(".elf"))
    native["build_artifacts"].pop(leaf)
    assert audit(audit_case)["tiers"]["L0"]["status"] == "unverified"


def test_unindexed_extra_build_file_refuses(audit_case):
    native = audit_case[1]
    build = next(iter(native["build_artifacts"].values()))["path"]
    from pathlib import Path

    Path(build).parent.joinpath("unindexed.bin").write_bytes(b"not in original build")
    assert audit(audit_case)["tiers"]["L0"]["status"] == "unverified"


def test_coherently_rehashed_console_cannot_replace_original_dispatch(audit_case):
    from pathlib import Path

    native = audit_case[1]
    tier = native["tiers"]["L2"]
    console = Path(tier["console"]["path"])
    console.write_text(console.read_text() + "counterfeit transcript\n")
    tier["console"] = A._pin(console)
    report = audit(audit_case)
    assert "original same-ELF dispatch" in report["tiers"]["L0"]["detail"]


def test_mid_audit_source_mutation_revokes_all_passes(audit_case, monkeypatch):
    import merlin.targetgen.native_numerical_audit as numerical

    original = numerical.audit_native_numerical_tier

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path = audit_case[2]["capsule_dir"] / "source.mlir"
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(numerical, "audit_native_numerical_tier", mutate)
    report = audit(audit_case)
    assert report["tiers"]["L0"]["status"] == "unverified"
    assert report["tiers"]["L2"]["status"] == "unverified"


def test_old_whole_program_tier_reader_is_not_called(audit_case, monkeypatch):
    monkeypatch.setattr(
        E.original, "audit_candidate_static_tiers", lambda *_, **__: pytest.fail("must not fabricate a global plan")
    )
    assert audit(audit_case)["tiers"]["L0"]["status"] == "pass"


def test_rehashed_foreign_dispatch_arguments_cannot_borrow_console(audit_case):
    from pathlib import Path

    native = audit_case[1]
    pin = native["execution_invocations"]["L2"]
    path = Path(pin["path"])
    invocation = json.loads(path.read_bytes())
    invocation["arguments"]["simulator"] = "foreign"
    path.write_text(json.dumps(invocation))
    pin["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = audit(audit_case)
    assert report["tiers"]["L0"]["status"] == "unverified"
    assert "original same-ELF dispatch" in report["tiers"]["L0"]["detail"]


def _refresh_build_inventory(native):
    build = Path(native["elf"]["path"]).parent
    native["build_artifacts"] = {
        path.relative_to(build).as_posix(): A._pin(path)
        for path in sorted(build.rglob("*")) if path.is_file()
    }


def test_coherent_build_inventory_without_translation_producer_never_passes(audit_case):
    native = audit_case[1]
    build = Path(native["elf"]["path"]).parent
    records = [path for path in build.rglob("invocation.json")
               if json.loads(path.read_bytes())["stage"] == "llvm_translation"]
    assert len(records) == 1
    records[0].unlink()
    _refresh_build_inventory(native)
    report = audit(audit_case)
    assert "joined LLVM/object/ELF producer chain" in report["tiers"]["L0"]["detail"]


def test_same_byte_foreign_translation_source_cannot_borrow_producer(audit_case, tmp_path):
    native = audit_case[1]
    build = Path(native["elf"]["path"]).parent
    record = next(path for path in build.rglob("invocation.json")
                  if json.loads(path.read_bytes())["stage"] == "llvm_translation")
    row = json.loads(record.read_bytes())
    foreign = tmp_path / "foreign.llvm.mlir"
    foreign.write_bytes(Path(row["inputs"][0]["path"]).read_bytes())
    row["inputs"][0]["path"] = str(foreign)
    record.write_text(json.dumps(row))
    _refresh_build_inventory(native)
    report = audit(audit_case)
    assert report["tiers"]["L0"]["status"] == "unverified"
    assert "escaped its owner" in report["tiers"]["L0"]["detail"]


def test_unselected_external_emission_root_retains_original_refusal(audit_case):
    from merlin_experiments.phase1.feedback.candidate_observations import reopen_native_products

    emission, native, *_ = audit_case
    build = Path(native["elf"]["path"]).parent
    with pytest.raises(ValueError, match="escaped its owner"):
        reopen_native_products(
            build, emission=emission, elf_pin=native["elf"], source=native["source"],
            build_artifacts=native["build_artifacts"], candidate=native["candidate"],
        )
