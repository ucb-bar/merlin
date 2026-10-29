"""Exact completed products for a new ordinary static verification, never a build.

The old partition is retained input identity, not completion authority. Fresh
source, route, host-work, supplier and linked-policy checks remain mandatory.
"""

from __future__ import annotations

import json
import weakref
from contextlib import contextmanager
from pathlib import Path

import yaml

from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256

from . import private_compilation_inputs as compilation
from .private_prebuilt_receipt import load_diagnostic_receipt

_ISSUED = weakref.WeakKeyDictionary()


def ordinary(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("retained static product requires an ordinary absolute file")
    return path


def pin(path):
    path = ordinary(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def reopen(identity):
    if type(identity) is not dict or set(identity) != {"path", "sha256"} or pin(identity["path"]) != identity:
        raise ValueError("retained static input changed")
    return Path(identity["path"])


@contextmanager
def selected_contract(record):
    """Reopen the original model capability for backend-dependent custody checks."""
    from merlin.targetgen import target_registry

    from . import private_full_models as models

    pins = {(item["path"], item["sha256"]): item for item in record["inputs"]
            if item.get("sha256") == record["private_spec_sha256"]}
    if len(pins) != 1:
        raise ValueError("retained partition needs one original private specification pin")
    spec_pin, = pins.values()
    spec = reopen(spec_pin)
    document = yaml.safe_load(spec.read_bytes())
    if not isinstance(document, dict) or document.get("target") != record["target"]:
        raise ValueError("retained private specification names another target")
    rows = [row for row in document.get("models", [])
            if isinstance(row, dict) and row.get("id") == record["selected_model"]]
    if len(rows) != 1:
        raise ValueError("retained private specification needs one selected model")
    row, = rows
    path = models._file(spec, row.get("capability_contract"), row.get("capability_contract_sha256"))
    contract = yaml.safe_load(path.read_bytes())
    if not isinstance(contract, dict) or contract.get("name") != record["target"]:
        raise ValueError("retained capability contract names another target")
    with target_registry.observed_contract(record["target"], contract, source_path=path):
        try:
            yield
        finally:
            reopen(spec_pin)
            models._file(spec, row.get("capability_contract"), row.get("capability_contract_sha256"))


class RetainedProducts:
    def __init__(self, original, *, run_dir, name, candidate, spec, compiler_profile, programs):
        from . import private_model_chain as chain

        original = pin(original)
        record = json.loads(reopen(original).read_bytes())
        if (record.get("schema") != chain.SCHEMA or record.get("scope") != "canonical_static_only"
                or record.get("run_dir") != str(run_dir) or record.get("selected_model") != name
                or record.get("candidate_tree_sha256") != strict_tree_sha256(candidate)["sha256"]
                or record.get("private_spec_sha256") != file_sha256(spec)
                or record.get("compiler_profile") != compiler_profile
                or record.get("required_programs") != programs
                or set(record.get("receipts", {})) != set(programs[name])):
            raise ValueError("retained partition differs from original run/source/profile/programs")
        identities = (original, *record["inputs"], *record["receipts"].values())
        for identity in identities:
            reopen(identity)
        chain.partitions._selection(record, target=record["target"], instruction_path=Path(record["instruction_path"]))
        _ISSUED[self] = (record, identities, {}, {}, json.dumps((record, identities), sort_keys=True))

    def verify(self):
        state = _ISSUED.get(self) if type(self) is RetainedProducts else None
        if state is None:
            raise ValueError("retained products have no original issued selection")
        if json.dumps(state[:2], sort_keys=True) != state[4]:
            raise ValueError("retained products changed their issued original selection")
        for identity in state[1]:
            reopen(identity)
        return state

    def require(self, name, programs, candidate, spec):
        record = self.verify()[0]
        if (record["selected_model"] != name or list(programs) != record["required_programs"][name]
                or candidate != record["candidate_tree_sha256"] or spec != record["private_spec_sha256"]):
            raise ValueError("retained product selection changed model, source or roster")

    def load(self, program, *, capture, host_package, candidate, package_digest, arena_mb, target,
             device, facts, options, math_policy, directory):
        from merlin.llvmlower.device_build import admit_linked_device_elf, freeze_device_elf_admission

        state = self.verify()
        if (strict_tree_sha256(candidate) != package_digest
                or package_digest["sha256"] != state[0]["candidate_tree_sha256"]):
            raise ValueError("retained product candidate source changed")
        receipt = load_diagnostic_receipt(
            reopen(state[0]["receipts"][program]), capture=capture, host_package=host_package,
            candidate=candidate, host_package_tree=strict_tree_sha256(host_package),
            candidate_tree=package_digest, arena_mb=arena_mb, target=target,
            device_selected=device is not None, selected_rtl_facts=facts,
        )
        inputs, output = receipt["inputs"], receipt["output"]
        expected = {**options, "host_math_policy": math_policy, "respect_captured_quantization_scope": True}
        if any(inputs.get(key) != value for key, value in expected.items()):
            raise ValueError("retained product changed selected build settings or numerical policy")
        elf = ordinary(output["elf"])
        compilation.verify(output, elf, required=True)
        policy = None
        emissions = []
        if device is not None:
            with selected_contract(state[0]):
                emissions = _device_custody(
                    output, target=target, candidate=candidate, profile=state[0]["compiler_profile"])
                policy = admit_linked_device_elf(
                    device, freeze_device_elf_admission(device), elf=elf, linked_sha256=output["elf_sha256"],
                    directory=directory,
                )
        state[2][program] = policy
        state[3][program] = emissions
        self.verify()
        return receipt

    def record(self):
        original, identities, policies, emissions, _ = self.verify()
        if set(policies) != set(original["receipts"]):
            raise ValueError("retained static verification omitted a declared program")
        return {"original_partition": identities[0], "receipts": original["receipts"],
                "linked_policy_observations": json.loads(json.dumps(policies)),
                "compiler_emission_observations": json.loads(json.dumps(emissions)),
                "original_producer_completion": "not_claimed",
                "scope": "new static checks of unchanged completed products; numerical execution separate"}


def require_selection(value, *, selected_model, diagnostic, source_only):
    if value is not None:
        if type(value) is not RetainedProducts or selected_model is None or diagnostic or source_only:
            raise ValueError("retained products require an explicit independent canonical static revalidation")
        value.verify()


def _device_custody(output, *, target, candidate, profile):
    from merlin.common import invocation_record as IR
    from merlin.llvmlower.device_build import executable_kernel_entry
    from merlin.targetgen.owned_host_model_execution import _pinned, _verify_device_custody

    from .private_group_provenance import routed_kernel_symbols

    artifacts = output.get("device_artifacts", {})
    kernels = artifacts.get("kernels")
    if artifacts.get("target") != target or type(kernels) is not dict or not kernels:
        raise ValueError("retained device products have no actual compilation custody")
    sidecar = json.loads(reopen(output["device_sidecar"]).read_bytes())
    if set(kernels) != routed_kernel_symbols(sidecar):
        raise ValueError("retained original emission roster omitted a routed kernel")
    link = compilation.verify(output, ordinary(output["elf"]), required=True)
    emissions = []
    for entry in kernels.values():
        pins = entry.get("artifacts", {})
        obj = _pinned(pins, "object")
        if not any(item["path"] == str(obj) and item["sha256"] == file_sha256(obj) for item in link["link_inputs"]):
            raise ValueError("retained original device object was not an actual final-link input")
        _verify_device_custody(
            entry, artifact=_pinned(pins, "llvm_mlir"), llvm_ir=_pinned(pins, "llvm_ir"),
            obj=obj, original_symbol=executable_kernel_entry(target),
            kernel_symbol=entry["kernel_symbol"],
        )
        translation = IR.verify(_pinned(entry["custody"], "translation"))
        emissions.append(_emission_custody(ordinary(translation["inputs"][0]["path"]), candidate, profile=profile))
    return emissions


def _emission_custody(artifact, candidate, *, profile):
    """Reopen the actual completed F emission, not the unfinished model wrapper."""
    from merlin.common import invocation_record as IR
    from merlin.targetgen.package_runtime import load_package
    from merlin.verify.lean_emission_context import _package_sources

    stem = artifact.name.removesuffix(".device.mlir")
    root = artifact.parent / (stem + ".lowering")
    observations = tuple(root.glob("emission-custody-*/invocations/*/invocation.json"))
    if len(observations) != 1:
        raise ValueError("retained device has no unique actual original compiler emission")
    observation = ordinary(observations[0])
    call = IR.verify(observation)
    package = load_package(str(candidate))
    sources = [pin(path) for path in _package_sources(package)]
    if (call.get("kind") != "python_call" or call.get("stage") != "ordinary_source_emission"
            or call.get("callable") != "merlin.targetgen.capsule_common.lower_interface"
            or any(source not in call["dependencies"] for source in sources)
            or call.get("inputs") != [pin(artifact.parent / (stem + ".iface.mlir"))]
            or pin(root / "input.interface.mlir")["sha256"] != call["inputs"][0]["sha256"]
            or not any(item.get("sha256") == file_sha256(artifact)
                       and item.get("path") == str(root / "lowered.llvm.mlir") for item in call["outputs"])):
        raise ValueError("retained device emission does not bind the selected original compiler/source")
    pending = json.loads((observation.parent.parent.parent / "emission_proof_chain.json").read_bytes())
    if (pending.get("schema") != "merlin.emission_pending.v1"
            or pending.get("original_profile_sha256") != profile["profile_sha256"]
            or any(pending.get("compiler_source_invocation", {}).get(key) != value
                   for key, value in pin(observation).items())
            or pending.get("source") != pin(artifact.parent / (stem + ".iface.mlir"))
            or pending.get("emitted_artifact") != pin(root / "lowered.llvm.mlir")):
        raise ValueError("retained original compiler emission changed its selected profile")
    commands = _package_commands(package, root, call, artifact)
    return {"source": pin(observation),
            "profile": pin(observation.parent.parent.parent / "emission_proof_chain.json"),
            "package_calls": commands}


def _package_commands(package, root, emission, artifact):
    from merlin.common import invocation_record as IR
    from merlin.targetgen import package_runtime as runtime

    roles = {"parse", "lower_interface_to_target"}
    roles.update({"emit_analysis_bundle"} if "emit_analysis_bundle" in package.manifest["commands"]
                 else {"emit_command_buffer", "lower_target_to_llvm"})
    observations = tuple(sorted(root.glob("invocations/*/invocation.json")))
    if len(observations) != len(roles):
        raise ValueError("retained original compiler package command roster is incomplete")
    result, seen = [], set()
    for path in observations:
        call = IR.verify(ordinary(path))
        argv = call.get("argv", [])
        if (call.get("kind") != "subprocess" or call.get("stage") != "original_profile_package"
                or call.get("inputs") != [pin(root / "input.interface.mlir")]
                or argv.count("original-package") != 1
                or not emission["started_ns"] <= call["started_ns"] <= call["finished_ns"] <= emission["finished_ns"]):
            raise ValueError("retained compiler package call is not its actual original boxed emission")
        tail = argv[argv.index("original-package") + 1:]
        matches = []
        for role in roles:
            incoming, outgoing = (Path(package.directory) / value for value in ("__saved_input__", "__saved_output__"))
            template = runtime._resolve_argv(package, role, incoming, outgoing)
            if runtime._needs_interpreter(package, template):
                import sys

                template = [sys.executable, *template]
            if _matches_argv(template, tail, (str(incoming), str(outgoing))):
                matches.append(role)
        if len(matches) != 1 or matches[0] in seen:
            raise ValueError("retained compiler package command differs from the selected manifest")
        role = matches[0]
        seen.add(role)
        expected_stdout = (artifact if role in {"emit_analysis_bundle", "lower_target_to_llvm"}
                           else root / "lowered.target.mlir" if role == "lower_interface_to_target" else None)
        if expected_stdout is not None and call["stdout"]["sha256"] != file_sha256(expected_stdout):
            raise ValueError("retained package stdout differs from actual emitted device bytes")
        result.append(pin(path))
    if seen != roles:
        raise ValueError("retained compiler package omitted an original entrypoint")
    return result


def _matches_argv(template, actual, placeholders):
    if len(template) != len(actual):
        return False
    bound = {}
    for expected, value in zip(template, actual, strict=True):
        keys = [key for key in placeholders if key in expected]
        if not keys:
            if expected != value:
                return False
            continue
        if len(keys) != 1 or expected.count(keys[0]) != 1:
            return False
        key = keys[0]
        prefix, _, suffix = expected.partition(key)
        if not value.startswith(prefix) or not value.endswith(suffix) or len(value) < len(prefix) + len(suffix):
            return False
        selected = value[len(prefix):len(value) - len(suffix) if suffix else None]
        if (not Path(selected).is_absolute() or ".." in Path(selected).parts
                or key in bound and bound[key] != selected):
            return False
        bound[key] = selected
    return True


def acquire(value, program, *, diagnostic_path, directory, options, math_policy, device, **inputs):
    """The historical unbound diagnostic stays distinct from typed revalidation."""
    if value is not None:
        return value.load(program, directory=directory, options=options, math_policy=math_policy,
                          device=device, **inputs), {}
    receipt = load_diagnostic_receipt(
        diagnostic_path, **{key: val for key, val in inputs.items() if key not in {"package_digest", "facts"}},
        host_package_tree=strict_tree_sha256(inputs["host_package"]),
        candidate_tree=inputs["package_digest"], selected_rtl_facts=inputs["facts"],
        device_selected=device is not None,
    )
    return receipt, {
        "path": str(diagnostic_path), "sha256": file_sha256(diagnostic_path),
        "historical_rtl_facts_identity": (
            "unavailable_in_prebuilt_receipt" if receipt["inputs"].get("rtl_facts_identity") is None
            else "recorded_exact_selected"
        ),
    }
