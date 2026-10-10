"""Prepared private controls for explicit independent source/native diagnostics.

The controls originate in generic upstream tensor IR and an independently
written private primitive compiler. They are never a public scaffold or author
input. Accelerator instruction effects, physical ownership/synchronization and
hardware/model equivalence remain UNKNOWN here: those missing mechanisms refuse
the fixed fourteen-control runtime qualifier rather than issuing authority.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import uuid
from dataclasses import asdict, dataclass, field
from dataclasses import replace as dataclass_replace
from pathlib import Path
from weakref import WeakKeyDictionary

from merlin.common import invocation_record
from merlin.targetgen import compiler_library as library_owner
from merlin.targetgen import package_runtime
from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_service import BuildOnlyService
from merlin.targetgen.contract.execution_service import FunctionalExecutionService
from merlin.targetgen.native_component_execution import execute_component
from merlin_experiments.execution.container_transport import PreparedContainerTransport
from merlin_experiments.phase1.component_package_execution import ComponentPackageExecutor, qualified_package_execution

from . import component_runtime_controls as controls
from . import component_runtime_copy_controls as copy_controls
from . import component_runtime_copy_support as copy_support_owner
from . import component_runtime_instruction_control as instruction_control
from . import component_runtime_source_selection as source_selection
from . import component_runtime_stage_products as stage_products
from .component_experiment import ComponentView, RuntimeGrant, verify_component_view
from .component_runtime_authority import IndependentRuntimeServices, _callback_identity
from .component_runtime_control_execution import PrivateRuntimeControlExecutor
from .component_runtime_fixture import prepare_source_control
from .component_runtime_qualification import (
    CONTROL_CASES,
    RuntimeControlRefusal,
    _argument_binding,
    _fixture_binding,
)
from .contracts import StageGateError, document_sha256, exact_tree_record, mapping_file, sha256_file, write_json

_PREPARED = WeakKeyDictionary()
_GRADED = WeakKeyDictionary()
_READBACK_CALLBACKS = WeakKeyDictionary()
_READBACK_SELECTIONS = WeakKeyDictionary()
_MEMORY_METHODS = ("prepare", "decode", "verify", "record")
_SUPPORTED_DIAGNOSTICS = {
    "source_correspondence",
    "original_output_roster",
    "original_numeric_gate",
}


def _plain(path):
    path = Path(path).absolute()
    if path.resolve() != path or any(part.is_symlink() for part in (path, *path.parents)):
        raise StageGateError("independent runtime preparation requires canonical unlinked paths")
    return path


@dataclass(frozen=True, eq=False)
class PreparedIndependentRuntimeContext:
    """Concrete source/native diagnostic context; not a runtime issuer.

    Construct with explicit independently selected build and functional transport.
    Neither constructor bytes nor a selected target name qualify those services.
    The fixed qualifier executes this owner and refuses missing physical proof.
    """

    hardware_intake: object
    target_descriptor: Path
    contract_root: Path
    build_service: BuildOnlyService
    execution_service: FunctionalExecutionService
    source_pins: tuple[tuple[Path, str], ...]
    instruction_check: object = None
    container_transport: PreparedContainerTransport | None = None
    compiler_view: ComponentView | None = None
    compiler_runtime: tuple[RuntimeGrant, ...] = ()
    readback_policy: RB.ReadbackPolicy = RB.ReadbackPolicy(RB.FULL_VALUES_B64)
    memory_readback: object = None
    copy_control_support: copy_support_owner.RuntimeCopyControlSupport | None = None
    source_observation: object = None
    compiler_library: library_owner.CompilerLibraryContract | None = None
    compiler_library_root: Path | None = None
    services: IndependentRuntimeServices = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "services", IndependentRuntimeServices(self.grade, self.stage_verifier))
        _PREPARED[self] = {}
        _GRADED[self] = {}
        _READBACK_CALLBACKS[self] = (
            tuple(_callback_identity(getattr(self.memory_readback, name, None)) for name in _MEMORY_METHODS)
            if self.memory_readback is not None
            else ()
        )

    def _readback_selection(self):
        """Attribute the selected observer; this does not qualify its semantics."""
        try:
            policy = RB.selected(self.readback_policy)
        except ValueError as error:
            raise StageGateError("runtime readback requires an explicit trusted policy") from error
        if policy is None:
            raise StageGateError("runtime readback requires an explicit trusted policy")
        memory = policy.transport in RB.MEMORY_TRANSPORTS
        if memory != (self.memory_readback is not None):
            raise StageGateError("runtime coherent policy and selected memory observer disagree")
        identities = _READBACK_CALLBACKS.get(self)
        if identities is None:
            raise StageGateError("runtime readback selection was not prepared by this context")
        if memory:
            current = tuple(_callback_identity(getattr(self.memory_readback, name, None)) for name in _MEMORY_METHODS)
            if identities != current:
                raise StageGateError("runtime coherent observer callbacks changed after preparation")
            execution_pins = {(Path(path), digest) for path, digest in self.execution_service.source_pins}
            for identity in identities:
                if (identity[2], identity[3]) not in execution_pins or (
                    identity[2],
                    identity[3],
                ) not in self.source_pins:
                    raise StageGateError("runtime coherent observer lacks selected execution/context source membership")
            # The selected reader owns the meaning of its immutable descriptor.
            # Per-ELF preparation/readback state is deliberately not serialized as
            # selection. Source identity/configuration attribution is not a role.
            self.memory_readback.verify()
            descriptor = self.memory_readback.record()
            if type(descriptor) is not dict or not descriptor:
                raise StageGateError("runtime coherent observer lacks an explicit immutable selection")
            descriptor = _argument_binding(descriptor)
            digest = document_sha256(descriptor)
            if self in _READBACK_SELECTIONS and _READBACK_SELECTIONS[self] != digest:
                raise StageGateError("runtime coherent observer immutable selection changed")
            _READBACK_SELECTIONS[self] = digest
        else:
            descriptor = None
        return {
            "policy": policy.record(),
            "callbacks": [(str(row[2]), row[3], row[4], row[5]) for row in identities],
            "immutable_selection": descriptor,
            "scope": "explicit transport attribution only; original outputs and runtime qualification remain required",
        }

    @property
    def sha256(self):
        recipe = asdict(self.build_service.recipe)
        error_class = recipe.pop("error_cls")
        recipe["error_cls"] = error_class.__module__ + "." + error_class.__qualname__
        return document_sha256(
            {
                "scope": "private source/native controls; physical target runtime remains unqualified",
                "hardware_intake_sha256": self.hardware_intake.sha256,
                "target_descriptor_sha256": sha256_file(self.target_descriptor),
                "contract_sha256": exact_tree_record(self.contract_root)["sha256"],
                "build_recipe": _argument_binding(recipe),
                "execution": self.execution_service.verify(self.build_service.target, self.execution_service.simulator),
                "source_pins": [(str(path), digest) for path, digest in self.source_pins],
                "instruction_check": self.instruction_check.verify() if self.instruction_check is not None else None,
                "compiler_commands": self._compiler_commands(),
                "readback_selection": self._readback_selection(),
                "copy_control_selection": self._copy_selection(),
                "source_observation_selection": source_selection.selection(self),
                "compiler_library_selection": self._library_selection(),
            }
        )

    def _library_selection(self):
        selection = library_owner.selected_library_record(self.compiler_library, self.compiler_library_root)
        if selection is not None:
            members = {self.compiler_library_root / member.path for member in self.compiler_library.members}
            members.add(Path(library_owner.__file__).resolve())
            if not members <= set(dict(self.source_pins)):
                raise StageGateError("runtime compiler library omits exact selected source membership")
            for path in members:
                if sha256_file(path) != dict(self.source_pins)[path]:
                    raise StageGateError("runtime compiler library source selection changed")
        return selection

    def _copy_selection(self):
        support = self.copy_control_support
        if support is None:
            return None
        if type(support) is not copy_support_owner.RuntimeCopyControlSupport:
            raise StageGateError("copy controls require the fixed source-selection declaration")
        support.verify(hardware=self.hardware_intake, build=self.build_service, context_pins=self.source_pins)
        selection = self._readback_selection()
        if self.memory_readback is None or selection["immutable_selection"].get("effect_source") != {
            "path": str(support.helper_source),
            "sha256": sha256_file(support.helper_source),
        }:
            raise StageGateError("copy controls require the exact independently selected coherent helper observer")
        return support.record()

    def verify(self):
        from merlin_experiments.phase0.rtl_intake import IndependentHardwareIntake

        if type(self.hardware_intake) is not IndependentHardwareIntake or self not in _PREPARED:
            raise StageGateError("runtime preparation requires live independently produced hardware intake")
        self.hardware_intake.verify()
        descriptor = mapping_file(_plain(self.target_descriptor), yaml_file=True)
        if (
            type(self.build_service) is not BuildOnlyService
            or type(self.execution_service) is not FunctionalExecutionService
            or descriptor.get("target") != self.hardware_intake.target
            or descriptor["target"] != self.build_service.target
        ):
            raise StageGateError("prepared runtime source/build/target selections disagree")
        _plain(self.contract_root)
        declared = {
            "schemas/" + name
            for name in (
                "manifest.schema.json",
                "capsule.schema.json",
                "command_buffer.schema.json",
            )
        }
        exact_tree_record(self.contract_root)
        if {
            path.relative_to(self.contract_root).as_posix() for path in self.contract_root.rglob("*") if path.is_file()
        } != declared:
            raise StageGateError("independent runtime contract must contain only the three shared ABI schemas")
        self.build_service.verify(descriptor["target"])
        self.execution_service.verify(descriptor["target"], self.execution_service.simulator)
        self._readback_selection()
        self._copy_selection()
        source_selection.selection(self)
        self._library_selection()
        required = {
            Path(inspect.getsourcefile(value)).resolve()
            for value in (
                PreparedIndependentRuntimeContext,
                controls.parse_primitive,
                execute_component,
                PrivateRuntimeControlExecutor,
                prepare_source_control,
                stage_products.collect,
                source_selection.evaluate,
            )
        }
        required.update(
            Path(path) for path, _ in (*self.build_service.source_pins, *self.execution_service.source_pins)
        )
        if self.instruction_check is not None:
            from .component_instruction_audit import IndependentLinkedInstructionCheck

            if (
                type(self.instruction_check) is not IndependentLinkedInstructionCheck
                or self.instruction_check.policy.command_intake.hardware is not self.hardware_intake
            ):
                raise StageGateError("instruction audit and runtime hardware source origins disagree")
            self.instruction_check.verify()
            required.update(path for path, _ in self.instruction_check.source_pins)
            required.add(Path(instruction_control.__file__))
        if self.copy_control_support is not None:
            required.update((Path(copy_controls.__file__), Path(copy_support_owner.__file__)))
            required.update(path for path, _ in self.copy_control_support.source_pins)
        required.add(self.target_descriptor)
        if self.source_observation is not None:
            from merlin.targetgen.contract import source_observation

            required.add(Path(source_observation.__file__))
            required.update(Path(path) for path, _ in self.source_observation.source_pins)
            required.update(Path(path) for path, _ in self.source_observation.implementation_pins)
        if self.memory_readback is not None:
            required.add(Path(RB.__file__))
        required.update(path for path in self.contract_root.rglob("*") if path.is_file())
        self._compiler_commands()
        if self.container_transport is not None:
            required.update(path for path, _ in self.container_transport.source_pins)
            required.update(path for path in self.compiler_view.root.rglob("*") if path.is_file())
            required.update(row.source for row in self.compiler_runtime)
            required.add(Path(inspect.getsourcefile(ComponentPackageExecutor)).resolve())
        if (
            not isinstance(self.source_pins, tuple)
            or len(self.source_pins) != len(dict(self.source_pins))
            or not required <= set(dict(self.source_pins))
        ):
            raise StageGateError("prepared runtime omits actual control, build or execution source membership")
        for path, digest in self.source_pins:
            if not _plain(path).is_file() or sha256_file(path) != digest:
                raise StageGateError("prepared independent runtime source/tool changed")
        return self.sha256

    def _compiler_commands(self):
        if self.container_transport is None:
            if self.compiler_view is not None or self.compiler_runtime:
                raise StageGateError("compiler grants require an explicit prepared command transport")
            return None
        if (
            type(self.container_transport) is not PreparedContainerTransport
            or type(self.compiler_view) is not ComponentView
            or not isinstance(self.compiler_runtime, tuple)
            or not self.compiler_runtime
            or any(type(row) is not RuntimeGrant for row in self.compiler_runtime)
        ):
            raise StageGateError("runtime compiler commands require exact prepared transport/view/runtime grants")
        self.container_transport.verify()
        verify_component_view(self.compiler_view)
        for row in self.compiler_runtime:
            row.verify()
        return {
            "transport_sha256": self.container_transport.sha256,
            "view_sha256": self.compiler_view.manifest_sha256,
            "runtime": [(str(row.source), row.destination, row.sha256) for row in self.compiler_runtime],
        }

    def prepare_control(self, name, workspace):
        self.verify()
        if name not in CONTROL_CASES:
            raise StageGateError("unknown independent runtime control")
        mechanism, _, direction = name.partition(".")
        copy = self.copy_control_support if mechanism in copy_support_owner.MECHANISMS else None
        if (
            mechanism not in _SUPPORTED_DIAGNOSTICS
            and copy is None
            and not (mechanism == "instruction_audit" and self.instruction_check is not None)
        ):
            raise StageGateError("independent physical/accelerator control remains UNKNOWN: " + mechanism)
        root = _plain(workspace)
        fixture = prepare_source_control(
            name=name,
            root=root,
            build_service=self.build_service,
            contract_root=self.contract_root,
            target_descriptor=self.target_descriptor,
            **({"copy_support": copy} if copy is not None else {}),
        )
        candidate, capsule = fixture.grade_arguments["package_dir"], fixture.capsule_root
        mutation = None
        if name == "instruction_audit.negative":
            mutation = instruction_control.prepare_mutation(root=root, check=self.instruction_check)
        if mechanism == "instruction_audit":
            fixture = dataclass_replace(
                fixture,
                required_invocation_stages=(
                    *fixture.required_invocation_stages,
                    "component_native_execution",
                    "whole_linked_instruction_policy",
                    "native_accessor_decode",
                ),
            )
        _PREPARED[self][root] = (
            fixture,
            exact_tree_record(candidate),
            exact_tree_record(capsule),
            sha256_file(root / "original_policy.json"),
            document_sha256(_fixture_binding(fixture)),
            mutation,
        )
        self.verify_control(fixture)
        return fixture

    def verify_control(self, fixture):
        self.verify()
        stored = _PREPARED[self].get(fixture.evidence_root)
        if stored is None or stored[0] is not fixture:
            raise StageGateError("runtime fixture was not actually prepared by this private context")
        candidate = fixture.grade_arguments["package_dir"]
        if (
            exact_tree_record(candidate) != stored[1]
            or exact_tree_record(fixture.capsule_root) != stored[2]
            or sha256_file(fixture.evidence_root / "original_policy.json") != stored[3]
            or document_sha256(_fixture_binding(fixture)) != stored[4]
            or (
                instruction_control.mutation_identity(fixture.evidence_root, self.instruction_check)
                if fixture.case_id == "instruction_audit.negative"
                else None
            )
            != stored[5]
        ):
            raise StageGateError("private runtime control source, original gate or compiler products changed")
        return fixture.case_id

    def _fixture_for(self, capsule):
        matches = [row[0] for row in _PREPARED[self].values() if row[0].capsule_root == capsule]
        return matches[0] if len(matches) == 1 else None

    def _evaluate_source(self, source, lowered, fixture):
        """Return an evaluated rejection as data before the normal gate raises."""
        return source_selection.evaluate(self, source, lowered, fixture)

    def _source_verifier(self, *, source, command_buffer, lowered_mlir, **kwargs):
        source, lowered = _plain(source), _plain(lowered_mlir)
        fixture = self._fixture_for(source.parent)
        owner, proof_path = lowered.parent, lowered.parent / "source_correspondence.json"
        inputs = (source, lowered, source.parent / "capsule.yaml")
        if fixture is not None:
            inputs += (fixture.evidence_root / "original_policy.json",)
        with invocation_record.observe_call(
            owner,
            stage="primitive_source_verification",
            function=self._evaluate_source,
            arguments={"entry_symbol": self.build_service.recipe.require_kernel_stack_frame().entry_symbol},
            inputs=inputs,
            outputs=(proof_path,),
            dependencies=(Path(controls.__file__), Path(source_selection.__file__)),
        ) as observation:
            evaluated = self._evaluate_source(source, lowered, fixture)
            write_json(proof_path, evaluated)
            observation.returned()
        if evaluated["status"] in {"accepted", "diagnostic_observed"}:
            return evaluated["proof"]
        reason = evaluated["actual_reason"]
        mechanism = fixture.case_id.partition(".")[0] if fixture is not None else None
        reasons = {
            "source_correspondence": "private primitive LLVM does not preserve every original output expression",
            "original_numeric_gate": "original numeric policy was weakened",
        }
        if (
            fixture is not None
            and fixture.case_id.endswith(".negative")
            and mechanism in reasons
            and reason.startswith(reasons[mechanism])
        ):
            raise RuntimeControlRefusal(
                case_id=fixture.case_id, mechanism=mechanism, evidence_files=((proof_path, sha256_file(proof_path)),)
            )
        raise StageGateError("independent source gate refused: " + reason)

    def grade(
        self,
        package_dir,
        *,
        capsules_root,
        runs_root,
        contract,
        target,
        timeout,
        max_workers=1,
        labels=None,
        model_snapshot_root=None,
        original_members=None,
    ):
        """Execute the shared ordinary source/build/original-full-numeric diagnostic."""
        self.verify()
        if original_members is not None:
            from merlin_experiments.phase1.component_original_members import OriginalCandidateMembers

            if type(original_members) is not OriginalCandidateMembers or (
                original_members.standard_ir.references.schema_intake.software.hardware is not self.hardware_intake
            ):
                raise StageGateError("original candidate grading differs from its live original hardware/source owner")
            original_members.verify()
        if (
            max_workers != 1
            or type(timeout) is not int
            or not 0 < timeout <= 600
            or Path(contract) != self.contract_root
            or target != self.build_service.target
        ):
            raise StageGateError("independent diagnostic grade changes its bounded source/runtime selection")
        root, package = _plain(runs_root), _plain(package_dir)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        rows = []
        for capsule in capsules_root:
            capsule = _plain(capsule)
            original_member = (
                original_members.member(capsule)
                if original_members is not None and capsule.is_relative_to(original_members.destination)
                else None
            )
            fixture = self._fixture_for(capsule)
            if fixture is not None:
                self.verify_control(fixture)
            active = package_runtime.active_package_executor()
            if active is None and (fixture is None or package != fixture.grade_arguments["package_dir"]):
                raise StageGateError("authored compiler diagnostics require the ordinary isolated package executor")
            if active is not None and (
                getattr(active, "container_transport", None) is not self.container_transport
                or self.container_transport is not None
                and (
                    type(active) is not ComponentPackageExecutor
                    or active.view != self.compiler_view
                    or active.runtime != self.compiler_runtime
                )
            ):
                raise StageGateError("active compiler executor differs from the prepared command transport/grants")
            declaration = mapping_file(capsule / "capsule.yaml", yaml_file=True)
            original_candidate = exact_tree_record(package)
            output = root / declaration["name"]
            context = (
                qualified_package_execution(
                    candidate=package,
                    view=self.compiler_view,
                    runtime=self.compiler_runtime,
                    evidence_root=root,
                    container_transport=self.container_transport,
                )
                if active is None and self.container_transport is not None
                else package_runtime.scoped_package_executor(
                    PrivateRuntimeControlExecutor(
                        package,
                        root,
                        exact_tree_record(package)["sha256"],
                    )
                )
                if active is None
                else contextlib.nullcontext()
            )
            try:
                with context:
                    result = execute_component(
                        package_dir=package,
                        capsule_dir=capsule,
                        contract_root=self.contract_root,
                        target=target,
                        out_dir=output,
                        build_service=instruction_control.scoped_build_service(
                            build=(
                                self.copy_control_support.selected_build(fixture=fixture, build=self.build_service)
                                if fixture is not None
                                and self.copy_control_support is not None
                                and fixture.case_id.partition(".")[0] in copy_support_owner.MECHANISMS
                                else self.build_service
                            ),
                            fixture=fixture,
                            check=self.instruction_check,
                        ),
                        execution_service=self.execution_service,
                        source_verifier=self._source_verifier,
                        readback_policy=self.readback_policy,
                        **({"memory_readback": self.memory_readback} if self.memory_readback is not None else {}),
                        **(
                            {
                                "compiler_library": self.compiler_library,
                                "compiler_library_root": self.compiler_library_root,
                            }
                            if self.compiler_library is not None
                            else {}
                        ),
                        timeout_s=timeout,
                        **({"original_member": original_member} if original_member is not None else {}),
                        elf_admission=(
                            self.instruction_check.admission_service() if self.instruction_check is not None else None
                        ),
                    )
            except Exception as error:
                # An output-roster defect is rejected before source validation,
                # so re-open its actual emitted buffer to attribute that failure.
                from merlin.targetgen.native_component_execution import (
                    NativeComponentAdmissionRefusal,
                    NativeComponentExecutionError,
                )

                if (
                    fixture is not None
                    and fixture.case_id == "instruction_audit.negative"
                    and type(error) is NativeComponentAdmissionRefusal
                ):
                    instruction_control.attribute_refusal(
                        fixture=fixture, check=self.instruction_check, result=error.result
                    )

                if (
                    fixture is not None
                    and fixture.case_id == "original_output_roster.negative"
                    and type(error) is NativeComponentExecutionError
                    and str(error) == "independent native ABI does not cover every input/output"
                ):
                    buffer_path = output / "generated" / "command_buffer.json"
                    if buffer_path.is_file():
                        actual = mapping_file(buffer_path)
                        if len((actual.get("kernel_abi") or {}).get("outputs", [])) != len(
                            fixture.member["output_roster"]
                        ):
                            raise RuntimeControlRefusal(
                                case_id=fixture.case_id,
                                mechanism="original_output_roster",
                                evidence_files=((buffer_path, sha256_file(buffer_path)),),
                            ) from error
                raise
            if exact_tree_record(package) != original_candidate:
                raise StageGateError("ordinary diagnostic compiler changed during grading")
            products = stage_products.collect(
                ordinary_result=output / "result.json",
                capsule=capsule,
                candidate=package,
                target_descriptor=self.target_descriptor,
                coherent=self.memory_readback is not None,
                grade_owner=root,
            )
            row = {
                "capsule": declaration["name"],
                "status": "pass" if result["numeric_report"]["status"] == "pass" else "fail",
                "numeric": result["numeric_report"]["status"],
                "scope": "ordinary source/build/full-output diagnostic; physical runtime unqualified",
                "ordinary_products": products,
            }
            summary = output / "capsule_result.json"
            write_json(summary, row)
            _GRADED[self][summary] = {
                "summary": stage_products.member(summary),
                "products": copy.deepcopy(products),
                "finite": (),
            }
            rows.append(row)
        self.verify()
        if original_members is not None:
            original_members.verify()
        return {
            "integrity_status": "clean",
            "per_capsule": rows,
            "scope": "original full-output numerical diagnostics only; no runtime qualification",
        }

    def stage_verifier(self, *, result_path, **kwargs):
        self.verify()
        state = self._graded_state(result_path)
        result = mapping_file(_plain(result_path))
        if result.get("status") != "pass" or result.get("numeric") != "pass":
            raise StageGateError("source/native diagnostic did not pass its original numerical gate")
        finite = tuple(
            stage_products.attach(
                collected=state["products"],
                product_path=Path(row["product"]["path"]),
                producer_record=Path(row["producer"]["path"]),
                context_sources=self.source_pins,
            )
            for row in state["finite"]
        )
        if finite != state["finite"]:
            raise StageGateError("finite stage observation changed after actual attachment")
        diagnostics = Path(result_path).parent / ("stage_refusal_" + uuid.uuid4().hex + ".json")
        with invocation_record.observe_call(
            Path(result_path).parent,
            stage="ordinary_stage_refusal_observation",
            function=stage_products.write_refusal_report,
            arguments={"scope": "observation only"},
            inputs=(
                Path(result_path),
                Path(state["products"]["ordinary_result"]["path"]),
                Path(state["products"]["original_source"]["path"]),
                *(Path(row[name]["path"]) for row in finite for name in ("product", "producer")),
            ),
            outputs=(diagnostics,),
            dependencies=(Path(stage_products.__file__),),
        ) as observed:
            stage_products.write_refusal_report(
                diagnostics=diagnostics, products=state["products"], finite=finite, required_controls=CONTROL_CASES
            )
            observed.returned()
        invocation_record.verify(observed.path)
        raise StageGateError(
            "independent accelerator ISA/effects/ownership/synchronization/hardware-runtime proofs remain UNKNOWN; "
            "actual diagnostics: " + str(diagnostics)
        )

    def _graded_state(self, result_path):
        path = _plain(result_path)
        state = _GRADED.get(self, {}).get(path)
        if state is None:
            raise StageGateError("stage proofs remain UNKNOWN: no actual context-owned ordinary grade")
        stage_products.reopen(state["summary"])
        stage_products.verify_collected(state["products"])
        return state

    def attach_stage_observation(self, *, result_path, product_path, producer_record):
        """Retain produced finite data; never invoke a caller's stage factory."""
        self.verify()
        state = self._graded_state(result_path)
        actual = stage_products.attach(
            collected=state["products"],
            product_path=product_path,
            producer_record=producer_record,
            context_sources=self.source_pins,
        )
        state["finite"] = (*state["finite"], copy.deepcopy(actual))
        return actual
