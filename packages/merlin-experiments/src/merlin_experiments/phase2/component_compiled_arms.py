"""Optional ordinary compile/static feature arms without per-member RTL runs.

Reuses the original consumed-artifact observer and normal scoped compiler
transport. Static extraction supplies no numerical, traffic or timing authority.
Analytical selection still requires the unchanged live measurement qualifier.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

from merlin.benchharness import hash_tree
from merlin.perf import compiled_static_features, component_source_demand
from merlin.perf.component_cost import ComponentCostRegion, ComponentFeatureObservation
from merlin.targetgen import compile_only_execution, component_program
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin_experiments.phase1.component_package_execution import qualified_package_execution

from . import component_compiled_features
from .component_compiled_features import CompileFeatureSelection, observe_component_compiled_features
from .component_execution import ComponentIndependentExecution
from .contracts import StageGateError, document_sha256, exact_tree_record, mapping_file, sha256_file


@dataclass(frozen=True, eq=False)
class CompiledStaticArmFeatures:
    execution: ComponentIndependentExecution
    selection: CompileFeatureSelection
    feature_ids: tuple[str, ...]
    domain_sha256: str

    @property
    def component_source_pins(self):
        self.verify()
        paths = (
            Path(__file__).resolve(),
            Path(compile_only_execution.__file__).resolve(),
            Path(component_program.__file__).resolve(),
            Path(component_compiled_features.__file__).resolve(),
            Path(compiled_static_features.__file__).resolve(),
            Path(component_source_demand.__file__).resolve(),
        )
        return {
            **dict(self.execution.independent_runtime.source_pins),
            **{
                Path(path): digest
                for path, digest in (
                    *self.selection.build_service.source_pins,
                    *self.selection.elf_admission.source_pins,
                    *self.selection.tools(),
                )
            },
            **{path: sha256_file(path) for path in paths},
        }

    def verify(self):
        if (
            type(self.execution) is not ComponentIndependentExecution
            or type(self.selection) is not CompileFeatureSelection
            or type(self.feature_ids) is not tuple
            or not self.feature_ids
            or len(set(self.feature_ids)) != len(self.feature_ids)
            or any(type(value) is not str or not value for value in self.feature_ids)
        ):
            raise StageGateError("compiled arms require the exact ordinary execution and static observation owners")
        self.execution.verify()
        target = self.execution.independent_runtime.hardware_intake.target
        self.selection.verify(target)
        context = self.execution.independent_runtime.qualification.context
        if self.selection.build_service is not context.build_service:
            raise StageGateError("compiled arms selected a build service outside the original runtime context")
        from merlin.common.digest import is_sha256

        if not is_sha256(self.domain_sha256):
            raise StageGateError("compiled arms require an explicit domain identity; it is not qualification")
        return document_sha256(
            {
                "execution": self.execution.sha256,
                "selection": self.selection.record(),
                "features": self.feature_ids,
                "domain": self.domain_sha256,
            }
        )

    def require_feedback_owner(self, runtime, callback):
        from .component_measurement_qualification import IndependentMeasurementQualification

        self.verify()
        if (
            type(callback) is not MethodType
            or callback.__self__ is not self
            or callback.__func__ is not CompiledStaticArmFeatures.pair
            or runtime.services.feature_provider is not callback
            or type(runtime.qualification) is not IndependentMeasurementQualification
        ):
            raise StageGateError("compiled feature feedback requires its actual live independently evaluated owner")
        runtime.verify(required_roles=("feature_provider",))
        qualification = runtime.qualification
        if (
            qualification.functional_runtime is not self.execution.independent_runtime
            or qualification.baseline_admission is not self.execution.baseline_admission
            or qualification.scope != self.execution.scope
        ):
            raise StageGateError("compiled feature feedback changed its original runtime/corpus/complete scope")
        if any(dict(runtime.source_pins).get(path) != digest for path, digest in self.component_source_pins.items()):
            raise StageGateError("compiled feature producer/tools are outside the qualified source membership")

    def compile(self, *, compiler, member, corpus, workspace, timeout_s):
        """Compile through the original ordinary source/ABI and scoped transport."""
        self.verify()
        self.execution.baseline_admission.verify(corpus=corpus)
        if not any(row is member for row in corpus.capsules):
            raise StageGateError("compiled feature arm is outside the original source roster")
        if type(timeout_s) not in (int, float) or not 1 <= timeout_s <= 600:
            raise StageGateError("compiled feature arm needs a bounded ordinary compile budget")
        workspace, compiler = Path(workspace), Path(compiler)
        if (
            workspace.resolve() != workspace
            or workspace.exists()
            or workspace.is_symlink()
            or not workspace.is_relative_to(self.execution.output)
            or compiler.resolve() != compiler
            or compiler.is_symlink()
            or workspace.is_relative_to(compiler)
            or compiler.is_relative_to(workspace)
        ):
            raise StageGateError("compiled feature arms require fresh private output and exact compiler selection")
        before = exact_tree_record(compiler)["sha256"]
        allowed = (self.execution.candidate, self.execution.baseline_admission.baseline)
        if before not in {exact_tree_record(path)["sha256"] for path in allowed}:
            raise StageGateError("compiled arm selected neither the fresh baseline nor its authorized candidate")
        descriptor = mapping_file(member.source_dir / "capsule.yaml", yaml_file=True)
        if descriptor.get("operation", {}).get("op") != "component_program":
            raise StageGateError("compiled static arms have no original checked integer source for this member")
        storage = descriptor["component_program"]["selected_storage"]
        typed, source_text = component_program.render(
            descriptor["operation"]["attributes"]["program"],
            operand_dtype=storage["operand"],
            accumulator_dtype=storage["accumulator"],
        )
        source = member.source_dir / descriptor["interface_mlir"]
        if source.read_text() != source_text or typed != descriptor["component_program"]:
            raise StageGateError("compiled static arm changed the original complete source/ABI")
        abi = CompileOnlySourceAbi(
            *(
                tuple(CompileOnlyTensor(row["name"], tuple(row["shape"]), row["dtype"]) for row in typed[role])
                for role in ("inputs", "outputs")
            )
        )
        workspace.mkdir(parents=True, mode=0o700)
        copied, products = workspace / "compiler", workspace / "products"
        shutil.copytree(compiler, copied)
        started = time.monotonic()
        with qualified_package_execution(
            candidate=copied,
            view=self.execution.view,
            runtime=self.execution.runtime,
            evidence_root=products,
            container_transport=self.execution.container_transport,
        ):
            compile_only_execution.compile_source_only(
                package_dir=copied,
                source=source,
                original_abi=abi,
                contract_root=self.execution.contract_root,
                target=self.execution.independent_runtime.hardware_intake.target,
                output_root=products,
                build_service=self.selection.build_service,
                elf_admission=self.selection.elf_admission,
                readelf=Path(self.selection.readelf[0]),
                timeout_s=int(timeout_s),
            )
        if time.monotonic() - started >= timeout_s:
            raise StageGateError("compiled static feature arm exhausted its aggregate compile/extraction budget")
        if exact_tree_record(compiler)["sha256"] != before or exact_tree_record(copied)["sha256"] != before:
            raise StageGateError("compiled static extraction changed the original compiler bytes")
        self.verify()
        return {"report_path": str(products / "compile_only_result.json"), "compiler_sha256": before}

    def extract(self, *, member, corpus, workspace):
        """Reopen actual ordinary compile consumption and derive static sites."""
        self.verify()
        workspace = Path(workspace)
        if workspace.resolve() != workspace or not workspace.is_relative_to(self.execution.output):
            raise StageGateError("compiled extraction requires the original private compile owner")
        observed = observe_component_compiled_features(
            report_path=workspace / "products" / "compile_only_result.json",
            package_dir=workspace / "compiler",
            contract_root=self.execution.contract_root,
            corpus=corpus,
            member=member,
            scope=self.execution.scope,
            selection=self.selection,
            evidence_root=workspace / "features",
        )
        self.verify()
        return observed

    def observe(self, *, compiler, member, corpus, workspace, timeout_s):
        """Compile and extract through fixed ordinary functions, with no engine run."""
        started = time.monotonic()
        self.compile(compiler=compiler, member=member, corpus=corpus, workspace=workspace, timeout_s=timeout_s)
        observed = self.extract(member=member, corpus=corpus, workspace=workspace)
        if time.monotonic() - started >= timeout_s:
            raise StageGateError("compiled feature arm exhausted its aggregate compile/extraction budget")
        return observed

    def _observation(self, observed, compiler, member, corpus, workspace):
        # No static site count fills a physical or complete-cost feature. The
        # existing cost model keeps these missing values UNKNOWN.
        context = {
            "compiled_static": observed["features"],
            "physical_memory_traffic": None,
            "dynamic_instruction_counts": None,
            "cold_warm_costs": None,
        }
        region = ComponentCostRegion("complete_program", self.execution.scope.stages, self.feature_ids, context)
        files = tuple((str(path), sha256_file(path)) for path in sorted(workspace.rglob("*")) if path.is_file())
        products = observed["products"]
        return ComponentFeatureObservation(
            compiler_sha256=str(hash_tree(compiler)["sha256"]),
            member_sha256=member.source_sha256,
            corpus_sha256=corpus.capsules_sha256,
            target_sha256=sha256_file(self.execution.baseline_admission.target_descriptor),
            scope_sha256=self.execution.scope.sha256,
            domain_sha256=self.domain_sha256,
            executable_sha256=products[-1]["sha256"],
            dependencies_sha256=document_sha256(observed["selection"]),
            inputs_sha256=document_sha256([member.source_sha256, self.execution.scope.input_policy_sha256]),
            evidence_sha256s=(sha256_file(workspace / "features" / "static_features.json"),),
            cold=(region,),
            warm=(region,),
            functional_status="UNKNOWN",
            legality_status="UNKNOWN",
            artifact_files=files,
        )

    def pair(self, *, baseline, candidate, member, corpus, target_descriptor, scope, workspace, timeout_s):
        self.verify()
        if target_descriptor != self.execution.baseline_admission.target_descriptor or scope != self.execution.scope:
            raise StageGateError("compiled feature pair changed target or complete-cost scope")
        started, results = time.monotonic(), []
        for name, compiler in (("baseline", baseline), ("candidate", candidate)):
            remaining = min(600.0, timeout_s - (time.monotonic() - started))
            arm = Path(workspace) / name
            observed = self.observe(compiler=compiler, member=member, corpus=corpus, workspace=arm, timeout_s=remaining)
            results.append(self._observation(observed, compiler, member, corpus, arm))
        return tuple(results)
