"""Explicit host-stage selection and final provider-closure audit.

Existing HostLLVMTransformChain schedules this stage before its legacy bounded
RNE terminal. Final helper supplier binding is a separate mandatory build gate.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common import owned_process
from merlin.llvmlower import codegen_env
from merlin.llvmlower import late_quant_rne as K
from merlin.llvmlower.compilation_recipe import (
    CompilationRecipe,
    verify_completed_recipe,
)
from merlin.llvmlower.host_transform_chain import HostLLVMArtifact as A
from merlin.llvmlower.host_transform_chain import (
    HostLLVMEmission,
    HostLLVMStageContract,
    HostLLVMTransformStage,
)
from merlin.llvmlower.link_supplier_trace import (
    observe_link_suppliers,
    trace_symbol_flags,
)

from .post_round_clamp import Contract, rewrite


def _contract_record(contract):
    if type(contract) is not Contract:
        raise TypeError("exact typed post-round contract required")
    contract.validate()
    contract.callee_policy.validate()
    return asdict(contract)


def selected(*, enabled=False, contract=None):
    """Absent defaults preserve existing lowering. Opt-in always needs a contract."""
    if type(enabled) is not bool:
        raise TypeError("explicit Boolean host post-round selection required")
    if not enabled:
        if contract is not None:
            raise ValueError("contract supplied without explicit selection")
        return False
    if contract is None:
        raise ValueError("post-round selection requires explicit source/effect contract")
    _contract_record(contract)
    return True


class PostRoundStage:
    def __init__(self, contract, assembler):
        self.contract = contract
        self.assembler = A.capture(Path(assembler))
        _contract_record(contract)

    def transform(self, source, work):
        self.assembler.validate()
        changed, receipt = rewrite(Path(source).read_text(), self.contract)
        target = Path(work) / "selected.ll"
        target.write_text(changed)
        (Path(work) / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return target

    def verify_emission(self, source, selected, work):
        self.assembler.validate()
        expected, receipt = rewrite(Path(source).read_text(), self.contract)
        actual = json.loads((Path(work) / "receipt.json").read_bytes())
        if Path(selected).read_text() != expected or actual != receipt:
            raise ValueError("actual post-round LLVM/receipt differs from independently recomputed emission")
        command = [
            str(self.assembler.path),
            str(selected),
            "-o",
            str(Path(work) / "selected.bc"),
        ]
        r = subprocess.run(command, capture_output=True, timeout=120)
        (Path(work) / "assembler.stdout").write_bytes(r.stdout)
        (Path(work) / "assembler.stderr").write_bytes(r.stderr)
        if r.returncode:
            raise ValueError("post-round emitted LLVM rejected by selected assembler")
        assembly = {
            "argv": command,
            "returncode": r.returncode,
            "assembler": self.assembler.to_dict(),
            "stdout": A.capture(Path(work) / "assembler.stdout").to_dict(),
            "stderr": A.capture(Path(work) / "assembler.stderr").to_dict(),
            "selected_bc": A.capture(Path(work) / "selected.bc").to_dict(),
        }
        (Path(work) / "assembly.json").write_text(json.dumps(assembly, indent=2) + "\n")
        return HostLLVMEmission(
            A.capture(source),
            A.capture(selected),
            (
                A.capture(Path(work) / "receipt.json"),
                A.capture(Path(work) / "assembly.json"),
            ),
        )


def supplier_link_flags(*, enabled=False, contract=None):
    return trace_symbol_flags(["roundevenf"]) if selected(enabled=enabled, contract=contract) else ()


def verify_final_supplier(contract, *, actual_recipe_path, link_argv, link_inputs, link_stderr):
    """Reopen actual freshly compiled helper then use the real final link trace.

    Caller binds this returned report in the final compilation recipe. A previous
    baseline receipt cannot substitute for this final-link observation.
    """
    _contract_record(contract)
    p = Path(actual_recipe_path)
    recipe = json.loads(p.read_bytes())
    rows = [r for r in recipe.get("commands", []) if Path(r.get("requested_output", "")).name == "mlir_rt.o"]
    if len(rows) != 1:
        raise ValueError("actual fresh helper command missing/ambiguous")
    row = rows[0]
    obj = Path(row["requested_output"])
    source = Path(contract.supplier_source_path)
    if (
        hashlib.sha256(obj.read_bytes()).hexdigest() != contract.supplier_sha256
        or row.get("output", {}).get("sha256") != contract.supplier_sha256
        or not any(
            x.get("path") == str(source) and x.get("sha256") == contract.supplier_source_sha256
            for x in row.get("inputs", [])
        )
        or any(x.startswith("-DMERLIN_EXTERNAL_ROUNDEVEN_F32") for x in row["argv"])
    ):
        raise ValueError("actual helper differs from source-selected reviewed binary implementation")
    observation = observe_link_suppliers(
        argv=link_argv,
        input_identities=link_inputs,
        expected_suppliers={"roundevenf": obj},
        stderr=link_stderr,
    )
    return {
        "schema": "merlin.post_round_clamp_final_supplier.v1",
        "actual_recipe": A.capture(p).to_dict(),
        "actual_helper": A.capture(obj).to_dict(),
        "observed_suppliers": observation,
        "contract": _contract_record(contract),
        "scope": "Actual final symbol supplier/source/object closure only; no numerical or physical verdict.",
    }


class BoundPostRoundStage(PostRoundStage):
    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError("selected stage owner is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, contract, assembler, contract_document):
        super().__init__(contract, assembler)
        self.document = A.capture(Path(contract_document))
        self.check_document()

    def check_document(self):
        self.document.validate()
        if json.loads(self.document.path.read_bytes()) != asdict(self.contract):
            raise ValueError("selected typed contract differs from pinned contract document")

    def transform(self, source, work):
        self.check_document()
        return super().transform(source, work)

    def verify_emission(self, source, selected, work):
        self.check_document()
        return super().verify_emission(source, selected, work)


def make_stage(
    *,
    enabled=False,
    contract=None,
    contract_document=None,
    assembler=None,
    semantic_proofs=(),
):
    if not selected(enabled=enabled, contract=contract):
        return None
    if assembler is None or contract_document is None or type(semantic_proofs) is not tuple or not semantic_proofs:
        raise ValueError("pinned typed contract document, assembler and provider evidence required")
    owner = BoundPostRoundStage(contract, assembler, contract_document)
    root = Path(__file__).parent
    sources = tuple(
        A.capture(p)
        for p in [
            Path(__file__),
            root / "post_round_clamp.py",
            root / "post_round_clamp_guard.py",
            Path(K.__file__),
        ]
    )
    proofs = tuple(A.capture(Path(p)) for p in semantic_proofs) + (
        owner.document,
        A.capture(Path(contract.callee_policy.path)),
        owner.assembler,
    )
    declaration = HostLLVMStageContract("post_round_clamp_rne", sources, proofs)
    result = HostLLVMTransformStage(declaration, owner.transform, owner.verify_emission)
    result.validate()
    return result


def identity_equal(left, right):
    if (
        Path(left).stat().st_size != Path(right).stat().st_size
        or A.capture(Path(left)).sha256 != A.capture(Path(right)).sha256
    ):
        raise ValueError("traced twin bytes differ from original executable")


def trace_completed_link(recipe_path, destination, *, expected_suppliers, timeout_s=180):
    if type(timeout_s) not in {int, float} or not 0 < timeout_s <= 600:
        raise ValueError("finite bounded trace-link timeout required")
    recipe_path = Path(recipe_path)
    before = A.capture(recipe_path)
    original = Path(json.loads(recipe_path.read_bytes())["executable"]["path"])
    record = verify_completed_recipe(recipe_path, executable=original, expected_recipe_sha256=before.sha256)
    link = record["commands"][-1]
    original_pin = A.capture(original)
    suppliers = {name: A.capture(Path(path)) for name, path in expected_suppliers.items()}
    destination = Path(destination)
    if destination.exists() or destination.is_symlink() or destination.parent.resolve() != destination.parent:
        raise ValueError("fresh canonical traced-twin output required")
    if codegen_env.snapshot() != link["environment"]:
        raise ValueError("selected MERLIN environment differs from original final link")
    flags = trace_symbol_flags(list(suppliers))
    argv = list(link["argv"])
    if (
        argv.count("-o") != 1
        or argv[argv.index("-o") + 1] != str(original)
        or any(argv.count(flag) > 1 for flag in flags)
        or any(
            not any(r["path"] == str(supplier.path) and r["sha256"] == supplier.sha256 for r in link["inputs"])
            for supplier in suppliers.values()
        )
    ):
        raise ValueError("original explicit final link/supplier binding unsupported")
    destination.mkdir()
    output = destination / "traced-twin.elf"
    argv[argv.index("-o") + 1] = str(output)
    for flag in flags:
        if flag not in argv:
            argv.insert(argv.index("-o"), flag)
    replay = CompilationRecipe(destination, producer=Path(__file__))
    replay.bind_preparation("original_recipe", recipe_path)
    replay.bind_preparation("original_executable", original)

    def run(command, **kwargs):
        r = owned_process.run(
            [str(x) for x in command],
            cwd=kwargs.get("cwd"),
            env=dict(os.environ),
            capture_output=True,
            timeout=timeout_s,
        )
        (destination / "link.stdout").write_bytes(r.stdout)
        (destination / "link.stderr").write_bytes(r.stderr)
        (destination / "owned-cleanup.json").write_text(json.dumps(r.owned_process_cleanup, indent=2) + "\n")
        return subprocess.CompletedProcess(
            [str(x) for x in command],
            r.returncode,
            r.stdout.decode(),
            r.stderr.decode(),
        )

    result = replay.run(
        argv,
        runner=run,
        inputs=[Path(r["path"]) for r in link["inputs"]],
        output=output,
        cwd=Path(link["cwd"]),
    )
    replay.record_link_suppliers({name: supplier.path for name, supplier in suppliers.items()}, result)
    identity_equal(original, output)
    before.validate()
    original_pin.validate()
    for supplier in suppliers.values():
        supplier.validate()
    verify_completed_recipe(recipe_path, executable=original, expected_recipe_sha256=before.sha256)
    replay.completed(output)
    observation = {
        "schema": "merlin.identical_executable_traced_link_twin.v1",
        "original_recipe": before.to_dict(),
        "original_executable": original_pin.to_dict(),
        "twin_executable": A.capture(output).to_dict(),
        "twin_recipe": A.capture(replay.path).to_dict(),
        "same_executable_bytes": True,
        "selected_suppliers": {name: supplier.to_dict() for name, supplier in suppliers.items()},
        "actual_link_argv": argv,
        "link_stderr": A.capture(destination / "link.stderr").to_dict(),
        "scope": (
            "New actual traced link produces exactsameELFbytes "
            "usingunchangedoriginalinputs. "
            "Itdoesnotclaimoriginalinvocationhadtraceflags,nor "
            "numerical/physicalqualification."
        ),
    }
    p = destination / "traced-twin.json"
    p.write_text(json.dumps(observation, indent=2) + "\n")
    return observation, result, replay.record["commands"][-1]


def audit_roundeven_supplier(contract, recipe_path, destination, *, timeout_s=180):
    row = [
        r
        for r in json.loads(Path(recipe_path).read_bytes())["commands"]
        if Path(r.get("requested_output", "")).name == "mlir_rt.o"
    ]
    if len(row) != 1:
        raise ValueError("actual helper unavailable")
    observed, result, command = trace_completed_link(
        recipe_path,
        destination,
        expected_suppliers={"roundevenf": Path(row[0]["requested_output"])},
        timeout_s=timeout_s,
    )
    final = verify_final_supplier(
        contract,
        actual_recipe_path=recipe_path,
        link_argv=command["argv"],
        link_inputs=command["inputs"],
        link_stderr=result.stderr,
    )
    p = Path(destination) / "roundeven-supplier.json"
    p.write_text(json.dumps({"traced_twin": observed, "supplier": final}, indent=2) + "\n")
    return A.capture(p).to_dict()


def audit_closed_callee_suppliers(
    contract,
    recipe_path,
    destination,
    *,
    support_policy_path,
    support_policy_sha256,
    timeout_s=180,
):
    """Trace every approved external provider, not just the replaced RNE helper."""
    contract.validate()
    policy = contract.callee_policy.validate()
    record = json.loads(Path(recipe_path).read_bytes())
    inputs = record["commands"][-1]["inputs"]
    selected = {}
    support_path = Path(support_policy_path)
    if A.capture(support_path).sha256 != support_policy_sha256:
        raise ValueError("selected support object policy changed")
    support = json.loads(support_path.read_bytes())
    if (
        support.get("schema") != "merlin.host_closed_support_object_policy.v1"
        or type(support.get("objects")) is not list
    ):
        raise ValueError("exact support object closure required")
    from collections import Counter

    expected = Counter((x["sha256"], x["bytes"]) for x in support["objects"])
    actual = Counter(
        (x["sha256"], x["bytes"])
        for x in inputs
        if x["path"].endswith(".o") and Path(x["path"]).name not in {"model.o", "weights_blob.o"}
    )
    if actual != expected:
        raise ValueError("actual non-model support closure changed; source/effect policy needs separate review")
    controls = Counter((x["sha256"], x["bytes"]) for x in support.get("link_control_inputs", []))
    actual_controls = Counter((x["sha256"], x["bytes"]) for x in inputs if not x["path"].endswith(".o"))
    if not controls or actual_controls != controls:
        raise ValueError("selected linker script/archive supplier closure changed")
    for x in support["objects"]:
        if A.capture(Path(x["path"])).sha256 != x["sha256"]:
            raise ValueError("original reviewed support object changed")
    for binding in policy["bindings"]:
        name = binding["actual_symbol"]
        if name in selected:
            raise ValueError("duplicate actual policy symbol")
        allowed = {p["sha256"] for p in binding["object_pins"]}
        choices = [Path(p["path"]) for p in inputs if p["sha256"] in allowed and p["path"].endswith(".o")]
        if len(choices) != 1:
            raise ValueError("actual final source-owned callee supplier absent/ambiguous: " + name)
        selected[name] = choices[0]
    if selected.get("roundevenf") is None:
        raise ValueError("roundevenf not part of closed supplier policy")
    observed, result, command = trace_completed_link(
        recipe_path, destination, expected_suppliers=selected, timeout_s=timeout_s
    )
    helper = verify_final_supplier(
        contract,
        actual_recipe_path=recipe_path,
        link_argv=command["argv"],
        link_inputs=command["inputs"],
        link_stderr=result.stderr,
    )
    contract.validate()
    contract.callee_policy.validate()
    p = Path(destination) / "closed-callee-suppliers.json"
    p.write_text(
        json.dumps(
            {
                "schema": "merlin.closed_value_only_callee_supplier_recheck.v1",
                "policy": {
                    "path": contract.callee_policy.path,
                    "sha256": contract.callee_policy.sha256,
                },
                "support_policy": A.capture(support_path).to_dict(),
                "all_actual_symbols": sorted(selected),
                "traced_twin": observed,
                "helper": helper,
                "scope": (
                    "Everydirectexternalprovider source/object binding selectedbyexplicitpolicy "
                    "is reclosed against actual identicalELF tracedtwin. Unknown/unmatched "
                    "providers refuse. Transitive source/effect premises remain provider-owned."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return A.capture(p).to_dict()


@dataclass(frozen=True)
class ClosedCalleeQualification:
    """Explicit provider-context selection; never an inferred FENV theorem.

    The caller owns the reviewed effect premise. Current source/object evidence
    and an identical-ELF final-supplier audit preserve that exact selection.
    A successful transform emission alone does not fulfill this obligation.
    """

    contract: Contract
    contract_document: A
    support_policy: A
    context_review: A

    def record(self):
        _contract_record(self.contract)
        for item in (
            self.contract_document,
            self.support_policy,
            self.context_review,
        ):
            if type(item) is not A:
                raise TypeError("typed current closed-context evidence required")
            item.validate()
            if item.bytes == 0:
                raise ValueError("nonempty reviewed closed-context evidence required")
        if json.loads(self.contract_document.path.read_bytes()) != asdict(self.contract):
            raise ValueError("selected contract differs from reviewed document")
        support = json.loads(self.support_policy.path.read_bytes())
        if (
            support.get("schema") != "merlin.host_closed_support_object_policy.v1"
            or type(support.get("objects")) is not list
            or not support["objects"]
            or type(support.get("link_control_inputs")) is not list
            or not support["link_control_inputs"]
            or support.get("source_effect_review") != self.context_review.to_dict()
        ):
            raise ValueError("explicit support, link-control and matching context-review closure required")
        paths = set()
        for row in support["objects"] + support["link_control_inputs"]:
            if type(row) is not dict or set(row) != {"path", "sha256", "bytes"}:
                raise ValueError("exact reviewed support identity required")
            if type(row["bytes"]) is not int or type(row["sha256"]) is not str:
                raise ValueError("typed reviewed support size and digest required")
            if type(row["path"]) is not str or not Path(row["path"]).is_absolute():
                raise ValueError("absolute reviewed support path required")
            if row["path"] in paths or A.capture(Path(row["path"])).to_dict() != row:
                raise ValueError("reviewed support input duplicated or changed")
            paths.add(row["path"])
        return {
            "schema": "merlin.host_closed_callee_qualification_selection.v1",
            "contract": asdict(self.contract),
            "contract_document": self.contract_document.to_dict(),
            "support_policy": self.support_policy.to_dict(),
            "context_review": self.context_review.to_dict(),
            "premise": "caller_reviewed_no_fenv_errno_observation_not_inferred",
        }

    def audit(
        self,
        *,
        recipe_path,
        recipe_sha256,
        executable,
        executable_sha256,
        destination,
    ):
        before = self.record()
        executable = Path(executable)
        original = A.capture(executable)
        if original.sha256 != executable_sha256:
            raise ValueError("qualification requires the actual selected executable")
        recipe_path = Path(recipe_path)
        verify_completed_recipe(
            recipe_path,
            executable=executable,
            expected_recipe_sha256=recipe_sha256,
        )
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            raise ValueError("fresh closed-context qualification output required")
        destination.mkdir()
        observation = audit_closed_callee_suppliers(
            self.contract,
            recipe_path,
            destination / "traced-link",
            support_policy_path=self.support_policy.path,
            support_policy_sha256=self.support_policy.sha256,
        )
        verify_completed_recipe(
            recipe_path,
            executable=executable,
            expected_recipe_sha256=recipe_sha256,
        )
        if self.record() != before or A.capture(executable) != original:
            raise ValueError("closed-context inputs or executable changed during audit")
        result = {
            "schema": "merlin.host_closed_callee_build_qualification.v1",
            "selection": before,
            "executable": original.to_dict(),
            "recipe": A.capture(recipe_path).to_dict(),
            "actual_supplier_audit": observation,
            "scope": (
                "current reviewed provider context and actual final-link closure; not numerical or timing qualification"
            ),
        }
        target = destination / "qualification.json"
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return A.capture(target).to_dict()


def qualification_record(chain, qualification):
    """Require an exact context for this concrete stage, before build I/O."""
    from merlin.llvmlower.host_transform_chain import HostLLVMTransformChain

    if chain is None:
        if qualification is not None:
            raise ValueError("closed-context qualification supplied without a chain")
        return None
    if type(chain) is not HostLLVMTransformChain:
        raise TypeError("exact typed host chain required")
    chain.validate()
    selected_stages = [
        stage
        for stage in chain.stages
        if stage.contract.name == "post_round_clamp_rne"
        or isinstance(getattr(stage.transform, "__self__", None), PostRoundStage)
    ]
    if not selected_stages:
        if qualification is not None:
            raise ValueError("closed-context qualification supplied for another stage")
        return None
    if len(selected_stages) != 1 or type(qualification) is not ClosedCalleeQualification:
        raise TypeError("post-round stage requires one exact typed current closed-context qualifier")
    stage = selected_stages[0]
    owner = getattr(stage.transform, "__self__", None)
    if (
        type(owner) is not BoundPostRoundStage
        or getattr(stage.verify_emission, "__self__", None) is not owner
        or getattr(stage.transform, "__func__", None) is not BoundPostRoundStage.transform
        or getattr(stage.verify_emission, "__func__", None) is not BoundPostRoundStage.verify_emission
        or asdict(owner.contract) != asdict(qualification.contract)
        or qualification.contract_document not in stage.contract.semantic_proofs
    ):
        raise ValueError("selected stage and closed-context qualification do not match")
    return qualification.record()
