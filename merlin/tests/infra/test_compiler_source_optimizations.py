"""Pure original-owner transport; synthetic artifacts issue no qualification.

Real pass/checker reports and completed recipe correspondence are exercised, but
fixture object/ELF bytes are not compiled or executed. Target admission remains
an independent original-owner obligation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import MethodType

import pytest

from merlin.common.paths import merlin_dir
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.llvmlower.source_optimization_policy import SourceOptimizationPolicy
from merlin.llvmlower.source_optimization_request import GATHER_PASS, PACKING_PASS
from merlin.targetgen import compiler_source_optimizations as bridge
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection

ROLES = ("hardware", "software", "numerical_policy", "runtime", "library")
PERMISSIONS = (PACKING_PASS, GATHER_PASS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OriginalOwner:
    """Explicit neutral original qualifier, never a production target owner."""

    source_optimization_protocol = bridge.PROTOCOL
    source_optimization_permissions = PERMISSIONS
    source_optimization_ceiling = 4096

    def __init__(self, evidence):
        self.evidence = evidence
        self.identity = digest(evidence)
        self.calls = []
        self.mutation = None
        self.policy_mutation = None

    def revalidate(self):
        if digest(self.evidence) != self.identity:
            raise ValueError("original neutral qualification changed")

    def check_compiler_binding(self, package):
        record = json.loads((package / "policy.json").read_text())
        if set(record) != {"producer_packing", "padding_gather", "maximum_added_operations"}:
            raise ValueError("candidate policy tried to select original authority")

    def source_optimization_binding(self, *, inputs):
        self.calls.append("binding")
        values = []
        for role in ROLES:
            row = inputs[role]
            actual = digest(row["path"])
            if actual != row["sha256"]:
                raise ValueError("original context no longer names actual inputs")
            values.append(actual)
        return PassCapabilityBinding(*values)

    def source_optimization_policy(self, package, *, inputs):
        self.calls.append("policy")
        record = json.loads((package / "policy.json").read_text())
        result = SourceOptimizationPolicy(
            **record,
            permitted_passes=self.source_optimization_permissions,
            original_maximum_added_operations=self.source_optimization_ceiling,
        )
        if self.mutation:
            self.mutation()
        if self.policy_mutation:
            self.policy_mutation(result)
        return result

    def prepare_compiler(self, source, destination, *, policy):
        shutil.copytree(source, destination)
        (destination / "policy.json").write_text(json.dumps(policy))
        self.check_compiler_binding(destination)
        return destination

    def check_functional_guard(self, *args, **kwargs):
        raise ValueError("neutral fixture supplies no production functional qualification")


@pytest.fixture
def case(tmp_path):
    files = {}
    for role in ROLES:
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps({"original_role": role}))
        files[role] = path
    evidence = tmp_path / "qualification.txt"
    evidence.write_text("explicit neutral original fixture; no target qualification\n")
    owner = OriginalOwner(evidence)
    binding = PassCapabilityBinding(*(digest(files[role]) for role in ROLES))
    selection = CompilerPassSelection(
        ReviewedPassCapabilities((ReviewedPassCapability(bridge.MECHANISM, binding, owner),)),
        binding,
        mechanisms=(bridge.MECHANISM,),
    )
    original = tmp_path / "original"
    original.mkdir()
    (original / "compiler.py").write_text("# original neutral compiler package\n")
    policy = dict(producer_packing=True, padding_gather=False, maximum_added_operations=4096)
    package = tmp_path / "candidate"
    selection.prepare_compiler(bridge.MECHANISM, original, package, policy=policy)
    capture = tmp_path / "capture"
    capture.mkdir()
    source = (merlin_dir() / "tests/ir/producer_packing_connected.mlir").read_bytes()
    (capture / "model.mlir").write_bytes(source)
    context = {**files, "source": capture / "model.mlir"}
    return owner, selection, package, capture, context


def prepared_compiler(**options):
    """Synthetic object/link fixture with real current pass/checker consumption."""
    from merlin.llvmlower import automatic_destinations, automatic_gathers
    from merlin.llvmlower.compilation_recipe import CompilationRecipe
    from merlin.llvmlower.source_optimization_selection import SourceCompilationOptimization
    from merlin.runtime.source_optimization_build import SourceOptimizationBuild

    policy = options["source_optimizations"]
    work = Path(options["output"]) / "build"
    lower = work / "lower"
    lower.mkdir(parents=True)
    prepared = work / "model.prepared.mlir"
    prepared.write_text("// actual preparation checkpoint\n" + (options["capture"] / "model.mlir").read_text())
    producer = SourceOptimizationBuild(policy)
    producer.bind(prepared)
    current = SourceCompilationOptimization(prepared.read_text(), producer.options()["source_optimizations"])
    after, gather = automatic_gathers.prepare_automatic_gathers(
        prepared.read_text(), serial=True, **current.options(GATHER_PASS)
    )
    selected, packing = automatic_destinations.prepare_automatic_destinations(
        after, serial=True, permuted_outputs=True, **current.options(PACKING_PASS)
    )
    assert packing["rewrites"] == 1
    (lower / "source-optimization-selection.json").write_text(
        json.dumps(current.report(gather=gather, packing=packing))
    )
    llvm, obj, elf = (work / name for name in ("model.ll", "model.o", "model.elf"))
    llvm.write_text("; fixture selected IR SHA " + hashlib.sha256(selected.encode()).hexdigest())
    recipe = CompilationRecipe(work, producer=Path(__file__))
    producer.consumed(recipe, lower)
    tool = work / "fixture-tool"
    tool.write_bytes(b"pure fixture; never executed")
    tool.chmod(0o500)
    for source, destination in ((llvm, obj), (obj, elf)):

        def runner(argv, destination=destination):
            destination.write_bytes(b"pure artifact correspondence fixture " + source.read_bytes())
            return type("Result", (), {"returncode": 0})()

        recipe.run([tool, "-c", source, "-o", destination], runner=runner, inputs=[source], output=destination)
    producer.finish(recipe, elf)
    return {
        "inputs": {"source_optimizations": producer.selection_record()},
        "output": {"elf": str(elf), "source_optimizations": producer.record()},
    }


def invoke(case, tmp_path, *, compiler=prepared_compiler):
    _owner, selection, package, capture, context = case
    return bridge.compile_with_source_optimizations(
        compiler,
        selection=selection,
        compiler_package=package,
        original_inputs=lambda: context,
        capture=capture,
        package=package,
        output=tmp_path / "out",
    )


def test_real_pass_reports_once_prepared_binding_and_completed_recipe(case, tmp_path):
    owner, _selection, _package, capture, context = case
    before = {key: digest(value) for key, value in context.items()}
    result = invoke(case, tmp_path)
    record = result["output"]["source_optimizations"]
    assert record["schema"] == "merlin.source_optimization_build.v2"
    assert record["selection"]["request"]["source_sha256"] == record["input"]["sha256"]
    assert record["input"]["sha256"] != digest(capture / "model.mlir")
    assert json.loads(Path(record["report"]["path"]).read_text())["consumers"][PACKING_PASS]["rewrites"] == 1
    assert before == {key: digest(value) for key, value in context.items()}
    assert owner.calls == ["binding", "policy", "binding"]


@pytest.mark.parametrize("selected", [None, "unrelated"])
def test_no_relevant_selection_preserves_exact_kwargs_without_context_or_files(selected, monkeypatch):
    calls = []

    def compiler(**options):
        calls.append(options)
        return "legacy"

    selection = None if selected is None else type("Unrelated", (), {"mechanisms": ("configuration_cleanup",)})()
    monkeypatch.setattr(bridge, "_pin", lambda *_: pytest.fail("legacy route read input files"))
    assert (
        bridge.compile_with_source_optimizations(
            compiler,
            selection=selection,
            compiler_package="absent",
            original_inputs=lambda: pytest.fail("legacy context"),
            untouched="original",
        )
        == "legacy"
    )
    assert calls == [{"untouched": "original"}]


@pytest.mark.parametrize("role", ROLES)
def test_actual_independent_five_way_input_changes_refuse(case, tmp_path, role):
    *_, context = case
    context[role].write_text("independently changed original " + role)
    with pytest.raises(ValueError, match="binding is stale"):
        invoke(case, tmp_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["unknown", "permission", "ceiling", "wrong_type"])
def test_candidate_cannot_supply_original_permissions_or_limits(case, tmp_path, mutation):
    owner, _, package, *_ = case
    record = json.loads((package / "policy.json").read_text())
    if mutation == "unknown":
        record["permitted_passes"] = list(PERMISSIONS)
    elif mutation == "ceiling":
        record["maximum_added_operations"] = 4097
    elif mutation == "wrong_type":
        record["producer_packing"] = 1
    else:
        owner.source_optimization_permissions = (GATHER_PASS,)
    (package / "policy.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        invoke(case, tmp_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "field", ["source_optimization_protocol", "source_optimization_binding", "source_optimization_policy"]
)
def test_selected_missing_original_protocol_or_context_refuses(case, tmp_path, field):
    owner = case[0]
    setattr(owner, field, None)
    with pytest.raises(ValueError, match="protocol|verification context"):
        invoke(case, tmp_path)


@pytest.mark.parametrize("role", ROLES)
def test_owner_cannot_change_actual_input_before_preparation(case, tmp_path, role):
    owner, *_, context = case
    owner.mutation = lambda: context[role].write_text("changed by policy callback")
    with pytest.raises(ValueError, match="changed"):
        invoke(case, tmp_path)
    assert not (tmp_path / "out").exists()


def test_same_reader_current_context_drift_after_compilation_refuses(case, tmp_path):
    context = case[-1]
    replacement = tmp_path / "changed-runtime.json"
    replacement.write_text("another current runtime")

    def compiler(**options):
        result = prepared_compiler(**options)
        context["runtime"] = replacement
        return result

    with pytest.raises(ValueError, match="context changed"):
        invoke(case, tmp_path, compiler=compiler)


@pytest.mark.parametrize("method", ["record", "bind", "verify_selection", "__post_init__"])
def test_genuine_policy_cannot_shadow_original_methods(case, tmp_path, method):
    owner = case[0]
    executed = []

    def shadow(*_args, **_kwargs):
        executed.append(True)
        raise AssertionError("shadow must never execute")

    owner.policy_mutation = lambda policy: object.__setattr__(policy, method, shadow)
    with pytest.raises(ValueError, match="shadowed"):
        invoke(case, tmp_path)
    assert executed == []


@pytest.mark.parametrize("change", ["function", "code", "defaults"])
def test_original_callback_implementation_cannot_change_during_policy(case, tmp_path, monkeypatch, change):
    owner = case[0]
    original = OriginalOwner.source_optimization_binding

    def replacement(self, *, inputs):
        raise AssertionError("replacement must not execute")

    if change == "function":
        owner.mutation = lambda: monkeypatch.setattr(OriginalOwner, "source_optimization_binding", replacement)
    elif change == "code":
        owner.mutation = lambda: monkeypatch.setattr(original, "__code__", replacement.__code__)
    else:
        owner.mutation = lambda: monkeypatch.setattr(original, "__kwdefaults__", {"inputs": {"foreign": True}})
    with pytest.raises(ValueError, match="callback changed"):
        invoke(case, tmp_path)


def test_actual_callback_source_file_drift_refuses(case, tmp_path):
    owner = case[0]
    path = tmp_path / "original_policy_callback.py"
    path.write_text("def policy(self, package, *, inputs):\n    return self.original_policy(package, inputs=inputs)\n")
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    owner.original_policy = owner.source_optimization_policy
    owner.source_optimization_policy = MethodType(namespace["policy"], owner)
    owner.mutation = lambda: path.write_text("# changed original source\n" + path.read_text())
    with pytest.raises(ValueError, match="changed"):
        invoke(case, tmp_path)


def test_ordinary_compiler_cannot_ignore_the_selected_policy(case, tmp_path):
    def ignored(**_options):
        return {"output": {"elf": "unused"}, "inputs": {}}

    with pytest.raises(ValueError, match="ignored"):
        invoke(case, tmp_path, compiler=ignored)


@pytest.mark.parametrize("change", ["prepared", "report", "recipe", "executable"])
def test_late_original_binding_callback_cannot_rewrite_completed_products(case, tmp_path, monkeypatch, change):
    owner = case[0]
    original = owner.source_optimization_binding
    completed = []
    rewritten = []

    def binding(self, *, inputs):
        current = original(inputs=inputs)
        if completed:
            record = completed[0]["output"]["source_optimizations"]
            key = "input" if change == "prepared" else change
            path = Path(record[key]["path"])
            path.write_bytes(path.read_bytes() + b"\nlate original callback rewrite\n")
            rewritten.append(change)
        return current

    owner.source_optimization_binding = MethodType(binding, owner)

    def compiler(**options):
        result = prepared_compiler(**options)
        completed.append(result)
        return result

    with pytest.raises(ValueError, match="changed|differs|recipe|report|hash"):
        invoke(case, tmp_path, compiler=compiler)
    assert len(completed) == 1 and rewritten == [change]


@pytest.mark.parametrize("component", ["bind", "checker", "build-helper", "recipe-check"])
def test_first_owner_cannot_replace_consumed_builder_or_independent_checker(case, tmp_path, monkeypatch, component):
    from merlin.llvmlower import compilation_recipe, destination_check
    from merlin.runtime import source_optimization_build
    from merlin.runtime.source_optimization_build import SourceOptimizationBuild

    owner = case[0]

    def replacement(*_args, **_kwargs):
        pytest.fail("changed implementation must not execute")

    target, name = {
        "bind": (SourceOptimizationBuild, "_bind"),
        "checker": (destination_check, "verify_permutation_rewrite"),
        "build-helper": (source_optimization_build, "_record"),
        "recipe-check": (compilation_recipe, "verify_completed_recipe"),
    }[component]
    owner.mutation = lambda: monkeypatch.setattr(target, name, replacement)
    with pytest.raises(ValueError, match="code.*changed"):
        invoke(case, tmp_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["depth", "count", "bytes"])
def test_context_construction_is_bounded_before_owner_callbacks(case, tmp_path, change):
    owner, selection, package, capture, _ = case
    if change == "depth":
        context = 0
        for _ in range(34):
            context = [context]
    elif change == "count":
        context = list(range(4097))
    else:
        context = "x" * ((256 << 10) + 1)
    with pytest.raises(ValueError, match="construction bound"):
        bridge.compile_with_source_optimizations(
            prepared_compiler,
            selection=selection,
            compiler_package=package,
            original_inputs=lambda: context,
            capture=capture,
            package=package,
            output=tmp_path / "out",
        )
    assert owner.calls == []


def test_original_owner_cannot_replace_actual_function_local_parser(case, tmp_path, monkeypatch):
    from merlin.frontends import linalg_mlir

    owner = case[0]

    def replacement(*_args, **_kwargs):
        pytest.fail("replaced source parser must not execute")

    owner.mutation = lambda: monkeypatch.setattr(linalg_mlir, "parse_mlir_text", replacement)
    with pytest.raises(ValueError, match="code.*changed"):
        invoke(case, tmp_path)
    assert not (tmp_path / "out").exists()


def test_first_return_context_reader_cannot_mutate_just_returned_receipt(case, tmp_path):
    _owner, selection, package, capture, context = case
    completed = []
    mutated = []

    def compiler(**options):
        result = prepared_compiler(**options)
        completed.append(result)
        return result

    def original_inputs():
        if completed and not mutated:
            completed[0]["owner_mutated_return"] = True
            mutated.append(True)
        return context

    with pytest.raises(ValueError, match="returned products changed"):
        bridge.compile_with_source_optimizations(
            compiler,
            selection=selection,
            compiler_package=package,
            original_inputs=original_inputs,
            capture=capture,
            package=package,
            output=tmp_path / "out",
        )
    assert mutated == [True]
