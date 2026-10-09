"""Explicit single-arm process extraction and independent baseline cache reuse.

Each command receives one compiler and the same already prepared operands. The
diagnostic process mode proves attribution/cache mechanics only; experimental
selection requires the strict namespace and unchanged live measurement gates.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, fields
from pathlib import Path
from types import MethodType

from merlin.benchharness import hash_tree
from merlin.common import invocation_record as I
from merlin.perf.component_cost import ComponentCostRegion, ComponentFeatureObservation

from .component_experiment import RuntimeGrant
from .component_feature_inputs import PreparedFeatureInputs, pin, plain
from .contracts import StageGateError, document_sha256, exact_tree_record
from .feedback_protocol import FeedbackValueLimits, _syntax_depth, encode_feedback_value

_TOKENS = {
    "{compiler}": "/compiler",
    "{source}": "/source.mlir",
    "{inputs}": "/inputs.json",
    "{request}": "/request.json",
    "{output}": "/output/observation.json",
}


def _closed_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StageGateError("feature process JSON repeats a field")
        result[key] = value
    return result


def _read(path, limits):
    path = plain(path)
    if not path.is_file() or path.stat().st_size > limits.max_bytes:
        raise StageGateError("feature process product exceeds its closed byte budget")
    with path.open("rb") as stream:
        data = stream.read(limits.max_bytes + 1)
    if len(data) > limits.max_bytes:
        raise StageGateError("feature process product grew past its byte budget")
    try:
        _syntax_depth(data, limits, None)
        value = json.loads(data, object_pairs_hook=_closed_pairs)
        encode_feedback_value(value, limits=limits)
    except (ValueError, RecursionError) as error:
        raise StageGateError("feature process product is not bounded plain JSON") from error
    return value


def _observation(path, limits):
    value = _read(path, limits)
    if type(value) is not dict or set(value) != {field.name for field in fields(ComponentFeatureObservation)}:
        raise StageGateError("feature process omitted its exact observation roster")
    value = dict(value)
    for regime in ("cold", "warm"):
        regions = value[regime]
        if type(regions) is not list or not regions:
            raise StageGateError("feature process omitted a cold/warm region roster")
        converted = []
        for row in regions:
            if type(row) is not dict or set(row) != {field.name for field in fields(ComponentCostRegion)}:
                raise StageGateError("feature process region is not the closed cost declaration")
            row = dict(row)
            row["stages"], row["feature_ids"] = tuple(row["stages"]), tuple(row["feature_ids"])
            converted.append(ComponentCostRegion(**row))
        value[regime] = tuple(converted)
    value["evidence_sha256s"] = tuple(value["evidence_sha256s"])
    artifacts = []
    for row in value["artifact_files"]:
        if type(row) is not list or len(row) != 2 or any(type(item) is not str for item in row):
            raise StageGateError("feature process artifact membership is malformed")
        member = Path(row[0])
        member = plain(member if member.is_absolute() else path.parent / member)
        if not member.is_relative_to(path.parent):
            raise StageGateError("feature process artifact escapes its actual private output owner")
        artifacts.append((str(member), row[1]))
    value["artifact_files"] = tuple(artifacts)
    observation = ComponentFeatureObservation(**value)
    if not observation.artifact_files:
        raise StageGateError("feature process lacks actual artifact membership")
    for path, digest in observation.artifact_files:
        if pin(Path(path)) != (path, digest):
            raise StageGateError("feature process actual artifacts changed")
    if observation.executable_sha256 not in {digest for _path, digest in observation.artifact_files}:
        raise StageGateError("feature process omitted its actual declared executable artifact")
    return observation


@dataclass(frozen=True)
class FeatureArmCommand:
    executable: Path
    argv_template: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    source_pins: tuple[tuple[str, str], ...]
    mode: str
    runtime: tuple[RuntimeGrant, ...] = ()
    sandbox: Path | None = None

    def verify(self):
        if (
            type(self) is not FeatureArmCommand
            or type(self.argv_template) is not tuple
            or any(type(token) is not str or "\0" in token for token in self.argv_template)
            or any(self.argv_template.count(token) != 1 for token in _TOKENS)
            or any(("{" in token or "}" in token) and token not in _TOKENS for token in self.argv_template)
            or type(self.environment) is not tuple
            or len(dict(self.environment)) != len(self.environment)
            or any(
                type(row) is not tuple or len(row) != 2 or any(type(v) is not str for v in row)
                for row in self.environment
            )
            or self.mode not in {"strict_namespace.v1", "diagnostic_process.v1"}
            or type(self.source_pins) is not tuple
            or not self.source_pins
            or len(dict(self.source_pins)) != len(self.source_pins)
        ):
            raise StageGateError("feature arms require a closed explicit command/environment selection")
        required = (self.executable, Path(__file__).resolve(), Path(I.__file__).resolve())
        if self.mode == "strict_namespace.v1":
            if (
                self.sandbox is None
                or not self.runtime
                or any(type(row) is not RuntimeGrant for row in self.runtime)
                or len({row.destination for row in self.runtime}) != len(self.runtime)
            ):
                raise StageGateError("independent feature command requires its selected strict namespace tools")
            required += (self.sandbox,)
            for row in self.runtime:
                row.verify()
                if dict(self.source_pins).get(str(row.source)) != row.sha256:
                    raise StageGateError("feature namespace tool is outside its exact source/dependency roster")
            if not any(row.source == self.executable for row in self.runtime):
                raise StageGateError("feature command tool is outside explicit namespace membership")
        elif self.runtime or self.sandbox is not None:
            raise StageGateError("diagnostic command cannot claim namespace grants")
        pins = dict(self.source_pins)
        for path in required:
            if pins.get(str(path)) != pin(path)[1]:
                raise StageGateError("feature command omitted its fixed source/tool ownership")
        for path, digest in self.source_pins:
            if pin(Path(path)) != (path, digest):
                raise StageGateError("feature command source or tool bytes changed")
        if not os.access(self.executable, os.X_OK):
            raise StageGateError("feature command selected executable is unavailable")
        return document_sha256(
            {
                "tool": pin(self.executable),
                "argv": self.argv_template,
                "environment": I.environment_identity(dict(self.environment)),
                "mode": self.mode,
                "source_pins": self.source_pins,
                "runtime": [(str(row.source), row.destination, row.sha256) for row in self.runtime],
            }
        )


@dataclass(frozen=True, eq=False)
class IndependentArmFeatures:
    inputs: PreparedFeatureInputs
    command: FeatureArmCommand
    cache_root: Path
    limits: FeedbackValueLimits

    @property
    def component_source_pins(self):
        self.verify()
        return {
            Path(path): digest
            for path, digest in (
                *self.command.source_pins,
                *self.inputs.source_pins,
                pin(self.inputs.record_path),
                pin(self.inputs.invocation),
                *((str(path), digest) for _member, path, digest in self.inputs.members),
            )
        }

    def verify(self):
        if type(self.inputs) is not PreparedFeatureInputs or type(self.command) is not FeatureArmCommand:
            raise StageGateError("independent arms require their actual input/command owners")
        if type(self.limits) is not FeedbackValueLimits:
            raise StageGateError("independent arms require explicit bounded result limits")
        self.limits.verify()
        encode_feedback_value(
            [self.command.argv_template, self.command.environment, self.command.source_pins], limits=self.limits
        )
        self.inputs.verify()
        self.command.verify()
        root = plain(self.cache_root)
        if root.exists() and (not root.is_dir() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077):
            raise StageGateError("independent feature cache must be private and owned")
        return document_sha256([self.command.verify(), self.inputs.sha256, self.limits.__dict__])

    def require_feedback_owner(self, runtime, callback):
        from .component_measurement_qualification import IndependentMeasurementQualification

        self.verify()
        if (
            self.command.mode != "strict_namespace.v1"
            or type(callback) is not MethodType
            or callback.__self__ is not self
            or callback.__func__ is not IndependentArmFeatures.pair
            or type(runtime.qualification) is not IndependentMeasurementQualification
            or runtime.services.feature_provider is not callback
        ):
            raise StageGateError("per-arm feedback lacks its actual independently evaluated strict process owner")
        runtime.verify(required_roles=("feature_provider",))
        if (
            self.inputs.corpus is not runtime.qualification.baseline_admission.corpus
            or self.inputs.scope_sha256 != runtime.qualification.scope.sha256
        ):
            raise StageGateError("per-arm feedback selects another admitted input corpus or complete scope")
        if any(dict(runtime.source_pins).get(path) != digest for path, digest in self.component_source_pins.items()):
            raise StageGateError("per-arm feature inputs/tools are outside qualified measurement membership")

    def _selection(self, compiler, member, corpus, target, scope, context):
        self.verify()
        if corpus is not self.inputs.corpus or scope.sha256 != self.inputs.scope_sha256:
            raise StageGateError("feature arm changed its original input corpus or timer/accuracy/input scope")
        prepared, digest = self.inputs.member(member)
        compiler = plain(compiler)
        if not compiler.is_dir():
            raise StageGateError("feature arm compiler is absent")
        files = self._compiler_members(compiler)
        if not files:
            raise StageGateError("feature arm compiler has no members")
        return (
            {
                "owner_sha256": self.verify(),
                "compiler_sha256": str(hash_tree(compiler)["sha256"]),
                "compiler_consumption_sha256": exact_tree_record(compiler)["sha256"],
                "member_sha256": member.source_sha256,
                "corpus_sha256": corpus.capsules_sha256,
                "member_identity_sha256": document_sha256([member.family, member.capsule, member.source_sha256]),
                "manifest_sha256": corpus.manifest_sha256,
                "target_sha256": pin(target)[1],
                "scope_sha256": scope.sha256,
                "inputs_sha256": digest,
                "context": context,
            },
            prepared,
            files,
        )

    def _compiler_members(self, compiler):
        # The legacy source identity excludes generated directories. The actual
        # process receives the whole selected tree, so bound and pin all members
        # separately before hashing or admitting a cache hit.
        files, byte_count, node_count = [], 0, 0
        for path in compiler.rglob("*"):
            plain(path)
            node_count += 1
            if path.is_file():
                files.append(path)
                byte_count += path.stat().st_size
            elif not path.is_dir():
                raise StageGateError("feature compiler contains unsupported special members")
            if node_count > self.limits.max_nodes or byte_count > self.limits.max_bytes:
                raise StageGateError("feature compiler exceeds its whole byte/member budget")
        return tuple(sorted(files))

    def _argv(self, compiler, source, prepared, request, product):
        mapping = dict(zip(_TOKENS, map(str, (compiler, source, prepared, request, product)), strict=True))
        if self.command.mode == "strict_namespace.v1":
            mapping = _TOKENS
            executable = next(row.destination for row in self.command.runtime if row.source == self.command.executable)
            argv = [
                str(self.command.sandbox),
                "--unshare-all",
                "--die-with-parent",
                "--new-session",
                "--clearenv",
                "--tmpfs",
                "/tmp",
                "--dev",
                "/dev",
            ]
            for row in self.command.runtime:
                argv += ["--ro-bind", str(row.source), row.destination]
            for key, value in self.command.environment:
                argv += ["--setenv", key, value]
            for src, dest in (
                (compiler, "/compiler"),
                (source, "/source.mlir"),
                (prepared, "/inputs.json"),
                (request, "/request.json"),
            ):
                argv += ["--ro-bind", str(src), dest]
            argv += ["--bind", str(product.parent), "/output", "--chdir", "/compiler", "--", executable]
        else:
            argv = [str(self.command.executable)]
        return [*argv, *(mapping.get(token, token) for token in self.command.argv_template)]

    def _reopen(self, saved, selection, owner):
        if (
            type(saved) is not dict
            or set(saved) != {"selection", "record", "product"}
            or document_sha256(saved["selection"]) != document_sha256(selection)
        ):
            raise StageGateError("feature cache belongs to another exact arm/input/domain selection")
        record_path, record_sha = saved["record"]
        product, product_sha = saved["product"]
        for path in (Path(record_path), Path(product)):
            if not plain(path).is_relative_to(plain(owner)):
                raise StageGateError("feature cache producer/product escapes its actual private owner")
        if pin(Path(record_path)) != (record_path, record_sha) or pin(Path(product)) != (product, product_sha):
            raise StageGateError("feature cache producer/product changed")
        record = I.verify(Path(record_path))
        I.require_environment(Path(record_path), environment=dict(self.command.environment))
        if (
            record["kind"] != "subprocess"
            or record["stage"] != "independent_feature_arm"
            or record["executable"]
            != {
                "path": str(self.command.sandbox or self.command.executable),
                "sha256": pin(self.command.sandbox or self.command.executable)[1],
            }
            or {"path": product, "sha256": product_sha} not in record["outputs"]
        ):
            raise StageGateError("feature cache lacks its selected actual native producer")
        original_compiler = plain(Path(record["cwd"]))
        request = Path(product).parent.parent / "request.json"
        if Path(record_path).parent.parent.parent != request.parent or Path(product).name != "observation.json":
            raise StageGateError("feature cache producer is outside its original product owner")
        if document_sha256(_read(request, self.limits)) != document_sha256(selection):
            raise StageGateError("feature cache native request differs from the exact current arm")
        member = next(
            row
            for row in self.inputs.corpus.capsules
            if document_sha256([row.family, row.capsule, row.source_sha256]) == selection["member_identity_sha256"]
        )
        prepared, _input_sha = self.inputs.member(member)
        source = plain(member.source_dir / member.descriptor["interface_mlir"])
        original_files = self._compiler_members(original_compiler)
        expected_inputs = [
            {"path": path, "sha256": digest}
            for path, digest in sorted({pin(path) for path in (*original_files, source, prepared, request)})
        ]
        expected_deps = [{"path": path, "sha256": digest} for path, digest in sorted(set(self.command.source_pins))]
        if (
            record["inputs"] != expected_inputs
            or record["dependencies"] != expected_deps
            or record["argv"] != self._argv(original_compiler, source, prepared, request, Path(product))
            or str(hash_tree(original_compiler)["sha256"]) != selection["compiler_sha256"]
            or exact_tree_record(original_compiler)["sha256"] != selection["compiler_consumption_sha256"]
        ):
            raise StageGateError("feature cache actual tool/argv/source consumption changed")
        observation = _observation(Path(product), self.limits)
        expected = tuple(
            selection[key]
            for key in (
                "compiler_sha256",
                "member_sha256",
                "corpus_sha256",
                "target_sha256",
                "scope_sha256",
                "inputs_sha256",
            )
        )
        actual = tuple(
            getattr(observation, key)
            for key in (
                "compiler_sha256",
                "member_sha256",
                "corpus_sha256",
                "target_sha256",
                "scope_sha256",
                "inputs_sha256",
            )
        )
        if expected != actual:
            raise StageGateError("feature process did not observe the requested exact arm/input scope")
        return observation

    def _arm(self, *, compiler, member, corpus, target, scope, workspace, deadline, context):
        selection, prepared, files = self._selection(compiler, member, corpus, target, scope, context)
        key = document_sha256(selection)
        cache = self.cache_root / (key + ".json")
        if context is not None and (cache.exists() or cache.is_symlink()):
            # A changed product is a miss, never a reused stale observation.
            try:
                observation = self._reopen(_read(cache, self.limits), selection, self.cache_root / "executions")
                if time.monotonic() >= deadline:
                    raise StageGateError("per-arm cached features returned after the original deadline")
                current, _prepared, _files = self._selection(compiler, member, corpus, target, scope, context)
                if document_sha256(current) != document_sha256(selection):
                    raise StageGateError("feature selection changed during cached artifact reopening")
                return observation
            except (ValueError, OSError, StageGateError):
                pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StageGateError("per-arm feature extraction exhausted its original deadline")
        execution_root = workspace
        if context is not None:
            execution_root = self.cache_root / "executions"
            execution_root.mkdir(mode=0o700, exist_ok=True)
            plain(execution_root)
        call = execution_root / uuid.uuid4().hex
        call.mkdir(mode=0o700)
        output = call / "output"
        output.mkdir(mode=0o700)
        request = call / "request.json"
        request.write_text(json.dumps(selection, sort_keys=True) + "\n")
        request.chmod(0o444)
        source = plain(member.source_dir / member.descriptor["interface_mlir"])
        product = output / "observation.json"
        argv = self._argv(compiler, source, prepared, request, product)
        deps = tuple(Path(path) for path, _ in self.command.source_pins)
        native_inputs = (*files, source, prepared, request)
        result = I.run(
            argv,
            directory=call,
            stage="independent_feature_arm",
            cwd=compiler,
            env=dict(self.command.environment),
            inputs=native_inputs,
            outputs=(product,),
            dependencies=deps,
            capture_output=True,
            timeout=min(remaining, 600),
        )
        result.check_returncode()
        if time.monotonic() >= deadline:
            raise StageGateError("per-arm feature extraction returned after its original deadline")
        records = tuple(call.glob("invocations/*/invocation.json"))
        if len(records) != 1:
            raise StageGateError("feature process lacks a unique owned invocation")
        saved = {"selection": selection, "record": pin(records[0]), "product": pin(product)}
        observation = self._reopen(saved, selection, call)
        current, _prepared, _files = self._selection(compiler, member, corpus, target, scope, context)
        if document_sha256(current) != document_sha256(selection):
            raise StageGateError("feature compiler/input/source selection changed during extraction")
        if context is not None:
            temporary = self.cache_root / (key + "." + uuid.uuid4().hex + ".partial")
            with temporary.open("x") as stream:
                json.dump(saved, stream, sort_keys=True)
            temporary.replace(cache)
        return observation

    def pair(
        self, *, baseline, candidate, member, corpus, target_descriptor, scope, workspace, timeout_s, cache_context=None
    ):
        """Actual independent calls; no paired callback conversion or cycle grant."""
        if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 600:
            raise StageGateError("feature arms require a bounded original total deadline")
        self.verify()
        baseline, candidate = plain(baseline), plain(candidate)
        roots = (baseline, candidate, corpus.root, self.inputs.record_path.parent)
        if baseline.is_relative_to(candidate) or candidate.is_relative_to(baseline):
            raise StageGateError("feature compiler arms overlap")
        for path, _digest in self.command.source_pins:
            if any(Path(path).is_relative_to(root) for root in roots):
                raise StageGateError("feature tool/dependency selection grants another compiler or original answers")
        if any(self.cache_root.is_relative_to(root) or root.is_relative_to(self.cache_root) for root in roots):
            raise StageGateError("feature cache overlaps original inputs or compiler arms")
        workspace = plain(workspace)
        if not workspace.is_dir() or any(
            workspace.is_relative_to(root) or root.is_relative_to(workspace) for root in roots
        ):
            raise StageGateError("feature products overlap original compiler/input owners")
        self.cache_root.mkdir(mode=0o700, exist_ok=True)
        self.verify()
        context = cache_context
        encode_feedback_value(context, limits=self.limits)
        deadline = time.monotonic() + timeout_s
        pair = tuple(
            self._arm(
                compiler=compiler,
                member=member,
                corpus=corpus,
                target=target_descriptor,
                scope=scope,
                workspace=workspace,
                deadline=deadline,
                context=context,
            )
            for compiler in (baseline, candidate)
        )
        if pair[0].inputs_sha256 != pair[1].inputs_sha256:
            raise StageGateError("feature arms did not consume the same prepared inputs")
        self.verify()
        if time.monotonic() >= deadline:
            raise StageGateError("per-arm features returned after their original total deadline")
        return pair
