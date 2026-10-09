"""Prepare bounded ordinary operands once, independently of compiler arms.

The live owner binds a complete development corpus, original source, input
policy and preparation products. It grants neither correctness nor physical
traffic, timing, isolation or measurement qualification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from merlin.common import invocation_record as I
from merlin.runtime import commandbuffer, tensor
from merlin.targetgen import capsule_inputs, component_program, input_palette
from merlin_experiments.phase0 import component_execution_budget as B

from . import contracts, feedback_protocol
from . import corpus as C
from .contracts import StageGateError, document_sha256, exact_tree_record, sha256_file
from .feedback_protocol import FeedbackValueLimits, encode_feedback_value

_PREPARED = WeakKeyDictionary()


def plain(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise StageGateError("feature preparation requires canonical unlinked members")
    return path


def pin(path):
    path = plain(path)
    if not path.is_file():
        raise StageGateError("feature preparation requires regular selected files")
    return str(path), sha256_file(path)


def sources():
    return tuple(
        sorted(
            pin(Path(module.__file__).resolve())
            for module in (
                I,
                B,
                C,
                contracts,
                feedback_protocol,
                capsule_inputs,
                component_program,
                input_palette,
                commandbuffer,
                tensor,
            )
        )
    ) + (pin(Path(__file__).resolve()),)


@dataclass(frozen=True, eq=False)
class PreparedFeatureInputs:
    corpus: C.FrozenPerformanceCorpus
    scope_sha256: str
    policy_json: str
    source_pins: tuple[tuple[str, str], ...]
    members: tuple[tuple[str, Path, str], ...]
    record_path: Path
    record_sha256: str
    invocation: Path
    invocation_sha256: str

    @property
    def sha256(self):
        return self.record_sha256

    def _binding(self):
        return document_sha256(
            [
                self.corpus.manifest_sha256,
                self.corpus.capsules_sha256,
                [document_sha256(row.descriptor) for row in self.corpus.capsules],
                self.scope_sha256,
                self.policy_json,
                self.source_pins,
                [(ident, str(path), digest) for ident, path, digest in self.members],
                str(self.record_path),
                self.record_sha256,
                str(self.invocation),
                self.invocation_sha256,
            ]
        )

    def verify(self):
        if _PREPARED.get(self) != self._binding():
            raise StageGateError("feature inputs require actual ordinary preparation, not a saved declaration")
        C.verify_frozen_performance_corpus(self.corpus)
        if sources() != self.source_pins:
            raise StageGateError("feature input preparation source selection changed")
        for path, digest in (
            *self.source_pins,
            (str(self.record_path), self.record_sha256),
            (str(self.invocation), self.invocation_sha256),
            *((str(path), digest) for _member, path, digest in self.members),
        ):
            if pin(Path(path)) != (path, digest):
                raise StageGateError("prepared feature inputs or their source/product records changed")
        record = I.verify(self.invocation)
        if record["kind"] != "python_call" or record["stage"] != "prepare_feature_inputs":
            raise StageGateError("feature inputs lack their original preparation invocation")
        return self.record_sha256

    def member(self, member):
        self.verify()
        if member not in self.corpus.capsules:
            raise StageGateError("feature arm selected a foreign member")
        identity = document_sha256([member.family, member.capsule, member.source_sha256])
        matches = [(path, digest) for ident, path, digest in self.members if ident == identity]
        if len(matches) != 1:
            raise StageGateError("prepared feature input membership is ambiguous")
        return matches[0]


def prepare_feature_inputs(*, corpus, scope, execution_budget, destination, limits):
    """Check the whole roster before realizing any ordinary integer DAG leaves.

    Unsupported source or input domains refuse. This explicit first version
    uses the existing deterministic integer-program input owner; it does not
    replace captured/raw floating inputs, independently checked references, or
    permit new huge tensors merely because their metadata is small.
    """
    from merlin.perf.component_cost import ComponentCostScope

    if type(corpus) is not C.FrozenPerformanceCorpus or type(scope) is not ComponentCostScope:
        raise StageGateError("feature preparation requires its exact corpus and complete scope")
    if type(limits) is not FeedbackValueLimits:
        raise StageGateError("feature preparation requires explicit bounded metadata limits")
    limits.verify()
    B.validate(execution_budget)
    if not corpus.capsules or len(corpus.capsules) > limits.max_nodes:
        raise StageGateError("feature preparation exceeds its whole member roster budget")
    plain(corpus.root)
    plain(corpus.capsules_root)
    if not corpus.capsules_root.is_relative_to(corpus.root):
        raise StageGateError("feature preparation capsule owner escapes its original corpus")
    for member in corpus.capsules:
        if not plain(member.source_dir).is_relative_to(corpus.capsules_root):
            raise StageGateError("feature preparation member escapes its original capsule owner")
    byte_count, node_count = 0, 0
    for path in corpus.capsules_root.rglob("*"):
        plain(path)
        node_count += 1
        if path.is_file():
            byte_count += path.stat().st_size
        elif not path.is_dir():
            raise StageGateError("feature preparation cannot read special members")
        if byte_count > limits.max_bytes or node_count > limits.max_nodes:
            raise StageGateError("feature preparation exceeds its whole source byte/member budget")
    if plain(corpus.manifest_path).stat().st_size > limits.max_bytes:
        raise StageGateError("feature preparation manifest exceeds its metadata budget")
    C.verify_frozen_performance_corpus(corpus)
    root = plain(destination)
    if root.exists() or any(
        root.is_relative_to(path) or path.is_relative_to(root) for path in (corpus.root, corpus.capsules_root)
    ):
        raise StageGateError("feature inputs need a fresh disjoint private destination")
    # Bound the whole declaration and every integer before native source/type
    # arithmetic, then admit aggregate source costs before allocating values.
    encode_feedback_value([row.descriptor for row in corpus.capsules], limits=limits)
    totals = dict.fromkeys(B._METRICS, 0)
    plans, original_inputs = [], []
    for member in corpus.capsules:
        if member.n_bytes > limits.max_bytes:
            raise StageGateError("feature preparation member bytes exceed its declared budget")
        if exact_tree_record(member.source_dir)["sha256"] != member.source_sha256:
            raise StageGateError("feature preparation source member changed")
        capsule = member.descriptor
        if C.CONTRACTS.mapping_file(member.source_dir / "capsule.yaml", yaml_file=True) != capsule:
            raise StageGateError("feature preparation descriptor differs from its actual source member")
        if capsule.get("application_signature_match") or capsule["operation"]["op"] != "component_program":
            raise StageGateError("feature preparation has no selected ordinary input contract for this source")
        source = B.source_for_capsule(capsule)
        cost = B.measure(source)
        if B._exceeded(execution_budget, cost, totals):
            raise StageGateError("feature preparation exceeded whole-roster input/reference budgets")
        for key in totals:
            totals[key] += cost[key]
        name = capsule.get("interface_mlir")
        if type(name) is not str or Path(name).name != name:
            raise StageGateError("feature preparation source interface is not an explicit direct member")
        interface = plain(member.source_dir / name)
        _typed, expected = component_program.render(
            capsule["operation"]["attributes"]["program"],
            operand_dtype=source["program"]["selected_storage"]["operand"],
            accumulator_dtype=source["program"]["selected_storage"]["accumulator"],
        )
        if interface.read_text() != expected:
            raise StageGateError("feature preparation source does not match the original typed DAG")
        plans.append((member, source, cost))
        original_inputs.extend(path for path in member.source_dir.rglob("*") if path.is_file())
    root.mkdir(mode=0o700)
    selected_sources = sources()
    products = tuple(root / (str(index) + ".inputs.json") for index in range(len(plans)))
    record_path = root / "preparation.json"
    with I.observe_call(
        root,
        stage="prepare_feature_inputs",
        function=prepare_feature_inputs,
        arguments={"scope_sha256": scope.sha256, "execution_budget": execution_budget, "limits": limits.__dict__},
        inputs=(corpus.manifest_path, *original_inputs),
        outputs=(*products, record_path),
        dependencies=tuple(Path(path) for path, _ in selected_sources),
    ) as call:
        rows = []
        for (member, source, cost), product in zip(plans, products, strict=True):
            # Use the original ordinary leaf realization, without a compiler,
            # candidate emission, answer document or provider-specific fill.
            context = capsule_inputs._context()
            if any(
                getattr(context, name) is not getattr(capsule_inputs, name)
                for name in ("materialize_capsule_leaves", "capsule_stimulus_range", "Tensor")
            ):
                raise StageGateError("feature inputs cannot inherit another evaluator's realization overrides")
            leaves = capsule_inputs.materialize_capsule_leaves(member.descriptor)
            values = [
                {"name": row["name"], "dtype": row["dtype"], "shape": row["shape"], "values": leaves[row["name"]].data}
                for row in source["program"]["inputs"]
            ]
            encode_feedback_value(values, limits=limits)
            product.write_text(
                json.dumps(
                    {"schema": "merlin.prepared_feature_inputs.v1", "inputs": values}, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
            product.chmod(0o444)
            rows.append(
                (document_sha256([member.family, member.capsule, member.source_sha256]), product, sha256_file(product))
            )
        record = {
            "schema": "merlin.feature_input_preparation.v1",
            "corpus_sha256": corpus.capsules_sha256,
            "manifest_sha256": corpus.manifest_sha256,
            "scope_sha256": scope.sha256,
            "execution_budget": execution_budget,
            "limits": limits.__dict__,
            "aggregate_cost": totals,
            "members": [[ident, str(path), digest] for ident, path, digest in rows],
            "source_pins": selected_sources,
            "scope": "original source/input preparation only; no qualification",
        }
        record_path.write_text(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        call.returned(stdout=json.dumps(record, sort_keys=True))
    result = PreparedFeatureInputs(
        corpus,
        scope.sha256,
        json.dumps(execution_budget, sort_keys=True),
        selected_sources,
        tuple(rows),
        record_path,
        sha256_file(record_path),
        call.path,
        sha256_file(call.path),
    )
    _PREPARED[result] = result._binding()
    result.verify()
    return result
