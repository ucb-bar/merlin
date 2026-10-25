"""Byte-bound Core/Sem source bridge to checked logical region replacement.

This additive leaf issues descriptive evidence only. Authoritative MLIR parsing,
JSON/literal serialization, xDSL semantic-key hashing and file binding are named
host trust components. The kernel proves the actual AST/model correspondence,
execution on initialized inputs and the logical observer theorem. Native primitive,
alias and completion fidelity are deliberately outside that theorem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.verify import lean_graph_regions as base
from merlin.verify import lean_graph_regions_mlir as compiler
from merlin.verify import lean_graph_regions_projection as projection
from merlin.verify import lean_graph_regions_source_ast as ast
from merlin.verify import lean_graph_regions_source_build as builder
from merlin.verify import lean_graph_regions_source_model as model

SCHEMA = "merlin.graph-regions.source-operational.v2"
CLAIMS = {
    "source_root": "Merlin.GraphRegionsSource.checkSource_sound",
    "program_root": "Merlin.GraphRegionsSource.checkSource_program",
    "source_semantics": "MerlinVerify.denote MerlinVerify.Sem.Ops.opSem; exact one-function fuel 2",
    "domain": "initialized, exact-width integer tensor arguments of the full actual signature",
    "projection_trusted_python": True,
    "native_implementation_contract_proved": False,
    "native_alias_completion_proved": False,
    "full_model_or_pipeline_verification": False,
    "shared_acceptance_authority": False,
    "reviewed_contract": False,
}
ROOTS = (
    "Merlin.GraphRegionsSource.primitive_execution",
    "Merlin.GraphRegionsSource.body_execution",
    "Merlin.GraphRegionsSource.generic_execution",
    "Merlin.GraphRegionsSource.stage_execution",
    "Merlin.GraphRegionsSource.sourceMeaning",
    "Merlin.GraphRegionsSource.source_stage_agrees",
    "Merlin.GraphRegionsSource.source_program_execution",
    "Merlin.GraphRegionsSource.entry_typed",
    "Merlin.GraphRegionsSource.program_denote",
    "Merlin.GraphRegionsSource.Program.matches_sound",
    "Merlin.GraphRegionsSource.checkSource_sound",
    "Merlin.GraphRegionsSource.checkSource_program",
    "Merlin.GraphRegions.checkRealization_sound",
)


def save(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def association(original, snapshot):
    first, second = base.fresh_ref(original), base.fresh_ref(snapshot)
    if first.read_bytes() != second.read_bytes():
        raise ValueError("original/snapshot actual content differs")


def oracle_selection(invocation: Path):
    # Content identity follows the symlink, while execution must preserve the
    # selected virtual environment. Running the resolved base Python loses it.
    selected = Path(os.path.abspath(invocation))
    return dict(
        invocation=str(selected),
        executable=base.file_ref(selected),
        configuration=base.file_ref(selected.parent.parent / "pyvenv.cfg"),
    )


def selected_oracle(receipt):
    selection = receipt["oracle_selection"]
    if type(selection) is not dict or set(selection) != {"invocation", "executable", "configuration"}:
        raise ValueError("complete oracle execution selection required")
    path = selection["invocation"]
    if type(path) is not str or not Path(path).is_absolute() or os.path.abspath(path) != path:
        raise ValueError("absolute normalized oracle invocation required")
    base.fresh_ref(selection["executable"])
    base.fresh_ref(selection["configuration"])
    if oracle_selection(Path(path)) != selection or selection["executable"] != receipt["oracle"]:
        raise ValueError("changed oracle invocation/configuration/content association")
    return Path(path)


def source_origin(path: Path, before, after):
    """Bind actual producer source bytes separately from untrusted plan hints."""
    scope = json.loads(path.read_text())
    if scope.get("schema") == "merlin.graph-regions.capture.v1":
        retained = scope["retained_source"]
        refs = list(scope["artifacts"].values())
        refs += [row["file"] for row in scope["production_sources"]]
        refs += list(scope["selected_inputs"].values())
        for role, filename in (("source", "source.mlir"), ("retained_source", "retained-source.mlir")):
            if scope[role] != scope["artifacts"][filename]:
                raise ValueError("producer original/artifact association differs")
    elif scope.get("schema") == "merlin.graph-regions.source-production-probe.v1":
        retained = scope["after"]
        refs = list(scope["production_sources"].values())
        refs += [scope[k] for k in ("source_input", "worker", "python")]
    else:
        raise ValueError("unsupported actual producer origin")
    for ref in refs:
        base.fresh_ref(ref)
    association(scope["source"], before)
    association(retained, after)


def identity_environment(identity_file: Path, *, full=True):
    identity = json.loads(identity_file.read_text())
    lean = base.fresh_ref(identity["lean"])
    if identity["lean"]["sha256"] != base.LEAN_SHA or identity["lean_arguments"] != ["-j1"]:
        raise ValueError("changed source bridge tool selection")
    base.fresh_ref(identity["selected_manifest"])
    base.fresh_ref(identity["builder"])
    for row in identity["project"].values():
        base.fresh_ref(row["source"])
        base.fresh_ref(row["object"])
    if full:
        closure = identity["import_closure"]
        if not closure:
            raise ValueError("missing structural full import closure")
        project_sources = {name: row["source"]["path"] for name, row in identity["project"].items()}
        if set(project_sources) != {row["module"] for row in closure.values() if row["project"] is True}:
            raise ValueError("project/full import closure domains differ")
        for name, row in closure.items():
            if name != row["source"]["path"] or not set(row["dependencies"]) <= set(closure):
                raise ValueError("incomplete structural dependency associations")
            base.fresh_ref(row["source"])
            discovery = json.loads(base.fresh_ref(row["discovery"]).read_text())
            if discovery["source"] != row["source"] or discovery["returncode"] != 0:
                raise ValueError("unbound or failed dependency discovery")
            if discovery["command"] != [str(lean), "-j1", "--src-deps", name]:
                raise ValueError("changed structural dependency parser")
            parsed = list(
                dict.fromkeys(str(Path(p).resolve(strict=True)) for p in discovery["stdout"].splitlines() if p)
            )
            if row["dependencies"] != parsed:
                raise ValueError("dependency graph/discovery output differs")
            if row["project"] is True and project_sources[row["module"]] != name:
                raise ValueError("project source/import association differs")
            for reference in row.get("objects", {}).values():
                base.fresh_ref(reference)
        pending = [project_sources["Merlin.GraphRegionsTensorWitness"]]
        reached = set()
        while pending:
            name = pending.pop()
            if name not in reached:
                reached.add(name)
                pending.extend(closure[name]["dependencies"])
        if reached != set(closure):
            raise ValueError("unreachable or missing full import closure records")
    env = os.environ.copy()
    env["LEAN_PATH"] = os.pathsep.join(
        [str(identity_file.parent.resolve() / "compiled")]
        + [row["library"] for row in identity["packages"] if row["objects_available"]]
    )
    return lean, env


def graph_term(value):
    return (
        "⟨["
        + ",".join(base._node(n) for n in value["nodes"])
        + "],"
        + base._nats(value["functionArguments"])
        + ","
        + base._nats(value["observations"])
        + ","
        + base._strings(value["context"])
        + "⟩"
    )


def export(source: Path, destination: Path, python: Path):
    exporter = repo_root() / "src/merlin/verify/lean/mlir_export.py"
    refs = dict(source=base.file_ref(source), exporter=base.file_ref(exporter), python=base.file_ref(python))
    command = [str(python), "-B", str(exporter), str(source), str(destination)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    save(
        destination.with_suffix(".process.json"),
        dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, inputs=refs),
    )
    result.check_returncode()
    for ref in refs.values():
        base.fresh_ref(ref)
    return json.loads(destination.read_text()), refs


def actual_description(document, source: Path):
    module = projection.parse_source(source.read_text())
    keys = {
        ordinal: hashlib.sha256(projection.ir_text(op).encode()).hexdigest()
        for ordinal, op in enumerate(module.walk())
        if op.name == "linalg.generic"
    }
    proposed = model.propose(document, keys)
    return proposed, model.graph(proposed, projection.ir_text(module))


def numerical_witness(program, arguments):
    """Independent Python integer arithmetic supplies goals, never proof facts."""
    values = {
        pair[0]: value % (1 << pair[1]["width"]) for pair, value in zip(program["arguments"], arguments, strict=True)
    }

    def signed(value, width):
        return value - (1 << width) if value >= (1 << (width - 1)) else value

    def constant(value):
        return value["literal"] % (1 << value["width"])

    widths = dict((pair[0], pair[1]["width"]) for pair in program["arguments"])
    for item in program["items"]:
        if item["kind"] == "constant":
            values[item["value"]["id"]] = constant(item["value"])
        if item["kind"] != "generic":
            continue
        stage, body = item["stage"], item["stage"]["body"]
        local = {pair[0]: values[outer] for pair, outer in zip(body["inputs"], stage["inputs"], strict=True)}
        local.update((capture["id"], constant(capture)) for capture in body["captures"])
        for instruction in body["instructions"]:
            code = instruction["code"]
            args = [local[id] for id in instruction["operands"]]
            if code["kind"] == "constant":
                width, value = code["width"], code["literal"]
            elif code["kind"] == "cast":
                width, value = code["output"], args[0]
                if code["operator"] == "signed":
                    value = signed(value, code["input"])
            else:
                width, (x, y), operator = code["width"], args, code["operator"]
                operations = {
                    "add": lambda: x + y,
                    "sub": lambda: x - y,
                    "mul": lambda: x * y,
                    "band": lambda: x & y,
                    "bor": lambda: x | y,
                    "bxor": lambda: x ^ y,
                    "minsi": lambda: min(signed(x, width), signed(y, width)),
                    "maxsi": lambda: max(signed(x, width), signed(y, width)),
                }
                value = operations[operator]()
            local[instruction["result"]] = value % (1 << width)
        values[stage["result"]], widths[stage["result"]] = local[body["returned"]], body["width"]
    return [[widths[id], values[id]] for id in program["returned"]]


def proof_text(
    before,
    after,
    program,
    graph,
    retained,
    *,
    expected,
    refusal,
    plan=None,
    selection=None,
    fused=None,
    graph_expected=True,
    witness=None,
):
    text = (
        "import Merlin.GraphRegionsTensorWitness\n"
        "open MerlinVerify Merlin.GraphRegionsSource Merlin.GraphRegions\n"
        "set_option maxHeartbeats 2000000\nset_option maxRecDepth 8000\n"
        f"def actualBeforeModule : MerlinVerify.Module := {ast.module(before)}\n"
        f"def actualAfterModule : MerlinVerify.Module := {ast.module(after)}\n"
        f"def actualProgram : Merlin.GraphRegionsSource.Program := {model.program_term(program)}\n"
        f"def actualGraph : Graph := {graph_term(graph)}\n"
        f"def actualRetainedGraph : Graph := {graph_term(retained)}\n"
    )
    source_goal = "checkSource actualBeforeModule actualAfterModule actualProgram actualGraph"
    text += f"theorem actualSourceVerdict : {source_goal} = {str(expected).lower()} := by\n"
    if expected:
        text += "  apply checkSource_iff.mpr\n  exact ⟨⟨⟨rfl, rfl⟩, by decide⟩, by decide⟩\n"
    elif refusal == "after":
        text += "  apply checkSource_refuse_changed\n  decide\n"
    elif refusal in ("static", "graph"):
        text += "  apply Bool.eq_false_iff.mpr\n  intro accepted\n"
        if refusal == "static":
            text += "  exact (by decide : ¬actualProgram.Valid) (checkSource_iff.mp accepted).1.2\n"
        else:
            text += (
                "  have failed : actualProgram.graphMatches actualGraph = false := by decide\n"
                "  have passed := (checkSource_iff.mp accepted).2\n  simp_all\n"
            )
    else:
        raise ValueError("explicit semantic refusal strategy required")
    audit_names = ["actualSourceVerdict", *ROOTS]
    if expected:
        text += (
            "def actualResults (values : List Tensor) := actualGraph.observations.map\n"
            "  (executeNodes (sourceMeaning actualProgram.entries actualProgram.types) actualGraph.nodes\n"
            "    (entryState actualProgram.types actualProgram.arguments values))\n"
            "theorem actualSourceExecution (values : List Tensor)\n"
            "    (domain : InputsDomain actualProgram.arguments values) (ctx : Ctx) :\n"
            "    actualBeforeModule = actualAfterModule ∧\n"
            "    denote MerlinVerify.Sem.Ops.opSem ctx actualBeforeModule actualProgram.name\n"
            "      (values.map RVal.tensor) =\n"
            "      .ok ((actualResults values).map RVal.tensor) ∧\n"
            "      ResultsDefined ((actualResults values).map RVal.tensor) :=\n"
            "  checkSource_sound actualSourceVerdict values domain ctx\n"
        )
        audit_names.append("actualSourceExecution")
        if witness is not None:
            samples = numerical_witness(program, witness)
            text += (
                f"def witnessValues : List Nat := {base._nats(witness)}\n"
                "def witnessInputs : List Tensor := (actualProgram.arguments.zip witnessValues).map\n"
                "  fun (pair, value) => filled pair.2 value\n"
                "theorem witnessDomain : InputsDomain actualProgram.arguments witnessInputs :=\n"
                "  initialized_inputs _ _ (by decide)\n"
                "theorem witnessExecution (ctx : Ctx) :\n"
                "    denote MerlinVerify.Sem.Ops.opSem ctx actualBeforeModule actualProgram.name\n"
                "      (witnessInputs.map RVal.tensor) =\n"
                "      .ok ((actualResults witnessInputs).map RVal.tensor) ∧\n"
                "      ResultsDefined ((actualResults witnessInputs).map RVal.tensor) :=\n"
                "  (actualSourceExecution witnessInputs witnessDomain ctx).2\n"
            )
            point = [0] * len(program["arguments"][0][1]["shape"])
            result = ast.sequence(samples, lambda pair: f"(.some (.bits {pair[0]} (BitVec.ofNat {pair[0]} {pair[1]})))")
            text += (
                f"theorem witnessPoints : (actualResults witnessInputs).map\n"
                f"    (fun tensor => tensor.get {base._nats(point)}) =\n"
                f"    {result} := by decide\n"
            )
            audit_names.extend(("witnessDomain", "witnessExecution", "witnessPoints"))
    if plan is not None:
        if not expected or fused is None or not selection:
            raise ValueError("logical realization requires an admitted source and actual fused body")
        start = next(i for i, n in enumerate(graph["nodes"]) if n["ordinal"] == selection[0])
        text += (
            f"def actualPlan : Plan := {base._plan(plan)}\n"
            f"def actualFusion : FusedBody := ⟨[{','.join(base._node(n) for n in fused['body'])}],"
            f"{base._nats(fused['inputs'])},{base._nats(fused['outputs'])}⟩\n"
            "theorem actualGraphVerdict : checkRealization actualGraph actualRetainedGraph "
            f"{start} {len(selection)} {base._nats(selection)} actualPlan actualFusion\n"
            f"    = {str(graph_expected).lower()} := by decide\n"
        )
        audit_names.append("actualGraphVerdict")
        if graph_expected:
            text += (
                "theorem actualProgramCorrespondence (values : List Tensor)\n"
                "    (domain : InputsDomain actualProgram.arguments values) (ctx : Ctx) :\n"
                "    denote MerlinVerify.Sem.Ops.opSem ctx actualBeforeModule actualProgram.name\n"
                "      (values.map RVal.tensor) =\n"
                "    .ok ((actualGraph.observations.map\n"
                f"      (executeNodes (sourceMeaning actualProgram.entries actualProgram.types)\n"
                f"        (actualGraph.nodes.drop ({start} + {len(selection)}))\n"
                "        (actualFusion.execute (sourceMeaning actualProgram.entries actualProgram.types)\n"
                f"          (executeNodes (sourceMeaning actualProgram.entries actualProgram.types)\n"
                f"            (actualGraph.nodes.take {start})\n"
                "            (entryState actualProgram.types actualProgram.arguments values))))).map RVal.tensor) :=\n"
                "  checkSource_program actualSourceVerdict actualGraphVerdict values domain ctx\n"
            )
            audit_names.append("actualProgramCorrespondence")
    text += "".join(f"#print axioms {name}\n#check {name}\n" for name in audit_names)
    return text, audit_names


def audit(stdout, names):
    found = {}
    for line in stdout.splitlines():
        if "depends on axioms:" in line:
            name, _, rest = line.partition(" depends on axioms:")
            axioms = set(rest.strip().removeprefix("[").removesuffix("]").replace(",", " ").split())
            if not axioms <= base.ALLOWED_AXIOMS:
                raise ValueError("unapproved source theorem dependency")
            found[name.strip("'")] = sorted(axioms)
        elif "does not depend on any axioms" in line:
            found[line.split("'")[1]] = []
    if set(found) != set(names):
        raise ValueError("incomplete source theorem dependency audit")
    return found


def verify(
    before: Path,
    after: Path,
    destination: Path,
    *,
    identity: Path,
    oracle: Path,
    expected=True,
    refusal=None,
    proposed: Path | None = None,
    graph_input: Path | None = None,
    plan: Path | None = None,
    fused: Path | None = None,
    selection=None,
    graph_expected=True,
    witness=None,
    origin: Path | None = None,
):
    if type(expected) is not bool or type(graph_expected) is not bool:
        raise ValueError("strict goal Boolean required; goals are not trusted verdict facts")
    destination.mkdir(parents=True, exist_ok=False)
    lean, environment = identity_environment(identity)
    oracle_binding = oracle_selection(oracle)
    inputs = dict(before=before, after=after)
    for role, path in (
        ("proposed", proposed),
        ("graph_input", graph_input),
        ("plan", plan),
        ("fused", fused),
        ("origin", origin),
    ):
        if path is not None:
            inputs[role] = path
    originals, snapshots = {}, {}
    for role, path in inputs.items():
        originals[role] = base.file_ref(path)
        snapshot = destination / (role + path.suffix)
        snapshot.write_bytes(base.fresh_ref(originals[role]).read_bytes())
        snapshots[role] = base.file_ref(snapshot)
        association(originals[role], snapshots[role])
    if origin is not None:
        source_origin(base.fresh_ref(snapshots["origin"]), snapshots["before"], snapshots["after"])
    documents, exporter_refs, compiler_refs = {}, {}, {}
    for role in ("before", "after"):
        path = base.fresh_ref(snapshots[role])
        validity = compiler.compiler_verify(path, destination / (role + "-compiler"), python=oracle)
        if validity["oracle"]["valid"] is not True:
            raise ValueError("compiler-invalid source; not a Lean semantic refusal")
        compiler_refs[role] = base.file_ref(destination / (role + "-compiler") / "EVIDENCE.json")
        documents[role], exporter_refs[role] = export(path, destination / (role + "-export.json"), oracle)
        if documents[role]["source"] != {key: snapshots[role][key] for key in ("sha256", "bytes")}:
            raise ValueError("authoritative export/source bytes differ")
    proposed_model, graph = actual_description(documents["before"], base.fresh_ref(snapshots["before"]))
    if proposed is not None:
        proposed_model = json.loads(base.fresh_ref(snapshots["proposed"]).read_text())
    if graph_input is not None:
        graph = json.loads(base.fresh_ref(snapshots["graph_input"]).read_text())
    _, retained = actual_description(documents["after"], base.fresh_ref(snapshots["after"]))
    save(destination / "model.json", proposed_model)
    save(destination / "graph.json", graph)
    save(destination / "retained-graph.json", retained)
    text, names = proof_text(
        documents["before"],
        documents["after"],
        proposed_model,
        graph,
        retained,
        expected=expected,
        refusal=refusal,
        plan=json.loads(base.fresh_ref(snapshots["plan"]).read_text()) if plan else None,
        fused=json.loads(base.fresh_ref(snapshots["fused"]).read_text()) if fused else None,
        selection=selection,
        graph_expected=graph_expected,
        witness=witness,
    )
    proof = destination / "proof.lean"
    proof.write_text(text)
    result = subprocess.run([str(lean), "-j1", str(proof)], env=environment, capture_output=True, text=True, timeout=60)
    save(
        destination / "PROCESS.json",
        dict(
            command=[str(lean), "-j1", str(proof)],
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        ),
    )
    (destination / "audit.stdout").write_text(result.stdout)
    result.check_returncode()
    axioms = audit(result.stdout, names)
    for role in inputs:
        association(originals[role], snapshots[role])
    identity_environment(identity)
    receipt = dict(
        schema=SCHEMA,
        claims=CLAIMS,
        source_accepted=expected,
        logical_program_correspondence=expected and plan is not None and graph_expected,
        graph_accepted=graph_expected if plan is not None else None,
        originals=originals,
        snapshots=snapshots,
        identity=base.file_ref(identity),
        exporter_inputs=exporter_refs,
        compiler_evidence=compiler_refs,
        primary_python=base.file_ref(Path(sys.executable)),
        proof=base.file_ref(proof),
        audit=base.file_ref(destination / "audit.stdout"),
        model=base.file_ref(destination / "model.json"),
        graph=base.file_ref(destination / "graph.json"),
        export_before=base.file_ref(destination / "before-export.json"),
        export_after=base.file_ref(destination / "after-export.json"),
        host_sources={
            key: base.file_ref(Path(module.__file__))
            for key, module in dict(
                checker=__import__(__name__, fromlist=["*"]),
                ast=ast,
                model=model,
                builder=builder,
                projection=projection,
                base=base,
                compiler=compiler,
            ).items()
        },
        axiom_policy=sorted(base.ALLOWED_AXIOMS),
        theorem_audit=axioms,
        parameters=dict(
            expected=expected, refusal=refusal, selection=selection, graph_expected=graph_expected, witness=witness
        ),
        oracle=base.file_ref(oracle),
        oracle_selection=oracle_binding,
    )
    save(destination / "EVIDENCE.json", receipt)
    return receipt


def validate(receipt):
    claims = receipt.get("claims")
    if (
        receipt.get("schema") != SCHEMA
        or type(claims) is not dict
        or set(claims) != set(CLAIMS)
        or any(type(claims[key]) is not type(value) or claims[key] != value for key, value in CLAIMS.items())
    ):
        raise ValueError("changed source claim scope")
    if receipt.get("axiom_policy") != sorted(base.ALLOWED_AXIOMS):
        raise ValueError("changed axiom policy")
    selected_oracle(receipt)
    for key in ("source_accepted", "logical_program_correspondence"):
        if type(receipt.get(key)) is not bool:
            raise ValueError("strict verdict type required")
    if set(receipt["originals"]) != set(receipt["snapshots"]):
        raise ValueError("complete original/snapshot role inventory required")
    if not {"before", "after"} <= set(receipt["originals"]) or not set(receipt["originals"]) <= {
        "before",
        "after",
        "proposed",
        "graph_input",
        "plan",
        "fused",
        "origin",
    }:
        raise ValueError("changed source role domain")
    for role, original in receipt["originals"].items():
        association(original, receipt["snapshots"][role])
    if "origin" in receipt["snapshots"]:
        source_origin(
            base.fresh_ref(receipt["snapshots"]["origin"]),
            receipt["snapshots"]["before"],
            receipt["snapshots"]["after"],
        )
    for key in ("proof", "audit", "model", "graph", "export_before", "export_after", "oracle", "identity"):
        base.fresh_ref(receipt[key])
    if receipt["primary_python"] != base.file_ref(Path(sys.executable)):
        raise ValueError("changed primary Python interpreter")
    exporter = repo_root() / "src/merlin/verify/lean/mlir_export.py"
    for role in ("before", "after"):
        refs = receipt["exporter_inputs"][role]
        expected = dict(source=receipt["snapshots"][role], exporter=base.file_ref(exporter), python=receipt["oracle"])
        if refs != expected:
            raise ValueError("changed exporter/source/oracle association")
        for ref in refs.values():
            base.fresh_ref(ref)
        document = json.loads(base.fresh_ref(receipt["export_" + role]).read_text())
        if document["source"] != {key: refs["source"][key] for key in ("sha256", "bytes")}:
            raise ValueError("export/source content association differs")
        evidence = json.loads(base.fresh_ref(receipt["compiler_evidence"][role]).read_text())
        if evidence["original"] != refs["source"] or evidence["oracle"]["python_executable"] != receipt["oracle"]:
            raise ValueError("changed authoritative verifier/source/tool association")
        association(evidence["original"], evidence["snapshot"])
        if evidence["oracle"]["valid"] is not True:
            raise ValueError("no compiler-valid source evidence")
        for ref in evidence["oracle"]["native_modules"].values():
            base.fresh_ref(ref)
        base.fresh_ref(evidence["oracle"]["ir_module"])
        if evidence["worker"] != base.file_ref(Path(compiler.__file__)):
            raise ValueError("changed compiler verifier worker")
    modules = dict(
        checker=__import__(__name__, fromlist=["*"]),
        ast=ast,
        model=model,
        builder=builder,
        projection=projection,
        base=base,
        compiler=compiler,
    )
    if receipt["host_sources"] != {key: base.file_ref(Path(mod.__file__)) for key, mod in modules.items()}:
        raise ValueError("changed or incomplete host source selection")
    identity_environment(base.fresh_ref(receipt["identity"]))


def replay(receipt, destination: Path, *, fresh_identity: Path | None = None):
    validate(receipt)
    if fresh_identity is None:
        build_path = destination.parent / (destination.name + "-fresh-build")
        builder.build(build_path, ["Merlin.GraphRegionsTensorWitness"])
        fresh_identity = build_path / "IDENTITY.json"
    old = json.loads(base.fresh_ref(receipt["identity"]).read_text())
    new = json.loads(fresh_identity.read_text())
    for key in ("lean", "lean_arguments", "selected_manifest", "packages", "builder"):
        if old[key] != new[key]:
            raise ValueError("changed fresh build selection")
    if set(old["project"]) != set(new["project"]) or set(old["import_closure"]) != set(new["import_closure"]):
        raise ValueError("changed full import domain")
    for name, row in old["project"].items():
        if row["source"] != new["project"][name]["source"]:
            raise ValueError("changed project source")
        for field in ("sha256", "bytes"):
            if row["object"][field] != new["project"][name]["object"][field]:
                raise ValueError("fresh project object content differs")
    for name, row in old["import_closure"].items():
        repeated = new["import_closure"][name]
        for field in ("source", "module", "project", "dependencies", "objects"):
            if row.get(field) != repeated.get(field):
                raise ValueError("fresh transitive import association differs")
    originals = {key: base.fresh_ref(ref) for key, ref in receipt["originals"].items()}
    repeated = verify(
        originals["before"],
        originals["after"],
        destination,
        identity=fresh_identity,
        oracle=selected_oracle(receipt),
        proposed=originals.get("proposed"),
        graph_input=originals.get("graph_input"),
        plan=originals.get("plan"),
        fused=originals.get("fused"),
        origin=originals.get("origin"),
        **receipt["parameters"],
    )
    for key in (
        "claims",
        "source_accepted",
        "logical_program_correspondence",
        "graph_accepted",
        "theorem_audit",
        "oracle_selection",
    ):
        if type(receipt[key]) is not type(repeated[key]) or receipt[key] != repeated[key]:
            raise ValueError("replayed operational claim differs")
    for key in ("proof", "audit", "model", "graph", "export_before", "export_after"):
        if base.fresh_ref(receipt[key]).read_bytes() != base.fresh_ref(repeated[key]).read_bytes():
            raise ValueError("replayed operational bytes differ: " + key)
    for role in ("before", "after"):
        previous = json.loads(base.fresh_ref(receipt["compiler_evidence"][role]).read_text())
        current = json.loads(base.fresh_ref(repeated["compiler_evidence"][role]).read_text())
        if previous["oracle"] != current["oracle"]:
            raise ValueError("replayed authoritative parser environment differs")
    validate(receipt)
    return repeated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = replay(json.loads(args.receipt.read_text()), args.destination)
    print(
        json.dumps(
            dict(
                source_accepted=result["source_accepted"],
                logical_program_correspondence=result["logical_program_correspondence"],
            )
        )
    )


if __name__ == "__main__":
    main()
