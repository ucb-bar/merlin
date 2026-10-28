"""Bind composition sources to the existing truth, placement and trace gates."""

from __future__ import annotations

from pathlib import Path
import copy

from merlin.targetgen.composition_corpus import coverage as coverage


def prepare_entries(entries):
    from merlin.targetgen.functional_truth_compositions import expectation

    result = []
    for original in entries:
        if original.get("op") != "composition_program":
            result.append(original)
            continue
        entry = copy.deepcopy(original)
        semantic = entry["generalization"]
        if entry.get("validated_graph"):
            children = []
            for node in semantic["validated_node_semantics"]:
                node["composition_truth"] = expectation(
                    node["composition_chain"],
                    node["quantization_scheme"],
                    node["composition_instance"],
                    facts=node["composition_facts"],
                )
                children.append(node["composition_truth"])
            shared_modes = (
                {"shared_input"} if semantic["fusion_validation"]["graph"]["structure"] == "shared_input" else set()
            )
            commitment = copy.deepcopy(children[0])
            commitment.update(
                reasons=[reason for child in children for reason in child["reasons"]],
                required_modes=sorted(shared_modes | {mode for child in children for mode in child["required_modes"]}),
                members=[f"node{i}_{member}" for i, child in enumerate(children) for member in child["members"]],
            )
            commitment["outcome"] = "refuse" if commitment["reasons"] else "pass"
            commitment["lane"] = "must_refuse" if commitment["reasons"] else "fused_device"
            semantic["composition_truth"] = commitment
            if commitment["outcome"] == "refuse":
                raise ValueError(f"admitted nodes fail current composition truth: {commitment['reasons']}")
            result.append(entry)
            continue
        commitment = expectation(
            semantic["composition_chain"],
            semantic["quantization_scheme"],
            semantic["composition_instance"],
            facts=semantic["composition_facts"],
        )
        semantic["composition_truth"] = commitment
        if commitment["outcome"] == "refuse":
            entry.update(
                outcome="refuse",
                refusal=dict(
                    stage="lowering",
                    reason="; ".join(commitment["reasons"]),
                    evidence="resolved composition constraints",
                ),
            )
            semantic["must_accelerate"] = False
        result.append(entry)
    return result


def bind_capsule(cap, source):
    """Freeze actual top-level source ordinals, never counts or invented members."""
    from merlin.common import mlir_query as mq

    if cap.get("operation", {}).get("op") != "composition_program":
        return cap
    if "validated_graph" in cap["operation"]["attributes"]:
        return _bind_validated_capsule(cap, source)
    module = mq.parse(source)
    function = next(op for op in module.body.block.ops if mq.op_name(op) == "func.func")
    operations = [op for op in function.body.block.ops if mq.op_name(op) != "func.return"]
    stages = {}
    for index, operation in enumerate(operations):
        marker = operation.attributes.get("prov.composition_stage")
        if marker is not None:
            stages[marker.data] = [index]
    if not stages:
        raise ValueError("composition source has no semantic stage bindings")
    return _bind_stages(cap, stages)


def _bind_stages(cap, stages):
    semantic = cap["semantic"]
    commitment = semantic["composition_truth"]
    producer = semantic["composition_instance"]["producer"]
    prefixes = {"contraction": "mac", "operand_sum": "operand_sum", "movement": "readout"}
    producers = [i for name, indices in stages.items() if prefixes[producer] in name for i in indices]
    if not producers:
        raise ValueError("composition source has no bound producer")
    expected = cap.setdefault("expected", {})
    expected["composition"] = dict(
        source_op_indices=sorted(i for indices in stages.values() for i in indices),
        producer_source_op_indices=producers,
        stages=stages,
        producer=producer,
    )
    modes = {mode: True for mode in commitment["required_modes"]}
    if semantic["composition_instance"]["exit"] == "requant":
        modes["readout_multipliers"] = True
    if "activation" in semantic["composition_instance"]["readout_stages"]:
        modes["relu"] = True
    if semantic["composition_instance"].get("siblings"):
        modes["i32_exit"] = True
    expected["modes"] = modes
    expected["stage_scoped_modes"] = {name: dict(modes) for name in stages}
    case = semantic["composition_instance"]
    params = commitment["instruction_expectation"]
    parameters = {}
    if "readout_multipliers" in modes:
        rows = cap["operation"]["attributes"]["M"]
        if "pool" in stages:
            side = case["pool_in_dims"][0]
            rows = ((side + 2 * case["padding"][0] - case["pool_window"][0]) // case["pool_stride"][0] + 1) ** 2
        tile = semantic["composition_facts"]["tile_dim"]
        stores = (rows + tile - 1) // tile
        parameters["readout_multipliers"] = dict(
            acc_scale_bits=[bits for bits in params["readout_multiplier_bits"] for _ in range(stores)]
        )
        if "runtime_scale_input" in params:
            name = params["runtime_scale_input"]
            inputs = [index for index, leaf in enumerate(cap["inputs"]) if leaf["name"] == name]
            if len(inputs) != 1:
                raise ValueError("runtime readout scale must bind one capsule input")
            parameters["readout_multipliers"]["input"] = dict(kind="argbase", arg_index=inputs[0], offset=0)
        if semantic["quantization_scheme"]["weight_scale_granularity"] == "column_block":
            modes["column_block"] = True
            parameters["column_block"] = dict(
                acc_scale_bits=params["readout_multiplier_bits"],
                stores_per_block=[stores] * len(params["readout_multiplier_bits"]),
            )
    if "accumulator_preload" in modes:
        seeds = [index for index, row in enumerate(cap["inputs"]) if row["role"] == "bias"]
        if not seeds and "constant_seed" in case:
            # Execution must independently bind the linked literal's address before audit credit.
            parameters["accumulator_preload"] = dict(constant_seed=copy.deepcopy(case["constant_seed"]))
        elif len(seeds) != 1:
            raise ValueError("UNKNOWN: composition needs one explicit accumulator seed operand")
        else:
            parameters["accumulator_preload"] = dict(input=dict(kind="argbase", arg_index=seeds[0], offset=0))
    if "scaled_accumulate_load" in modes:
        parameters["scaled_accumulate_load"] = dict(scale_bits=params["operand_multiplier_bits"])
    if "pool" in stages:
        side = case["pool_in_dims"][0]
        window, stride = case["pool_window"][0], case["pool_stride"][0]
        output = (side + 2 * case["padding"][0] - window) // stride + 1
        parameters["pooled_store"] = dict(
            fields=dict(
                pool_stride=stride,
                pool_size=window,
                pool_out_dim=output,
                porows=output,
                pocols=output,
                orows=side,
                ocols=side,
                upad=case["padding"][0],
                lpad=case["padding"][1],
            )
        )
    if "k_extension" in modes and commitment["outcome"] != "refuse":
        from merlin.xdsl_dialects.lowering.k_extension import lower_k_extension

        facts = semantic["composition_facts"]
        extension = lower_k_extension(
            case["coefficient"],
            source_k=case["source_k"],
            residual_width=case["residual_width"],
            digit_bits=facts["operand_signed_bits"],
            max_extended_k=facts["k_extension_within"]["max_k"],
            accumulator_bits=facts["k_extension_within"]["partial_sum_bits"],
            source_bound=case["source_partial_bound"],
            residual_range=case["residual_range"],
        )
        parameters["k_extension"] = dict(
            source_k=case["source_k"], extension_k=extension["extended_k"] - case["source_k"]
        )
    if "shared_input" in modes:
        # Logical name only; the trace checker also needs the execution owner's
        # source-to-decoded-operand binding before it can award coverage.
        parameters["shared_input"] = dict(input=dict(kind="argbase", arg_index=0, offset=0))
    if case.get("siblings"):
        narrow, full = [], []
        for index, sibling in enumerate(case["siblings"]):
            operand = dict(kind="argbase", arg_index=len(cap["inputs"]) + index, offset=0)
            (narrow if sibling["exit"] == "requant" else full).append(operand)
        parameters["readout_multipliers"]["outputs"] = narrow
        if "column_block" in parameters:
            parameters["column_block"]["outputs"] = narrow
        parameters["i32_exit"] = dict(outputs=full)
    expected["stage_scoped_modes"] = {name: dict(modes) for name in stages}
    expected["mode_parameters"] = parameters
    semantic["grading"] = "placement_and_instruction_audit"
    return cap


def _bind_validated_capsule(cap, source):
    from merlin.common import mlir_query as mq

    module = mq.parse(source)
    function = next(op for op in module.body.block.ops if mq.op_name(op) == "func.func")
    operations = [op for op in function.body.block.ops if mq.op_name(op) != "func.return"]
    attrs = cap["operation"]["attributes"]
    shared = attrs["validated_structure"] == "shared_input"
    global_inputs = {leaf["name"]: i for i, leaf in enumerate(cap["inputs"])}
    global_outputs = {name: len(cap["inputs"]) + i for i, name in enumerate(attrs["outputs"])}
    stages, producers, scoped_modes, scoped_parameters = {}, [], {}, {}
    for index, original in enumerate(attrs["validated_graph"]):
        prefix = f"node{index}_"
        node = copy.deepcopy(original)
        node["semantic"] = cap["semantic"]["validated_node_semantics"][index]
        local_stages = {}
        for ordinal, operation in enumerate(operations):
            marker = operation.attributes.get("prov.trace_role")
            if marker is not None and marker.data.startswith(prefix):
                local_stages.setdefault(marker.data[len(prefix) :], []).append(ordinal)
        if not local_stages:
            raise ValueError("validated node has no source stage bindings")
        _bind_stages(node, local_stages)
        expected = node["expected"]
        parameters = expected["mode_parameters"]
        if shared:
            parameters["shared_input"] = dict(input=dict(kind="argbase", arg_index=global_inputs["A"], offset=0))
        outputs = [global_outputs[prefix + name] for name in node["operation"]["attributes"]["outputs"]]
        mode = "readout_multipliers" if node["operation"]["attributes"]["case"]["exit"] == "requant" else "i32_exit"
        parameters.setdefault(mode, {})["outputs"] = [dict(kind="argbase", arg_index=i, offset=0) for i in outputs]
        mapping = {
            i: global_inputs[leaf["name"] if not shared or leaf["name"] == "A" else prefix + leaf["name"]]
            for i, leaf in enumerate(node["inputs"])
        }

        for name, parameter in parameters.items():
            if "input" in parameter and name != "shared_input":
                operand = parameter["input"]
                operand["arg_index"] = mapping[operand["arg_index"]]
        producers.extend(expected["composition"]["producer_source_op_indices"])
        for stage, members in local_stages.items():
            name = prefix + stage
            stages[name] = members
            scoped_modes[name] = {**expected["modes"], **({"shared_input": True} if shared else {})}
            scoped_parameters[name] = copy.deepcopy(parameters)
    cap["expected"].update(
        composition=dict(
            source_op_indices=sorted({i for members in stages.values() for i in members}),
            producer_source_op_indices=sorted(producers),
            stages=stages,
            producer=cap["semantic"]["composition_instance"]["producer"],
        ),
        modes=dict.fromkeys(cap["semantic"]["composition_truth"]["required_modes"], True),
        stage_scoped_modes=scoped_modes,
        stage_scoped_mode_parameters=scoped_parameters,
    )
    cap["semantic"]["grading"] = "placement_and_instruction_audit"
    return cap


def _obligations(entry):
    """What one declared composition capsule covers: its chain, scheme, shape class and their pairings."""
    generalization = entry["generalization"]
    instance = generalization["composition_instance"]
    chain = generalization["composition_chain_id"]
    scheme = generalization["quantization_scheme_id"]
    shape = generalization["shape_class"]
    scale = "runtime_input" if "runtime_scale_input" in instance else "static"
    return {
        f"chain:{chain}",
        f"scheme:{scheme}",
        f"shape_class:{shape}",
        f"chain+scheme:{chain}+{scheme}",
        f"chain+shape_class:{chain}+{shape}",
        f"scheme+shape_class:{scheme}+{shape}",
        f"producer:{instance['producer']}",
        f"scale_source:{scale}",
    }


def sample_declared(demanded, *, budget, workload_count):
    """Choose the declared compositions that fit the budget left by the workloads, deterministically.

    The ordinary synthesis budget is never raised and the declared set is never silently cut: when every
    declared composition fits all are kept, otherwise the shortfall is a recorded sample. Each pick covers
    the most chain, scheme and shape-class obligations not yet covered, ties going to the chain with the
    fewest picks and then the name, so the sample spreads across the declared chains.
    """
    from merlin.common.jsonio import canonical_sha256
    from merlin.targetgen.corpus_synth import SynthesisError

    remaining = budget - workload_count
    if remaining < 1:
        raise SynthesisError(
            f"{workload_count} workload capsules leave no room for compositions in a budget of {budget}"
        )
    pool = sorted(demanded, key=lambda entry: entry["name"])
    obligations = {entry["name"]: _obligations(entry) for entry in pool}
    everything = set().union(*obligations.values()) if pool else set()
    allocation = min(len(pool), remaining)
    chosen, covered, per_chain = [], set(), {}
    while len(chosen) < allocation:
        options = [entry for entry in pool if entry["name"] not in {row["name"] for row in chosen}]
        entry = min(
            options,
            key=lambda e: (
                -len(obligations[e["name"]] - covered),
                per_chain.get(e["generalization"]["composition_chain_id"], 0),
                e["name"],
            ),
        )
        gain = sorted(obligations[entry["name"]] - covered)
        chosen.append(dict(name=entry["name"], new_obligations=gain))
        covered |= obligations[entry["name"]]
        chain = entry["generalization"]["composition_chain_id"]
        per_chain[chain] = per_chain.get(chain, 0) + 1
    picked = {row["name"] for row in chosen}
    by_name = {entry["name"]: entry for entry in pool}
    record = dict(
        schema="merlin.composition_selection.v1",
        policy="all_declared_fit" if allocation == len(pool) else "sample_to_budget",
        budget=budget,
        workload_count=workload_count,
        workload_rule="capsules other than performance companions (cat '_perf') and composition programs",
        declared_count=len(pool),
        selected_count=len(chosen),
        selected_plus_workload=len(chosen) + workload_count,
        pool_sha256=canonical_sha256([[entry["name"], sorted(obligations[entry["name"]])] for entry in pool]),
        selected=[
            dict(
                name=row["name"],
                chain=by_name[row["name"]]["generalization"]["composition_chain_id"],
                scheme=by_name[row["name"]]["generalization"]["quantization_scheme_id"],
                shape_class=by_name[row["name"]]["generalization"]["shape_class"],
                new_obligations=row["new_obligations"],
            )
            for row in sorted(chosen, key=lambda row: row["name"])
        ],
        unselected=sorted(entry["name"] for entry in pool if entry["name"] not in picked),
        obligations_total=len(everything),
        obligations_uncovered=sorted(everything - covered),
        chains={
            chain: dict(
                declared=sum(1 for e in pool if e["generalization"]["composition_chain_id"] == chain),
                selected=per_chain.get(chain, 0),
            )
            for chain in sorted({e["generalization"]["composition_chain_id"] for e in pool})
        },
    )
    return [by_name[row["name"]] for row in sorted(chosen, key=lambda row: row["name"])], record


def append_declared(entries, software_spec, facts, *, budget, selection_path=None, selection_sink=None):
    """Add the selected composition axis before any release source is written."""
    from merlin.targetgen.composition_corpus import entries as composition_entries
    from merlin.targetgen.corpus_synth import SynthesisError

    if not (software_spec or {}).get("compositions"):
        return entries
    demanded = composition_entries(software_spec, facts)
    by_name = {entry["name"]: entry for entry in entries}
    if len(by_name) != len(entries):
        raise ValueError("release has duplicate capsule identities")
    for entry in demanded:
        existing = by_name.get(entry["name"])
        if existing is not None and existing != entry:
            raise ValueError("selected composition capsule differs from its deterministic derivation")
    workload_count = sum(
        1 for entry in entries if entry.get("cat") != "_perf" and entry.get("op") != "composition_program"
    )
    chosen, record = sample_declared(demanded, budget=budget, workload_count=workload_count)
    if selection_path is not None:
        from merlin.common.jsonio import write_canonical_json

        write_canonical_json(Path(selection_path), record)
    if selection_sink is not None:
        selection_sink.update(record)
    for entry in chosen:
        by_name[entry["name"]] = entry
    result = list(by_name.values())
    perf = sum(1 for entry in result if entry.get("cat") == "_perf")
    if len(result) - perf > budget:
        raise SynthesisError(
            f"composition release would emit {len(result) - perf} capsules against a budget of {budget}"
        )
    return prepare_entries(result)


def selected_facts(evidence, requirement, binding):
    """Use the selected frozen composition facts; geometry must agree with binding."""
    from merlin.targetgen.capability_manifests import derived_output_path

    body = evidence.loaded_facts.get("facts", evidence.loaded_facts)
    context = copy.deepcopy((requirement or {}).get("composition_facts") or body.get("composition") or {})
    path = derived_output_path(evidence.target, facts=evidence.loaded_facts, contract=evidence.contract)
    if not path:
        raise ValueError("composition generation requires a selected derived output path")
    if context.get("output_path", path) != path:
        raise ValueError("composition facts disagree with selected output path")
    context["output_path"] = path
    from merlin.targetgen.composition_facts import derive as derive_composition_facts

    derived = derive_composition_facts(
        facts=evidence.loaded_facts,
        contract=evidence.contract,
        software_spec=evidence.software_spec,
        output_path=path,
    )
    for key, value in derived["facts"].items():
        if context.setdefault(key, value) != value:
            raise ValueError(f"composition facts disagree with the selected derivation of {key}")
    if context.get("tile_dim", binding.tile_dim) != binding.tile_dim:
        raise ValueError("composition facts disagree with selected tile geometry")
    context["tile_dim"] = binding.tile_dim
    scale = next((s for s in path["readout"] if s["kind"] == "scale"), {})
    context["columns_per_scale"] = scale.get("columns_per_scale")
    context["dtype"] = binding.operand_dtype
    # Remaining bounds/placements must come from the selected facts. Missing
    # information is a generation error or a truth refusal, never an assumed ISA.
    return context


def unplaced_members(cap, screen):
    """The original chain members of a positive composition the screen did not admit on the accelerator."""
    if cap.get("operation", {}).get("op") != "composition_program":
        return []
    if cap["semantic"]["composition_truth"]["outcome"] == "refuse":
        return []
    members = set(cap["expected"]["composition"]["source_op_indices"])
    placements = {}
    for row in screen.get("decisions", []):
        for index in row.get("source_op_indices", []):
            if index in placements:
                raise ValueError("composition placement has ambiguous original source identities")
            placements[index] = row.get("placement") if row.get("status") == "admitted" else None
    return sorted(index for index in members if placements.get(index) != "accelerator")


def require_placement(cap, screen):
    """Require independently screened accelerator placement for every member."""
    if unplaced_members(cap, screen):
        raise ValueError("composition requires accelerator placement for every original chain member")


def withhold_unplaced(cap, screen, directory):
    """Disclose a positive composition the screen cannot place whole, and remove what was written.

    Nothing is admitted here: every member decision of the screen stands, and a must-refuse composition
    is never withheld. A composition that needs a member the screen keeps off the accelerator cannot
    grade fused placement, so it leaves the output tree and is recorded as ``planned_not_generated``
    with the member decisions that kept it out. Returns ``None`` when every member is placed.
    """
    import shutil

    unplaced = unplaced_members(cap, screen)
    if not unplaced:
        return None
    truth = cap["semantic"]["composition_truth"]
    members = set(cap["expected"]["composition"]["source_op_indices"])
    record = dict(
        capsule=cap.get("name"),
        status="planned_not_generated",
        rule="composition_member_not_accelerator_admitted",
        chain=truth.get("chain_id"),
        scheme=truth.get("scheme_id"),
        shape_class=truth.get("shape_class"),
        members=sorted(members),
        unplaced_members=unplaced,
        screen_status=screen.get("status"),
        decisions=[
            dict(
                source_op_indices=list(row.get("source_op_indices") or []),
                status=row.get("status"),
                placement=row.get("placement"),
                classification=row.get("classification"),
                reason=row.get("reason"),
            )
            for row in screen.get("decisions", [])
            if members & set(row.get("source_op_indices") or [])
        ],
    )
    shutil.rmtree(directory)
    return record


def withhold_unaccelerated_twin(cap, screen, directory):
    """Withhold a must-refuse composition whose written program places nothing on the accelerator.

    Such a refusal twin grades none of the author's accelerator compiler (host placement is
    Merlin-supplied), and public corpus preparation refuses a capsule with no accelerator placement. The
    refusal is not exempted there: the twin leaves the output tree here and is disclosed with the verbatim
    ``accelerator_placement_refusal`` reason, in the ``host_placement_not_published`` record shape plus the
    chain identity. A positive composition, a twin that places an operation on the accelerator, and a
    capsule already in the ``_diagnostic`` cohort are kept (``None``).
    """
    import shutil

    from .program_admission import accelerator_placement_refusal

    if cap.get("operation", {}).get("op") != "composition_program" or directory.parent.name == "_diagnostic":
        return None
    truth = cap["semantic"]["composition_truth"]
    if truth.get("outcome") != "refuse":
        return None
    reason = accelerator_placement_refusal(screen)
    if reason is None:
        return None
    record = dict(
        capsule=cap.get("name"),
        placements=copy.deepcopy(screen.get("placements") or {}),
        reason=reason,
        rule="composition_refusal_twin_places_nothing_on_accelerator",
        outcome="refuse",
        chain=truth.get("chain_id"),
        scheme=truth.get("scheme_id"),
        shape_class=truth.get("shape_class"),
        screen_status=screen.get("status"),
    )
    shutil.rmtree(directory)
    return record


def read_bound_validation(binding, *, base, evidence, workers=None):
    """Reopen the declared record, selection, fixed suite and retained execution evidence.

    ``workers`` (else ``$MERLIN_FUSION_VERIFY_WORKERS``, else 1) parallelizes only the
    case-payload and retained-pass re-checks; every check and the binding are unchanged.
    """
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from . import fusion_validation as validation
    from .fusion_cases import file_digest

    workers = validation.verification_workers(workers)

    required = {"path", "sha256", "candidates", "cases", "facts", "spec", "backend", "provider", "extension"}
    missing = sorted(required - binding.keys())
    if missing:
        raise ValueError(f"fusion validation selection lacks {missing}")

    def selected_path(name):
        value = Path(binding[name]).expanduser()
        return (value if value.is_absolute() else Path(base) / value).resolve()

    path = selected_path("path")
    if file_digest(path) != binding["sha256"]:
        raise ValueError("fusion validation record digest mismatch")
    ledger = json.loads(path.read_bytes())
    validation.verify_seal(ledger)
    _require_complete_execution(ledger)
    args = SimpleNamespace(
        target=evidence.target,
        candidates=selected_path("candidates"),
        cases=selected_path("cases"),
        facts=selected_path("facts"),
        spec=selected_path("spec"),
        package=selected_path("backend"),
        provider=selected_path("provider"),
        extension=selected_path("extension"),
        snapshot=selected_path("snapshot") if binding.get("snapshot") else None,
        timeout=binding.get("timeout", 600),
    )
    if ledger["target"] != evidence.target or file_digest(args.facts) != evidence.raw_facts_sha256:
        raise ValueError("fusion validation facts differ from current selection")
    selected_specs = {source.sha256 for source in evidence.source_snapshots if source.role == "software-spec"}
    if file_digest(args.spec) not in selected_specs:
        import yaml

        from merlin.compile.fusion_enumeration import _spec_input, campaign_policy

        def campaign_input(document):
            return (_spec_input(document), campaign_policy(document))

        original = campaign_input(yaml.safe_load(args.spec.read_bytes()))
        current = [
            campaign_input(yaml.safe_load(source.content))
            for source in evidence.source_snapshots
            if source.role == "software-spec"
        ]
        if original not in current:
            raise ValueError("fusion validation spec differs from current selection")
    candidates = json.loads(args.candidates.read_bytes())
    if candidates.get("enumeration_complete") is not True:
        raise ValueError("fusion validation enumeration is incomplete")
    for key, member in (
        ("candidate_manifest_sha256", args.candidates),
        ("case_manifest_sha256", args.cases),
        ("facts_sha256", args.facts),
        ("spec_sha256", args.spec),
        ("simulator_extension_sha256", args.extension),
    ):
        if ledger["identity"].get(key) != file_digest(member):
            raise ValueError(f"fusion validation selection digest mismatch: {key}")
    if ledger["identity"].get("reference_backend_sha256") != validation._inventory(args.package):
        raise ValueError("fusion validation backend digest mismatch")
    validation.verify_snapshot(args, candidates)
    import yaml

    from merlin.compile.fusion_enumeration import verify_stratification

    spec = yaml.safe_load(args.spec.read_bytes())
    verify_stratification(candidates, spec, required=True)
    if ledger["scope"]["campaign"] != candidates["campaign"]:
        raise ValueError("fusion validation campaign differs from selected candidates")
    suite = validation.read_suite(args.cases, args.candidates, args.facts, spec=spec, workers=workers)
    if suite.get("profile", {}).get("execution_scope") == "smoke":
        raise ValueError("fusion validation requires a complete deterministic campaign; smoke scope is insufficient")
    from merlin.common.paths import merlin_dir
    from merlin.targetgen.composition_corpus import held_out_shapes

    from .fusion_cases import digest

    # In a frozen run this resolves to the declared private input tree. The
    # canonical alias also serves the suite's payload/publication re-checks.
    exclusion_manifest = (merlin_dir() / "contract/capsules/MANIFEST.yaml").resolve(strict=True)
    if suite["identity"].get("private_exclusion_sha256") != digest(sorted(held_out_shapes(exclusion_manifest))):
        raise ValueError("fusion validation held-out selection changed")
    identity = validation.execution_identity(args, suite)
    if ledger["identity"] != {**suite["identity"], **identity}:
        raise ValueError("fusion validation execution identity differs from current selection")
    root = selected_path("evidence_root") if binding.get("evidence_root") else path.parent
    verified = validation.read_validation(
        path, candidates, suite, identity, root, workers=workers, spec=spec, suite_root=args.cases.parent
    )
    _require_complete_execution(verified)
    return verified, suite, args.cases.parent


def _require_complete_execution(ledger):
    if ledger.get("execution_complete") is not True or any(
        row.get("outcome") == "pending_execution"
        or any(reason.startswith("pending_execution") for reason in row.get("reasons", []))
        for row in ledger["rows"]
    ):
        raise ValueError("fusion validation execution is incomplete or has pending_execution rows")
    if (
        "campaign" not in ledger.get("scope", {})
        or ledger.get("scope", {}).get("execution_scope") == "smoke"
        or any(row.get("state") == "not_selected_for_execution" for row in ledger["rows"])
    ):
        raise ValueError("fusion validation requires a complete deterministic campaign; smoke scope is insufficient")


def _release_chain(node, dataflow, instance):
    from .fusion_cases import chain_for_node

    chain = chain_for_node(node, dataflow)
    chain["producer"] = instance["producer"]
    if "pool" in instance["readout_stages"]:
        for constraint in [
            {"max_requires_store_path_pooling": "from_facts"},
            "scratchpad_source_pooling_fenced",
            {"padded_max_requires": "producer_lower_bound_or_derived_pad_value"},
        ]:
            if constraint not in chain["constraints"]:
                chain["constraints"].append(constraint)
    return chain


def _validated_entry(candidate, case, payload, facts, record_sha256, role):
    """Adapt a validated graph, retaining independent shared-input node semantics."""
    from merlin.targetgen.composition_corpus import assert_generic_shape, held_out_shapes

    semantic = candidate["semantic"]
    shared = semantic["structure"] == "shared_input" and len(semantic["nodes"]) == 2
    if len(semantic["nodes"]) != 1 and not shared:
        raise ValueError("unsupported admitted graph structure")
    cap = payload["nodes"][0]
    attrs = cap["operation"]["attributes"]
    instance = copy.deepcopy(attrs["case"])
    chain = _release_chain(semantic["nodes"][0], semantic["dataflow"], instance)
    scheme = copy.deepcopy(attrs["quantization_scheme"])
    entry = dict(
        name=f"SY_composition_{candidate['candidate_id']}_{case['case_id']}",
        cat="layers",
        kind="layer",
        op="composition_program",
        operand_dtype=facts["dtype"],
        source_role="derived_sweep",
        label="public",
        source_reference="sealed admitted fusion candidate and fixed case",
        lanes={"require": ["on_mesh"]},
        stimulus_range=[-8, 7],
        **{key: attrs[key] for key in ("M", "K", "N")},
        composition_instance=instance,
    )

    # Reuse the validated stimuli, including edge values; no private reference values enter inputs.
    def flatten(values):
        return [scalar for row in values for scalar in flatten(row)] if isinstance(values, list) else [values]

    entry["input_palette"] = dict(
        schema="merlin.component_input_palette.v1",
        inputs=[
            dict(
                name=spec["name"] if shared else spec["name"].removeprefix("node0_"),
                axis="linear",
                offset=0,
                values=flatten(payload["input_values"][spec["name"]]),
            )
            for spec in payload["inputs"]
        ],
    )
    entry["generalization"] = dict(
        composition_chain_id=chain["id"],
        quantization_scheme_id=scheme["id"],
        generalization_axis="composition",
        semantic_family="elementwise_map" if instance["producer"] == "operand_sum" else instance["producer"],
        composition_chain=chain,
        quantization_scheme=scheme,
        composition_instance=instance,
        composition_facts=copy.deepcopy(facts),
        shape_class=case["shape_class"],
        must_accelerate=True,
        eligible=True,
        fusion_validation=dict(
            candidate_id=candidate["candidate_id"],
            record_sha256=record_sha256,
            case_ids=[case["case_id"]],
            role=role,
            graph=copy.deepcopy(semantic),
        ),
    )
    entry["validated_graph"] = copy.deepcopy(
        {key: payload[key] for key in ("source", "source_sha256", "inputs", "nodes", "references")}
    )
    if not shared:
        for leaf in entry["validated_graph"]["inputs"]:
            leaf["name"] = leaf["name"].removeprefix("node0_")
    entry["generalization"]["validated_node_semantics"] = [
        dict(
            composition_chain=_release_chain(node, semantic["dataflow"], child["operation"]["attributes"]["case"]),
            quantization_scheme=copy.deepcopy(child["operation"]["attributes"]["quantization_scheme"]),
            composition_instance=copy.deepcopy(child["operation"]["attributes"]["case"]),
            composition_facts=copy.deepcopy(facts),
        )
        for node, child in zip(semantic["nodes"], payload["nodes"], strict=True)
    ]
    excluded = held_out_shapes()
    assert_generic_shape(entry, excluded)
    for view in payload["producer_views"]:
        a, w = view["inputs"]
        assert_generic_shape(dict(M=a["shape"][0], K=a["shape"][1], N=w["shape"][1]), excluded)
    return entry


def _display_aliases(software_spec, candidate):
    semantic = candidate["semantic"]
    aliases = []
    for chain in (software_spec.get("compositions") or {}).get("chains", []):
        producers = chain["producer"] if isinstance(chain["producer"], list) else [chain["producer"]]
        if bool(chain.get("multiplicity")) != (semantic["structure"] == "shared_input"):
            continue
        if all(
            ("movement" if node["form"] == "window_movement" else node["form"]) in producers
            and node["readout"]["exit"] in chain.get("exits", [])
            and bool(node.get("residual")) == bool(chain.get("extension"))
            and (
                "movement" not in producers
                or node["form"] == "window_movement"
                or any(stage["kind"] == "pool" for stage in node["readout"]["stages"])
            )
            for node in semantic["nodes"]
        ):
            aliases.append(chain["id"])
    return sorted(aliases)


def append_validated(
    entries, software_spec, facts, *, budget, validation, suite, suite_root, record_sha256, target_count=None
):
    """Sample only admitted fixed cases, sharing the ordinary synthesis budget."""
    import json
    from pathlib import Path

    from merlin.targetgen.composition_corpus import sample_validated

    from .fusion_cases import file_digest

    if any(entry.get("op") == "composition_program" for entry in entries):
        raise ValueError("bound validation cannot inherit authored composition capsules")
    selection = sample_validated(
        validation,
        suite,
        software_spec=software_spec,
        budget=budget,
        workload_count=len(entries),
        target_count=target_count,
        selection_path=Path(suite_root) / "fusion-corpus-selection.json",
    )
    candidates = {row["candidate_id"]: row for row in validation["rows"]}
    cases = {row["case_id"]: row for row in suite["rows"]}
    demanded = []
    for row in selection["selected"]:
        case = cases[row["case_id"]]
        member = Path(suite_root) / case["member"]
        if file_digest(member) != case["sha256"]:
            raise ValueError("validated composition case bytes changed")
        demanded.append(
            _validated_entry(
                candidates[row["candidate_id"]],
                case,
                json.loads(member.read_bytes()),
                facts,
                record_sha256,
                row["role"],
            )
        )
    for entry in demanded:
        provenance = entry["generalization"]["fusion_validation"]
        provenance["display_aliases"] = _display_aliases(software_spec, candidates[provenance["candidate_id"]])
    prepared = prepare_entries(demanded)
    for entry in prepared:
        truth = entry["generalization"]["composition_truth"]
        if truth["outcome"] != "pass":
            raise ValueError(f"admitted fusion case fails current composition truth: {truth['reasons']}")
    result = [*entries, *prepared]
    if len({entry["name"] for entry in result}) != len(result):
        raise ValueError("release has duplicate capsule identities")
    selection["validation_record_sha256"] = record_sha256
    return result, selection


def _require_frozen_inputs(cap, palette):
    """Rehash the emitted inputs against the frozen validated stimuli they were planned from."""
    from merlin.common.jsonio import canonical_json
    from merlin.targetgen.capsule_golden import materialize_capsule_leaves
    from merlin.targetgen.corpus_synth import SynthesisError

    frozen = {row["name"]: row["values"] for row in palette["inputs"]}
    if canonical_json(cap.get("input_palette")) != canonical_json(palette):
        raise SynthesisError(f"{cap['name']}: emitted input palette differs from its frozen validated stimuli")
    leaves = materialize_capsule_leaves(cap)
    if set(leaves) != set(frozen) or any(
        canonical_json(_flatten(leaves[name].to_list())) != canonical_json(frozen[name]) for name in frozen
    ):
        raise SynthesisError(f"{cap['name']}: emitted inputs differ from its frozen validated stimuli")


def _flatten(values):
    return [scalar for row in values for scalar in _flatten(row)] if isinstance(values, list) else [values]


def verify_written(entries, written, selection, publication_context):
    """Recheck actual emitted members before crediting the planned regime cover."""
    import yaml

    from merlin.targetgen.corpus_synth import SynthesisError
    from merlin_experiments.corpus.publication import require_public_capsules

    expected = {
        entry["name"]: entry["generalization"]["fusion_validation"]
        for entry in entries
        if (entry.get("generalization") or {}).get("fusion_validation")
    }
    palettes = {entry["name"]: entry["input_palette"] for entry in entries if entry["name"] in expected}
    observed = {}
    roots = []
    for root in written:
        cap = yaml.safe_load((root / "capsule.yaml").read_bytes())
        if cap["name"] in expected:
            observed[cap["name"]] = (cap.get("semantic") or {}).get("fusion_validation")
            roots.append(root)
            _require_frozen_inputs(cap, palettes[cap["name"]])
    if observed != expected or len(observed) != selection["composition_count"]:
        raise SynthesisError("actual emitted composition capsules do not match the admitted regime sample")
    require_public_capsules(publication_context, roots)


def append_selected(entries, evidence, selection):
    """Resolve the declared composition source once within ordinary Phase 0 generation."""
    from merlin.targetgen.composition_corpus import synthesize_compositions_enabled, validation_binding

    workload = selection["workload_spec"]
    enabled = synthesize_compositions_enabled(workload)
    spec = evidence.software_spec if evidence is not None else {}
    binding = validation_binding(spec, workload)
    if binding is None:
        if not enabled or (not spec.get("compositions") and workload.get("synthesize_compositions") is not True):
            return entries, None, None
        raise ValueError("derived composition synthesis requires a sealed fusion-validation binding")
    if evidence is None:
        raise ValueError("bound fusion validation requires selected software and hardware evidence")
    declared_in_spec = (spec.get("compositions") or {}).get("validation") is not None
    base = (
        next(source.path.parent for source in evidence.source_snapshots if source.role == "software-spec")
        if declared_in_spec
        else selection["descriptor"].parent
    )
    record, suite, root = read_bound_validation(
        binding, base=base, evidence=evidence, workers=selection.get("fusion_verify_workers")
    )
    if not enabled:
        return entries, record, None
    entries, report = append_validated(
        entries,
        spec,
        selected_facts(evidence, selection["requirement"], selection["binding"]),
        budget=int(workload.get("max_synthesized_capsules", 160)),
        validation=record,
        suite=suite,
        suite_root=root,
        record_sha256=binding["sha256"],
        target_count=workload.get("composition_target_count"),
    )
    return entries, record, report
