"""Resumable fixed-suite fusion campaign and a fail-closed, byte-bound ledger.

Run with ``python -m merlin_experiments.phase0.fusion_validation --help``.
No full campaign is started implicitly; smoke selection is stable and public.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from merlin.common.paths import artifacts_dir, module_source_path, repo_root

from . import fusion_cases
from .fusion_cases import canonical, digest, file_digest, generate, reference_identity, write_frozen


def verify_seal(document):
    payload = {key: value for key, value in document.items() if key != "seal"}
    if document.get("seal", {}).get("payload_sha256") != digest(payload):
        raise ValueError("fusion record seal changed")


VERIFY_WORKERS_ENV = "MERLIN_FUSION_VERIFY_WORKERS"
#: Start method of every fusion worker process (verification, case generation,
#: execution). The callers are multi-threaded, and a forked child of a
#: multi-threaded process can deadlock, so workers start from a clean forkserver
#: process and receive their inputs by pickling.
POOL_START_METHOD = "forkserver"


def worker_pool(max_workers, **kwargs):
    import multiprocessing

    context = multiprocessing.get_context(POOL_START_METHOD)
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=context, **kwargs)


def _require_workers(workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("fusion suite verification workers must be a positive integer")
    return workers


def verification_workers(value=None):
    """The explicit worker count, else ``$MERLIN_FUSION_VERIFY_WORKERS``, else 1.

    Never derived from the host: an unset count verifies serially, and a malformed
    explicit or environment value is refused rather than replaced by a default.
    """
    if value is None:
        text = os.environ.get(VERIFY_WORKERS_ENV)
        if text is None or text.strip() == "":
            return 1
        if not text.strip().isdigit():
            raise ValueError(f"{VERIFY_WORKERS_ENV} must be a positive integer, got {text!r}")
        value = int(text.strip())
    return _require_workers(value)


def read_suite(path, candidates, facts_path=None, *, spec=None, workers=1):
    """Reopen a sealed suite and rebuild every case payload, on ``workers`` processes."""
    _require_workers(workers)
    suite = json.loads(Path(path).read_bytes())
    from merlin.targetgen.contract.schemas import validate

    validate(suite, "fusion_cases")
    verify_seal(suite)
    if suite.get("schema") != "merlin.fusion_cases.v1" or suite["identity"]["candidate_manifest_sha256"] != file_digest(
        candidates
    ):
        raise ValueError("fusion case/candidate binding changed")
    if suite["identity"]["reference_implementation_sha256"] != reference_identity():
        raise ValueError("fusion reference implementation changed")
    ids = [row["case_id"] for row in suite["rows"]]
    if len(ids) != len(set(ids)) or len(ids) != suite["case_count"]:
        raise ValueError("fusion case denominator changed")
    manifest = json.loads(Path(candidates).read_bytes())
    from merlin.compile.fusion_enumeration import verify_stratification

    verify_stratification(manifest, spec, required=True)
    expected = {row["candidate_id"] for row in manifest["rows"] if row["state"] == "candidate"}
    if "selected_candidate_ids" in suite["profile"]:
        selected = suite["profile"]["selected_candidate_ids"]
        if (
            not isinstance(selected, list)
            or not selected
            or any(not isinstance(value, str) for value in selected)
            or len(selected) != len(set(selected))
            or not set(selected) <= expected
        ):
            raise ValueError("fusion suite candidate scope is invalid")
        if "campaign" in manifest:
            scope = set(smoke_ids(manifest)) if suite["profile"].get("execution_scope") == "smoke" else expected
            if set(selected) != scope:
                raise ValueError("fusion suite differs from deterministic execution scope")
        expected = set(selected)
    refused = suite["profile"].get("refused_candidates", [])
    refused_ids = [row["candidate_id"] for row in refused]
    if (
        len(set(refused_ids)) != len(refused_ids)
        or not set(refused_ids) <= expected
        or any(row.get("reason") != "independent unfused references disagree" for row in refused)
    ):
        raise ValueError("fusion suite candidate refusal is invalid")
    expected -= set(refused_ids)
    if {row["candidate_id"] for row in suite["rows"]} != expected:
        raise ValueError("fusion suite omits a candidate")
    required = {}
    for candidate in manifest["rows"]:
        if candidate["candidate_id"] in expected:
            for label, coefficient in fusion_cases.parameters(candidate):
                selection = dict(candidate_id=candidate["candidate_id"], shape_class=label, coefficient=coefficient)
                required[digest(selection)] = selection
    if set(ids) != required.keys() or any(
        any(row[key] != value for key, value in required[row["case_id"]].items())
        or row["required_engines"] != ["spike"]
        or row["positive"] is not True
        for row in suite["rows"]
    ):
        raise ValueError("fusion suite required case identities changed")
    if suite["candidate_count"] != len(manifest["rows"]) or suite["candidate_case_count"] != len(expected):
        raise ValueError("fusion suite candidate denominator changed")
    for row in suite["rows"]:
        relative = Path(row["member"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("fusion case member escapes suite")
        member = Path(path).parent / relative
        if (
            member.is_symlink()
            or not member.is_file()
            or member.stat().st_size != row["bytes"]
            or file_digest(member) != row["sha256"]
        ):
            raise ValueError("frozen case/reference bytes changed")
    _verify_case_payloads(path, manifest, suite, facts_path, workers=workers)
    return suite


def _case_inputs(manifest, facts_path):
    from merlin.compile.fusion_enumeration import _unordered_inputs, canonical_sha256
    from merlin.targetgen.core_aten_device import selected_facts

    with selected_facts(manifest["target"], facts_path) as raw_facts:
        if canonical_sha256(_unordered_inputs(raw_facts)) != manifest["identity"]["facts_sha256"]:
            raise ValueError("candidate facts binding changed")
        facts = fusion_cases.facts_for_cases(manifest, raw_facts)
    schemes = {row["id"]: row for row in manifest["scheme_resolution"]["schemes"]}
    return schemes, facts, fusion_cases.corpus.held_out_shapes(), fusion_cases.exclusion_context()


def _verify_case_payloads(path, manifest, suite, facts_path, *, workers=1):
    schemes, facts, excluded, context = _case_inputs(manifest, facts_path)
    if suite["identity"].get("private_exclusion_sha256") != digest(sorted(excluded)):
        raise ValueError("fusion validation held-out selection changed")
    candidates = {row["candidate_id"]: row for row in manifest["rows"]}
    with tempfile.TemporaryDirectory() as temporary:
        for refusal in suite["profile"].get("refused_candidates", []):
            candidate = candidates[refusal["candidate_id"]]
            try:
                for label, coefficient in fusion_cases.parameters(candidate):
                    fusion_cases.build_payload(
                        candidate, schemes, facts, label, coefficient, excluded, context, Path(temporary)
                    )
            except fusion_cases.ReferenceDisagreement:
                continue
            raise ValueError("fusion candidate reference refusal differs from deterministic generation")
    rows = suite["rows"]
    used = {row["candidate_id"]: candidates[row["candidate_id"]] for row in rows}
    inputs = (used, schemes, facts, excluded, context, Path(path).parent)
    if workers == 1 or len(rows) < 2:
        verified, failure = _verify_payload_rows(inputs, 0, rows)
    else:
        verified, failure = _verify_payload_rows_parallel(inputs, rows, workers)
    if failure is not None:
        raise failure[1]
    if verified != len(rows):
        raise ValueError("fusion case payload verification did not cover every case")


_PAYLOAD_INPUTS = None


def _set_payload_inputs(inputs):
    global _PAYLOAD_INPUTS
    _PAYLOAD_INPUTS = inputs


def _verify_payload_chunk(start, rows):
    return _verify_payload_rows(_PAYLOAD_INPUTS, start, rows)


def _verify_payload_rows(inputs, start, rows):
    """Rebuild ``rows`` in order; return the verified count and the first ``(index, error)``.

    The error is returned rather than raised so a parallel reader can report the
    failure of the earliest case in canonical order, whichever chunk finished first.
    """
    candidates, schemes, facts, excluded, context, root = inputs
    with tempfile.TemporaryDirectory() as temporary:
        for offset, row in enumerate(rows):
            try:
                expected = fusion_cases.build_payload(
                    candidates[row["candidate_id"]],
                    schemes,
                    facts,
                    row["shape_class"],
                    row["coefficient"],
                    excluded,
                    context,
                    Path(temporary),
                )
                if (root / row["member"]).read_bytes() != canonical(expected):
                    raise ValueError("fusion case stimuli/reference differ from deterministic generation")
            except Exception as error:  # re-raised by the reader for the earliest failing case
                return offset, (start + offset, error)
    return len(rows), None


def _verify_payload_rows_parallel(inputs, rows, workers):
    """Verify contiguous chunks on processes; the earliest failing chunk decides the error.

    Chunks are consumed in canonical order, so a reported failure is preceded only by
    fully verified chunks. A crashed worker raises from ``result()`` and fails the read.
    """
    size = max(1, min(32, len(rows) // (workers * 8)))
    chunks = [(start, rows[start : start + size]) for start in range(0, len(rows), size)]
    verified = 0
    with worker_pool(min(workers, len(chunks)), initializer=_set_payload_inputs, initargs=(inputs,)) as pool:
        futures = [pool.submit(_verify_payload_chunk, start, chunk) for start, chunk in chunks]
        try:
            for future in futures:
                count, failure = future.result()
                verified += count
                if failure is not None:
                    return verified, failure
        finally:
            for future in futures:
                future.cancel()
    return verified, None


def smoke_ids(candidates):
    """Cover each form/scheme/exit, plus seed, extension, sharing and pooling."""
    rows = [row for row in candidates["rows"] if row["state"] == "candidate"]

    def rank(row):
        nodes = row["semantic"]["nodes"]
        return (
            row["semantic"]["structure"] != "single",
            row["semantic"]["dataflow"] == "OS",
            sum(bool(n["residual"]) + n["seed"] for n in nodes),
            sum(len(n["readout"]["stages"]) for n in nodes),
            row["candidate_id"],
        )

    selected, coverage = [], set()
    for row in sorted(rows, key=rank):
        semantic = row["semantic"]
        keys = {
            (node["form"], node["realization"], node["scheme"], node["readout"]["exit"]) for node in semantic["nodes"]
        }
        if not keys <= coverage:
            selected.append(row["candidate_id"])
            coverage |= keys
    mechanisms = ["seed", "residual", "shared_input", "shared_input_pool", "pool", "pool_padded"]
    mechanisms.extend(f"constant_seed_{realization}" for realization in ("matmul", "conv_as_matmul"))
    for mechanism in mechanisms:
        for scheme in sorted({n["scheme"] for row in rows for n in row["semantic"]["nodes"]}):
            for exit_name in ("accumulator", "requant"):
                options = [
                    row
                    for row in rows
                    if all(
                        n["scheme"] == scheme and n["readout"]["exit"] == exit_name for n in row["semantic"]["nodes"]
                    )
                    and _smoke_mechanism(row["semantic"], mechanism)
                ]
                if options:
                    selected.append(min(options, key=rank)["candidate_id"])
    return sorted(set(selected))


def _smoke_mechanism(semantic, mechanism):
    nodes = semantic["nodes"]
    pooled = [any(s["kind"] == "pool" for s in n["readout"]["stages"]) for n in nodes]
    if mechanism in {"shared_input", "shared_input_pool"}:
        return semantic["structure"] == "shared_input" and any(pooled) == (mechanism == "shared_input_pool")
    if mechanism in {"pool", "pool_padded"}:
        return any(
            n["form"] == "contraction"
            and n["readout"].get("pool_padding") == "zero_or_padded"
            and n["readout"].get("pool_padding_class", "unpadded")
            == ("padded" if mechanism == "pool_padded" else "unpadded")
            for n in nodes
        )
    if mechanism.startswith("constant_seed_"):
        realization = mechanism.removeprefix("constant_seed_")
        return semantic["structure"] == "single" and all(
            n.get("seed_source") == "constant"
            and n["realization"] == realization
            and not any(pooled)
            and (
                n["readout"]["exit"] == "accumulator"
                or {s["kind"] for s in n["readout"]["stages"]} >= {"activation", "convert", "scale", "requant"}
            )
            for n in nodes
        )
    if mechanism == "seed":
        return any(n["seed"] and n.get("seed_source", "runtime_argument") == "runtime_argument" for n in nodes)
    return any(bool(n[mechanism]) for n in nodes)


def candidate_decision(candidate, required, observations):
    """A host result, split kernel, missing result, or one failure earns nothing."""
    if candidate["state"] == "deferred_by_stratification":
        return dict(admitted=False, outcome="not_admitted", reasons=["deferred_by_stratification"])
    if candidate["state"] != "candidate":
        return dict(admitted=False, outcome="not_admitted", reasons=[r["reason"] for r in candidate["reasons"]])
    if not required:
        return dict(admitted=False, outcome="not_admitted", reasons=["pending_execution: no positive required cases"])
    reasons = set()
    for case_id in required:
        row = observations.get(case_id)
        if row is None:
            reasons.add("pending_execution")
        elif not (
            row.get("outcome") == "pass"
            and row.get("numeric") is True
            and row.get("placement") is True
            and row.get("trace") is True
        ):
            reasons.add(row.get("reason") or "incomplete numeric/placement/trace evidence")
    return dict(admitted=not reasons, outcome="admitted" if not reasons else "not_admitted", reasons=sorted(reasons))


def _revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _repository_commit():
    from merlin_experiments import frozen_python, source_snapshot

    root = repo_root()
    active = frozen_python.active_source_identity()
    if active is None and not any(root.glob("snapshot.*.json")):
        return _revision(root)
    receipt = source_snapshot.verify(root)
    if active is not None:
        seal, _ = source_snapshot.load_seal(root, "snapshot")
        if active != dict(path=str(seal), sha256=file_digest(seal)):
            raise ValueError("fusion source snapshot differs from guarded execution")
    commit = receipt.get("repository_commit")
    if commit is None:
        raise ValueError("fusion source snapshot lacks originating repository commit; freeze a new run")
    return commit


def _inventory(root):
    """Pin backend source bytes as well as its revision, including dirty syncs."""
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], text=True).split("\0")
    return digest({name: file_digest(root / name) for name in paths if name and (root / name).is_file()})


def verify_snapshot(args, candidates):
    from merlin.compile.fusion_enumeration import (
        _producer_forms,
        _spec_input,
        _unordered_inputs,
        canonical_sha256,
        enumerate_candidates,
    )
    from merlin.targetgen.capability_manifests import _derived_output_epilogue
    from merlin.targetgen.core_aten_device import selected_facts
    from merlin.targetgen.software_spec import load_software_spec
    from merlin.targetgen.spec_fact_drift import _resolve_chain, fact_capabilities, resolve_spec
    from merlin.targetgen.target_registry import load_contract

    with selected_facts(args.target, args.facts) as facts:
        contract = load_contract(args.target)
        if args.snapshot is not None:
            frozen = json.loads(args.snapshot.read_bytes())
            runtime_source = {k: v for k, v in contract.items() if k not in {"encoding", "output_epilogue"}}
            frozen_source = {k: v for k, v in frozen["contract"].items() if k not in {"encoding", "output_epilogue"}}
            if frozen["target"] != args.target or runtime_source != frozen_source:
                raise ValueError("enumeration snapshot differs from runtime contract source")
            if canonical_sha256(_unordered_inputs(frozen["facts"])) != candidates["identity"]["facts_sha256"]:
                raise ValueError("enumeration snapshot facts changed")
            # Keep source YAML key types: JSON snapshots stringify numeric encoding keys.
            contract["output_epilogue"] = frozen["contract"]["output_epilogue"]
        if not (contract.get("output_epilogue") or {}).get("output_path"):
            contract["output_epilogue"] = _derived_output_epilogue(args.target, contract, facts)
        capabilities = fact_capabilities(target=args.target, contract=contract, raw_facts=facts, data_only=True)
        spec, _ = resolve_spec(_spec_input(load_software_spec(args.spec, target=args.target)), capabilities)
        producers = sorted({"movement" if form == "window_movement" else form for form, _ in _producer_forms(spec)})
        placement = _resolve_chain(
            dict(producer=producers, constraints=[]),
            capabilities,
            dict(output_path=contract["output_epilogue"].get("output_path")),
        )
        spec["composition_resolution"] = dict(chains=[placement])
        observed = dict(
            facts_sha256=canonical_sha256(_unordered_inputs(facts)),
            contract_sha256=canonical_sha256(_unordered_inputs(contract)),
            spec_sha256=canonical_sha256(_spec_input(spec)),
        )
        if any(candidates["identity"][key] != value for key, value in observed.items()):
            raise ValueError("execution facts, contract or resolved spec differ from enumeration")
        expected = enumerate_candidates(target=args.target, facts=facts, contract=contract, spec=spec)
        from merlin.compile.fusion_enumeration import stratify_candidates

        if "campaign" in candidates:
            expected = stratify_candidates(expected, load_software_spec(args.spec, target=args.target))
        if canonical(candidates) != canonical(expected):
            raise ValueError("fusion candidate manifest differs from deterministic enumeration")


def execution_identity(args, suite):
    from merlin.runtime.backends.spike import spike_path
    from merlin.targetgen.core_aten_device import load_execution_provider, selected_facts
    from merlin.targetgen.target_registry import load_contract

    from .fusion_execution import pinned_extension

    with selected_facts(args.target, args.facts):
        provider = load_execution_provider(args.target, args.provider.resolve(), candidate_writable_roots=())
        pinned_extension(args.target, args.extension.resolve(), provider.runner_options())
    code = [
        "merlin_experiments.phase0.fusion_validation",
        "merlin_experiments.phase0.fusion_execution",
        "merlin.targetgen.composition_audit",
        "merlin.targetgen.native_instruction_audit",
        "merlin.targetgen.frozen_placement",
        "merlin.runtime.backends.spike_model",
    ]
    tools = {
        name: file_digest(Path(os.environ[name]))
        for name in ("MERLIN_RISCV_GCC", "MERLIN_CLANG", "MERLIN_MLIR_TRANSLATE", "MERLIN_OBJDUMP")
    }
    return dict(
        repository_commit=_repository_commit(),
        reference_backend_revision=_revision(args.package),
        enumeration_snapshot_sha256=file_digest(args.snapshot) if args.snapshot else "not_supplied",
        resolved_spec_sha256=suite["identity"]["spec_sha256"],
        resolved_contract_sha256=suite["identity"]["contract_sha256"],
        reference_backend_sha256=_inventory(args.package),
        case_manifest_sha256=file_digest(args.cases),
        candidate_manifest_sha256=file_digest(args.candidates),
        facts_sha256=file_digest(args.facts),
        spec_sha256=file_digest(args.spec),
        contract_sha256=digest(load_contract(args.target)),
        simulator_extension_sha256=file_digest(args.extension),
        spike_sha256=file_digest(spike_path()),
        compiler_toolchain_sha256=digest(tools),
        execution_policy_sha256=digest(
            dict(
                engine="spike",
                timeout=args.timeout,
                tier_scope="functional_validation_only",
                required_checks=["exact_output", "same_region", "retired_lineage", "no_host_fallback"],
            )
        ),
        validation_implementation_sha256=digest({name: file_digest(module_source_path(name)) for name in code}),
        reference_implementation_sha256=suite["identity"]["reference_implementation_sha256"],
    )


def _evidence(directory, root):
    files = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {"result.json", "timing.json"}
    ]
    return [
        dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=file_digest(path)) for path in files
    ]


def verify_members(members, root):
    for member in members:
        relative = Path(member["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("evidence member escapes campaign")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != member["bytes"]
            or file_digest(path) != member["sha256"]
        ):
            raise ValueError("fusion evidence bytes changed")


def _worker(case, candidate, selection, root, suite_root, identity):
    from .fusion_execution import execute_case

    directory = Path(root) / "workers" / case["case_id"]
    path = directory / "result.json"
    if path.exists():
        result = json.loads(path.read_bytes())
        if result["identity"] != identity or result["case_sha256"] != case["sha256"]:
            raise ValueError("resume selection changed")
        verify_members(result["evidence_members"], Path(root))
        return result, json.loads((directory / "timing.json").read_bytes())["wall_s"]
    if directory.exists():
        # Preserve a killed attempt; it never becomes evidence for the fresh run.
        attempt = 0
        while directory.with_name(directory.name + f".incomplete-{attempt}").exists():
            attempt += 1
        directory.rename(directory.with_name(directory.name + f".incomplete-{attempt}"))
    directory.mkdir(parents=True)
    started = time.monotonic()
    payload = json.loads((Path(suite_root) / case["member"]).read_bytes())
    result = execute_case(payload, candidate, selection, directory)
    wall = time.monotonic() - started
    result.update(
        case_id=case["case_id"],
        candidate_id=case["candidate_id"],
        identity=identity,
        case_sha256=case["sha256"],
        evidence_members=_evidence(directory, Path(root)),
    )
    write_frozen(directory / "timing.json", canonical(dict(wall_s=wall)))
    write_frozen(path, canonical(result))
    return result, wall


def required_case_ids(candidates, suite):
    """Reconstruct positive obligations from graph semantics, including siblings."""
    grouped = {}
    if suite.get("schema") != "merlin.fusion_cases.v1":
        for case in suite["rows"]:
            grouped.setdefault(case["candidate_id"], []).append(case["case_id"])
        return grouped
    selected = suite.get("profile", {}).get("selected_candidate_ids")
    refused = {row["candidate_id"] for row in suite.get("profile", {}).get("refused_candidates", [])}
    for candidate in candidates["rows"]:
        identity = candidate["candidate_id"]
        if candidate["state"] == "candidate" and identity not in refused and (selected is None or identity in selected):
            grouped[identity] = [
                digest(dict(candidate_id=identity, shape_class=label, coefficient=coefficient))
                for label, coefficient in fusion_cases.parameters(candidate)
            ]
    return grouped


def record(candidates, suite, identity, results):
    grouped = required_case_ids(candidates, suite)
    refused = {row["candidate_id"]: row["reason"] for row in suite.get("profile", {}).get("refused_candidates", [])}
    rows, evidence = [], {}
    for candidate in candidates["rows"]:
        required = sorted(grouped.get(candidate["candidate_id"], []))
        state = candidate["state"]
        if candidate["candidate_id"] in refused:
            decision = dict(admitted=False, outcome="not_admitted", reasons=[refused[candidate["candidate_id"]]])
        elif (
            state == "candidate"
            and suite.get("profile", {}).get("selected_candidate_ids") is not None
            and candidate["candidate_id"] not in suite["profile"]["selected_candidate_ids"]
        ):
            state = "not_selected_for_execution"
            decision = dict(admitted=False, outcome="not_admitted", reasons=["outside_execution_scope"])
        else:
            decision = candidate_decision(candidate, required, results)
        outcomes = []
        for case_id in required:
            result = results.get(case_id)
            if result is None:
                outcomes.append(dict(case_id=case_id, outcome="pending_execution"))
            else:
                outcomes.append({k: v for k, v in result.items() if k not in {"identity", "evidence_members"}})
                for member in result["evidence_members"]:
                    evidence[member["path"]] = member
        rows.append(
            dict(
                candidate_id=candidate["candidate_id"],
                semantic=candidate["semantic"],
                state=state,
                **({"stratum_key": candidate["stratum_key"]} if "stratum_key" in candidate else {}),
                required_case_ids=required,
                required_case_engine_pairs=[[c, "spike"] for c in required],
                cases=outcomes,
                **decision,
            )
        )
    ledger = dict(
        schema="merlin.fusion_validation.v1",
        target=candidates["target"],
        configuration=dict(facts_sha256=identity["facts_sha256"], contract_sha256=identity["contract_sha256"]),
        scope=dict(
            profile=candidates["profile"],
            **({"campaign": candidates["campaign"]} if "campaign" in candidates else {}),
            qualification="fixed_cases_functional_spike_only",
            rtl_qualification="not_established",
        ),
        identity={**suite["identity"], **identity},
        candidate_count=len(rows),
        case_count=suite["case_count"],
        case_engine_count=suite["case_count"],
        enumeration_complete=candidates["enumeration_complete"],
        execution_complete=(
            len(results) == suite["case_count"]
            and set(results) == {case for cases in grouped.values() for case in cases}
        ),
        rows=rows,
        admitted_ids=sorted(row["candidate_id"] for row in rows if row["admitted"]),
        evidence_members=sorted(evidence.values(), key=lambda row: row["path"]),
    )
    ledger["seal"] = dict(
        canonicalization="merlin.canonical_json.v1",
        payload_sha256=digest(ledger),
        evidence_manifest_sha256=digest(ledger["evidence_members"]),
    )
    return ledger


def _verify_pass_timed(arguments):
    from .fusion_execution import verify_pass

    started = time.monotonic()
    verify_pass(*arguments)
    return time.monotonic() - started


def _verify_passes(audits, workers):
    """Audit every retained pass in canonical order; the earliest failing audit raises.

    Results are consumed in submission order, so a raised audit is preceded only by
    audits that passed; a crashed worker raises from ``result()`` and fails the read.
    """
    if workers == 1 or len(audits) < 2:
        return [_verify_pass_timed(arguments) for arguments in audits]
    with worker_pool(min(workers, len(audits))) as pool:
        futures = [pool.submit(_verify_pass_timed, arguments) for arguments in audits]
        try:
            measured = [future.result() for future in futures]
        finally:
            for future in futures:
                future.cancel()
    if len(measured) != len(audits):
        raise ValueError("fusion pass audit did not cover every retained pass")
    return measured


def read_validation(
    path, candidates, suite, identity, evidence_root, *, workers=1, audit_timings=None, spec=None, suite_root=None
):
    """Reopen all retained evidence and recompute the complete admission denominator."""
    _require_workers(workers)
    ledger = json.loads(Path(path).read_bytes())
    from merlin.compile.fusion_enumeration import verify_stratification
    from merlin.targetgen.contract.schemas import validate

    selection_path = Path(evidence_root) / "execution-selection.json"
    selection = json.loads(selection_path.read_bytes()) if selection_path.is_file() else {}
    if spec is None and selection.get("spec"):
        import yaml

        spec = yaml.safe_load(Path(selection["spec"]).read_bytes())
    verify_stratification(candidates, spec, required=True)

    validate(ledger, "fusion_validation")
    verify_seal(ledger)
    verify_members(ledger["evidence_members"], Path(evidence_root))
    observations = {}
    audits = []
    case_map = {row["case_id"]: row for row in suite["rows"]}
    candidate_map = {row["candidate_id"]: row for row in candidates["rows"]}
    for row in ledger["rows"]:
        for result in row["cases"]:
            if result["outcome"] != "pending_execution":
                case = case_map.get(result["case_id"])
                if case is None or case["candidate_id"] != row["candidate_id"]:
                    raise ValueError("fusion result names an unbound case")
                if case.get("sha256") != result.get("case_sha256"):
                    raise ValueError("fusion case/reference binding changed")
                if result["outcome"] == "pass":
                    directory = Path(evidence_root) / "workers" / result["case_id"]
                    saved = json.loads((directory / "result.json").read_bytes())
                    if {k: v for k, v in saved.items() if k not in {"identity", "evidence_members"}} != result or saved[
                        "identity"
                    ] != identity:
                        raise ValueError("fusion result differs from retained worker observation")
                    selected_root = suite_root
                    if selected_root is None and selection.get("cases"):
                        selected_root = Path(selection["cases"]).parent
                    if selected_root is None:
                        raise ValueError("fusion case suite root is absent from execution selection")
                    member = Path(selected_root) / case["member"]
                    if file_digest(member) != case["sha256"]:
                        raise ValueError("fusion source/reference member changed")
                    audits.append(
                        (json.loads(member.read_bytes()), candidate_map[row["candidate_id"]], selection, directory)
                    )
                observations[result["case_id"]] = {**result, "identity": identity, "evidence_members": []}
    measured = _verify_passes(audits, workers)
    if audit_timings is not None:
        audit_timings.extend(measured)
    expected = record(candidates, suite, identity, observations)
    for key in (
        "schema",
        "target",
        "configuration",
        "scope",
        "case_engine_count",
        "identity",
        "candidate_count",
        "case_count",
        "enumeration_complete",
        "execution_complete",
        "rows",
        "admitted_ids",
    ):
        if ledger.get(key) != expected[key]:
            raise ValueError(f"fusion validation binding/decision changed: {key}")
    return ledger


def campaign(args):
    import yaml

    candidates = json.loads(args.candidates.read_bytes())
    verify_snapshot(args, candidates)
    spec = yaml.safe_load(args.spec.read_bytes())
    suite = read_suite(args.cases, args.candidates, args.facts, spec=spec, workers=args.workers)
    identity = execution_identity(args, suite)
    args.output.mkdir(parents=True, exist_ok=True)
    write_frozen(args.output / "selection.json", canonical(identity))
    # read_suite checked the suite's sealed scope against the campaign stratification;
    # a non-smoke run executes exactly that scope, or every stratified candidate if none.
    sealed = suite["profile"].get("selected_candidate_ids")
    if args.smoke:
        selected = set(smoke_ids(candidates)) | set(sealed or [])
    elif sealed is not None:
        selected = set(sealed)
    else:
        selected = {r["candidate_id"] for r in candidates["rows"] if r["state"] == "candidate"}
    selected -= {row["candidate_id"] for row in suite["profile"].get("refused_candidates", [])}
    if not selected <= {row["candidate_id"] for row in suite["rows"]}:
        raise ValueError("fusion suite omits selected execution candidates")
    by_id = {row["candidate_id"]: row for row in candidates["rows"]}
    tasks = [case for case in suite["rows"] if case["candidate_id"] in selected]
    selection = dict(
        target=args.target,
        package=str(args.package.resolve()),
        provider=str(args.provider.resolve()),
        spec=str(args.spec.resolve()),
        cases=str(args.cases.resolve()),
        facts=str(args.facts.resolve()),
        extension_sha256=identity["simulator_extension_sha256"],
        timeout=args.timeout,
        extension=str(args.extension.resolve()),
        candidates={"normalized_facts": candidates["normalized_facts"]},
    )
    write_frozen(args.output / "execution-selection.json", canonical(selection))
    results, timings = {}, {}
    started = time.monotonic()
    with worker_pool(args.workers) as pool:
        futures = [
            pool.submit(_worker, case, by_id[case["candidate_id"]], selection, args.output, args.cases.parent, identity)
            for case in tasks
        ]
        for future in as_completed(futures):
            result, wall = future.result()
            results[result["case_id"]] = result
            timings[result["case_id"]] = wall
            if len(results) % 10 == 0 or len(results) == len(tasks):
                print(
                    json.dumps(
                        dict(completed=len(results), cases=len(tasks), elapsed_s=round(time.monotonic() - started, 2))
                    ),
                    flush=True,
                )
    # Include earlier completed shards with the same immutable execution selection.
    for path in sorted((args.output / "workers").glob("*/result.json")):
        result = json.loads(path.read_bytes())
        if result["identity"] != identity:
            raise ValueError("campaign contains another execution selection")
        verify_members(result["evidence_members"], args.output)
        results[result["case_id"]] = result
    ledger = record(candidates, suite, identity, results)
    (args.output / "fusion-validation.json").write_bytes(canonical(ledger))
    audit_timings = []
    read_validation(
        args.output / "fusion-validation.json",
        candidates,
        suite,
        identity,
        args.output,
        workers=args.workers,
        audit_timings=audit_timings,
        suite_root=args.cases.parent,
    )
    summary = dict(
        smoke=args.smoke,
        candidates=len(selected),
        cases=len(tasks),
        admitted=len(ledger["admitted_ids"]),
        wall_s=time.monotonic() - started,
        worker_seconds=sum(timings.values()),
        mean_case_s=sum(timings.values()) / len(timings),
        validation_worker_seconds=sum(audit_timings),
        full_case_count=suite["case_count"],
        full_40_worker_s=suite["case_count"] * (sum(timings.values()) + sum(audit_timings)) / len(timings) / 40,
        results_by_candidate=[
            {k: row[k] for k in ("candidate_id", "semantic", "outcome", "reasons")}
            for row in ledger["rows"]
            if row["candidate_id"] in selected
        ],
    )
    (args.output / "smoke-summary.json").write_bytes(canonical(summary))
    print(json.dumps({key: value for key, value in summary.items() if key != "results_by_candidate"}, sort_keys=True))
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--provider", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--snapshot", type=Path, help="original enumerator frozen facts/contract/spec inputs")
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.cases = args.cases or artifacts_dir() / "verification" / args.target / "fusion-cases.json"
    if args.generate:
        if args.spec is None:
            parser.error("--generate requires --spec")
        import yaml

        spec = yaml.safe_load(args.spec.read_bytes())
        suite = generate(args.candidates, args.facts, args.cases.parent, spec=spec, workers=args.workers)
        print(json.dumps(dict(case_count=suite["case_count"], candidate_count=suite["candidate_case_count"])))
        return 0
    if args.workers < 1 or args.timeout < 1:
        parser.error("workers and timeout must be positive")
    for name in ("package", "provider", "spec", "extension"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required for execution")
    args.output = args.output or args.cases.parent / "fusion-campaign"
    campaign(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
