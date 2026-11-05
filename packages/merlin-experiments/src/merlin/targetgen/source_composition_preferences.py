"""Bounded candidate requests; original owners retain all source authority."""

import json
from hashlib import sha256
from pathlib import Path


def read(candidate_package):
    path = Path(candidate_package) / "source_composition.json"
    defaults = {"shared_input_siblings": False, "prepack_weights": False}
    optional = {"share_exact_tensor_preparations", "dense_input_views", "op_profile", "group_profile",
                "independent_heads", "whole_head_batches", "shared_rhs_row_views", "host_bounded_rne",
                "host_roundeven", "host_elementwise_fusion",
                "propagate_layouts", "reuse_intermediate_storage"}
    if not path.exists() and not path.is_symlink():
        return defaults, None
    if (any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file() or path.stat().st_size > 8192):
        raise ValueError("source composition preferences need a small direct JSON file")
    raw = path.read_bytes()

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("source composition preferences repeat a field")
            result[key] = value
        return result

    record = json.loads(raw, object_pairs_hook=unique)
    if (type(record) is not dict or not {"schema", *defaults} <= set(record)
            or set(record) - {"schema", *defaults, *optional}
            or record["schema"] != "merlin.source-composition-preferences.v1"
            or any(type(record[key]) is not bool for key in record if key != "schema")):
        raise ValueError("source composition preferences contain unsupported fields or values")
    # Absent optional keys preserve the exact prior request/qualification bytes.
    return {key: record[key] for key in record if key != "schema"}, {
        "path": str(path.absolute()), "sha256": sha256(raw).hexdigest(),
    }


def weight_preparation(bundle, *, requested):
    """Resolve the original optional request; normal byte/ancestry gates remain."""
    if not requested:
        return False, {"status": "not_requested"}
    from merlin.llvmlower.weight_prepack import PrepackRefused, plan
    from merlin.llvmlower.weight_prepack_runtime import fixed_arguments

    try:
        manifest = json.loads((Path(bundle) / "weights.safetensors.manifest.json").read_text())
        immutable = fixed_arguments(bundle, manifest)
        proposal = plan(bundle, immutable_arguments=immutable)
        if not proposal.ok:
            raise PrepackRefused("; ".join(proposal.problems) or "no hoistable immutable weight layout")
    except (ValueError, PrepackRefused) as exc:
        return False, {"status": "not_applicable", "reason": str(exc), "baseline_retained": True}
    return True, {"status": "source_planned_not_prepared", "immutable_arguments": list(immutable),
                  "hoistable_layouts": proposal.hoistable,
                  "scope": "original source/ownership proposal; ordinary preparation must revalidate exact bytes"}


def host_preparation(preferences, *, target, dtype='fp32', composed=False, lane=None, descriptor_effects=None):
    """Select original frozen software permissions, never candidate assertions.

    The ordinary host lane's selected numeric recipe is an existing startup
    input. Resolve the declaration from its frozen bytes, then read that exact
    software input through the same snapshot owner. Missing declarations defer;
    malformed/stale selected inputs remain errors.
    """
    codegen = {"host_bounded_rne": "source_proof_and_scalar_isa_required",
               "host_roundeven": "scalar_isa_and_linked_helper_required",
               "host_elementwise_fusion": "scalar_lane_and_effective_pipeline_required"}
    options, selection = {}, {}
    for name, status in codegen.items():
        if name not in preferences:
            continue
        value = preferences[name]
        if type(value) is not bool:
            raise ValueError("host codegen requests must be explicit Boolean choices")
        options[name] = value
        selection[name] = {
            'status': status if value else 'not_requested',
            'effect_grant': False, 'historical_observer_permissions_imported': False,
        }
    requests = {'propagate_layouts': 'pre_group_layout', 'reuse_intermediate_storage': 'intermediate_storage'}
    if not any(preferences.get(name, False) for name in requests):
        return options, selection
    import yaml
    from merlin_experiments.corpus.numeric_policy import numeric_profile_path

    from merlin.llvmlower.host_graph_policy import read_host_graph_policy
    from merlin.targetgen.capsule_runner import _model_frozen_input, _resolve_model_host_lane
    from merlin.targetgen.software_spec import software_spec_path_for_recipe

    experiment, _, identity = lane if lane is not None else _resolve_model_host_lane(target, dtype)
    recipe = numeric_profile_path(experiment.numeric_profile, repo=experiment._source_root())
    software, carrier = None, None
    if recipe is not None:
        frozen_recipe = _model_frozen_input(recipe, identity.get('run_snapshot'))
        raw = frozen_recipe.read_bytes()
        selected = software_spec_path_for_recipe(recipe, document=yaml.safe_load(raw))
        carrier = {'numeric_profile': {'path': str(frozen_recipe), 'sha256': sha256(raw).hexdigest()}}
        if selected is not None:
            software = _model_frozen_input(selected, identity.get('run_snapshot'))
    policy = (read_host_graph_policy(
        software, target=target, entry='forward',
        connected_layout=preferences.get('propagate_layouts', False),
    ) if software is not None else None)
    for name, argument in requests.items():
        if not preferences.get(name, False):
            continue
        reason = 'original_software_not_selected' if policy is None else None
        if reason is not None:
            selection[name] = {'status': 'deferred', 'reason': reason, 'baseline_retained': True}
            continue
        context = policy.context
        permitted = ('logical_host' in context.permitted_stages
                     and 'propagate_layout' in context.permitted_optimizations
                     and context.independent_element_order_unobserved) if name == 'propagate_layouts' else (
                         'llvm' in context.permitted_stages and 'static_arena' in context.permitted_optimizations
                         and context.sequential_execution and context.successful_allocations
                         and context.non_reentrant_entry)
        options[argument] = policy
        selection[name] = {'status': 'policy_selected_not_applied', 'permissions_admitted': permitted,
                           'reason': None if permitted else 'permission_not_admitted', 'carrier': carrier,
                           'policy': json.loads(json.dumps(policy.record()))}
        if name == 'propagate_layouts':
            selection[name]['source_stage'] = 'after_source_projection' if composed else 'before_group_binding'
        elif descriptor_effects is not None:
            # A request/policy never supplies opaque-call effects. Only an
            # explicitly selected original factory can join the ordinary shim.
            from merlin.runtime.device_descriptor_results import _callback_source

            owner = _callback_source(descriptor_effects)
            selection[name]['descriptor_results'] = {
                'status': 'deferred' if composed or not permitted else 'original_factory_selected_not_qualified',
                'reason': 'composed_route_unsupported' if composed else None if permitted else 'permission_not_admitted',
                'original_factory': {'path': str(owner), 'sha256': sha256(owner.read_bytes()).hexdigest(),
                                     'function': descriptor_effects.__qualname__},
                'effect_grant': False,
            }
            if permitted and not composed:
                options.update(descriptor_results=True, descriptor_effects=descriptor_effects)
    return options, selection


def require_host_preparation(receipt, options, selection=None):
    """Rejoin actual build selections and original retained transform products."""
    from merlin.llvmlower.borrowed_arena import receipt_borrowed_lifetime
    from merlin.llvmlower.completed_host_storage import receipt_completed_storage, remember_completed_storage
    from merlin.llvmlower.graph_optimizations import recheck_host_graph_artifacts
    from merlin.llvmlower.host_storage_policy import recheck as recheck_storage

    for chosen in (selection or {}).values():
        carrier = chosen.get('carrier') or {}
        for pin in carrier.values():
            path = Path(pin['path'])
            if (any(part.is_symlink() for part in (path, *path.parents))
                    or sha256(path.read_bytes()).hexdigest() != pin['sha256']):
                raise ValueError('original host preparation carrier changed')
    device = receipt.get('inputs', {}).get('device') or {}
    output = receipt.get('output') or {}
    if options.get('descriptor_results', False):
        if (device.get('descriptor_results') is not True
                or output.get('descriptor_results', {}).get('selected') is not True):
            raise ValueError('owned host descriptor request lacks its actual consumed shim selection')
        original = (selection or {}).get('reuse_intermediate_storage', {}).get('descriptor_results', {}).get('original_factory')
        if original is not None and sha256(Path(original['path']).read_bytes()).hexdigest() != original['sha256']:
            raise ValueError('owned host original descriptor effects factory changed')
    for argument, recheck in (('pre_group_layout', recheck_host_graph_artifacts),
                              ('intermediate_storage', recheck_storage)):
        policy = options.get(argument)
        expected = json.loads(json.dumps(policy.record())) if policy is not None else None
        if json.loads(json.dumps(device.get(argument))) != expected:
            raise ValueError('owned host preparation differs from original software selection')
        if policy is None:
            if argument in output:
                raise ValueError('unselected owned host preparation claimed emitted products')
            continue
        policy.revalidate()
        artifacts = output.get(argument)
        if not isinstance(artifacts, dict) or 'report' not in artifacts:
            raise ValueError('selected owned host preparation lacks original retained products')
        if (argument == 'pre_group_layout'
                and (selection or {}).get('propagate_layouts', {}).get('source_stage') == 'after_source_projection'
                and 'source_binding' not in artifacts):
            raise ValueError('composed host layout lacks its original projected source binding')
        directory = Path(artifacts['report']['path']).parent
        completed = receipt_completed_storage(receipt) if argument == 'intermediate_storage' else None
        if completed is not None:
            observed = completed.recheck(directory, policy, receipt)
        else:
            live = ({'borrowed_lifetime': receipt_borrowed_lifetime(receipt)}
                    if argument == 'intermediate_storage' else {})
            observed = recheck(directory, policy, **live)
        if observed != artifacts:
            raise ValueError('owned host preparation products changed after compilation')
        if argument == 'intermediate_storage' and completed is None:
            remember_completed_storage(receipt, policy, observed)

    require_host_codegen(receipt, options)


def require_host_codegen(receipt, options):
    """Rejoin the existing explicit late-RNE source proof, not a requested flag."""
    host_codegen_evidence(receipt, options)
    if 'host_bounded_rne' not in options:
        return
    from merlin.llvmlower.host_bounded_rne import require_report, selected

    enabled = selected(options['host_bounded_rne'])
    marker = receipt.get('inputs', {}).get('host_bounded_rne')
    expected = {'host_isa': 'rv64gc', 'combine_clamp': True} if enabled else None
    if marker != expected:
        raise ValueError('owned host bounded RNE differs from its original selected request')
    if not enabled:
        return
    recipe_pin = receipt.get('output', {}).get('compilation_recipe') or {}
    recipe_path = Path(recipe_pin['path'])
    if sha256(recipe_path.read_bytes()).hexdigest() != recipe_pin['sha256']:
        raise ValueError('owned host bounded RNE compilation recipe changed')
    recipe = json.loads(recipe_path.read_bytes())
    report_pin = recipe.get('preparation', {}).get('host_bounded_rne') or {}
    report_path = Path(report_pin['path'])
    if sha256(report_path.read_bytes()).hexdigest() != report_pin['sha256']:
        raise ValueError('owned host bounded RNE source proof changed')
    require_report(json.loads(report_path.read_bytes()), require_combined=True)


def host_codegen_evidence(receipt, options):
    """Reopen only selected codegen products within the already audited build.

    Fusion proves which pipeline executed, not that a pattern rewrote or saved
    time. Round-even proves the selected helper's compile/link custody. The
    original completed-recipe/ELF audit and numerical checks remain mandatory.
    """
    names = ("host_roundeven", "host_elementwise_fusion")
    inputs, output = receipt.get('inputs') or {}, receipt.get('output') or {}
    enabled = []
    for name in names:
        if name not in options:
            continue
        choice = options[name]
        if type(choice) is not bool:
            raise ValueError('host codegen requests must be explicit Boolean choices')
        expected = ({'host_isa': 'rv64gc', 'abi': 'lp64d', 'symbol': 'roundevenf'}
                    if name == 'host_roundeven' else True) if choice else None
        if (inputs.get(name) != expected
                or name == 'host_elementwise_fusion' and choice and inputs.get(name) is not True):
            raise ValueError('owned host codegen differs from its original selected request')
        if choice:
            enabled.append(name)
    if not enabled:
        return {}

    def reopen(pin):
        path = Path(pin['path'])
        if (not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents))
                or sha256(path.read_bytes()).hexdigest() != pin['sha256']):
            raise ValueError('owned host codegen product changed')
        return path

    recipe_pin = output.get('compilation_recipe') or {}
    recipe = json.loads(reopen(recipe_pin).read_bytes())
    if recipe.get('status') != 'completed':
        raise ValueError('owned host codegen needs the original completed recipe')
    preparation = recipe.get('preparation') or {}
    evidence = {}
    if 'host_roundeven' in enabled:
        from merlin.runtime import host_rounding

        implementation = reopen(preparation['host_roundeven_implementation'])
        if implementation.read_bytes() != Path(host_rounding.__file__).read_bytes():
            raise ValueError('owned host roundeven implementation differs from selected shared source')
        commands = recipe.get('commands') or []
        assembly = host_rounding.roundeven_f32_assembly(isa='rv64gc', abi='lp64d', symbol='roundevenf')
        compiles = [c for c in commands if Path(c.get('output', {}).get('path', '')).name == 'host_roundeven.o']
        if len(compiles) != 1 or compiles[0].get('status') != 'returned':
            raise ValueError('owned host roundeven lacks one original helper compilation')
        command = compiles[0]
        sources = [p for p in command['inputs'] if Path(p['path']).name == 'host_roundeven.S']
        if len(sources) != 1 or reopen(sources[0]).read_text() != assembly:
            raise ValueError('owned host roundeven assembly differs from selected shared implementation')
        helper = command['output']
        reopen(helper)
        link = commands[-1]
        if (link.get('status') != 'returned' or helper not in link.get('inputs', [])
                or not any('-DMERLIN_EXTERNAL_ROUNDEVEN_F32' in c.get('argv', []) for c in commands)):
            raise ValueError('owned host roundeven helper is not in the original final link')
        evidence['host_roundeven'] = {
            'status': 'linked_shared_implementation', 'implementation_sha256': preparation[
                'host_roundeven_implementation']['sha256'], 'object_sha256': helper['sha256'],
            'rewritten_work': 'unmeasured', 'performance_qualified': False,
        }
    if 'host_elementwise_fusion' in enabled:
        from merlin.llvmlower import impr_features

        implementation = reopen(preparation['host_elementwise_fusion_implementation'])
        if implementation.read_bytes() != Path(impr_features.__file__).read_bytes():
            raise ValueError('owned host fusion implementation differs from selected shared source')
        pin = preparation['host_elementwise_fusion']
        report = json.loads(reopen(pin).read_bytes())
        pipeline = report.get('effective_pipeline')
        anchor = 'func.func(linalg-generalize-named-ops),func.func(linalg-fuse-elementwise-ops)'
        if (report.get('schema') != 'merlin.host_elementwise_fusion.v1'
                or report.get('feature') != impr_features.FUSE_AFTER_GENERALIZE_NAME
                or not isinstance(pipeline, str) or anchor not in pipeline
                or pipeline != (output.get('index_lowering') or {}).get('effective_pipeline')):
            raise ValueError('owned host fusion lacks the selected effective lowering pipeline')
        evidence['host_elementwise_fusion'] = {
            'status': 'pipeline_selected', 'report_sha256': pin['sha256'],
            'rewritten_work': 'unmeasured', 'performance_qualified': False,
        }
    return evidence
