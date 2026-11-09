"""Source layout, binding, integer observers and bounded choice contracts.

Pure discovery metadata; no admission or execution authority.
"""

from __future__ import annotations


def records(prefix):
    """Build fresh records in their original family order."""
    return (
        {
            "name": "declared_persistent_scalar_state_in_scf",
            "symbols": [
                prefix + "scf_commands." + name
                for name in (
                    "materialize_scf_commands",
                    "verify_scf_commands",
                    "unroll_small_command_loops",
                    "group_scf_command_loops",
                    "emit_scf_commands",
                )
            ]
            + [prefix + "llvm_program.FnBuilderIndexAdapter", "merlin.llvmlower.host_builder.FnBuilder"],
            "usage": [
                "Attach closed signed-i64 Const sets/assumes to actual command Calls.",
                "Pass allow_persistent_state=True explicitly at every SCF bridge boundary.",
                "Print/parse or transform actual command-bearing SCF, reverify, then emit through the callback.",
            ],
            "requires": [
                "unknown initial state; preceding setter, matching value and compatible loop reentry",
                "selected target visibility/clobber law remains NOT_ESTABLISHED by this scalar-state checker",
                "explicit async dependencies, source, alias, completion and numerical obligations remain",
            ],
        },
        {
            "name": "current_source_quantized_residual_edges",
            "symbols": ["merlin.llvmlower.quantized_sum_source.discover_quantized_sum_sources"],
            "usage": [
                "Discover current sole-use contraction-to-residual edges from actual host SSA.",
                "Inspect each unchanged producer/readout and residual/readout program and source shape.",
                "Try the target composition adapter before selecting an ordinary QuantizedSumRoute.",
                "Keep unsupported alternatives separate; compare complete region cycles and private workspace.",
            ],
            "requires": [
                "current ordinary DeviceRewrite and private complete borrowed source declarations",
                "fresh unobserved intermediate, preserved Q1/Q2, target effects and actual completion",
                "independent numerical execution and target admission; discovery supplies neither",
            ],
        },
        {
            "name": "parallel_emitted_variant_shape_evaluation",
            "symbols": [
                "merlin.transforms.evaluated_exploration." + name
                for name in (
                    "EvaluationRequest",
                    "EvaluationCompletion",
                    "evaluate_emitted",
                    "explore_evaluated",
                )
            ],
            "usage": [
                "Apply real registered passes and emit distinct alternatives with explore_candidates.",
                "Interleave requests from representative shapes; each retains its original source and timing domain.",
                "Use one caller-owned executor and ordinary resource-leased evaluator, with bounded max_in_flight.",
                "Keep completion publication compact; verification runs in the evaluator and cleanup off the hot path.",
                "Consume each completed result immediately; identical programs reuse only exact-domain evidence.",
                "Rank complete same-ROI cycles within each shape; UNKNOWN never becomes an instruction-count score.",
                "Promote measured shape/resource rules through shape_policy while retaining unaffected incumbents.",
            ],
            "requires": [
                "target-owned emitted-program execution adapter and original numerical/provider/ROI identities",
                "existing shared resource admission; this library creates no CPU pool or simulator",
                "input lifetime covering all evaluation workers and the completion callback",
            ],
        },
        {
            "name": "selected_logical_kernel_argument_binding",
            "symbols": [
                "merlin.compile.logical_abi." + name
                for name in (
                    "logical_abi",
                    "logical_abi_from_document",
                    "logical_interface",
                    "kernel_arg_order",
                )
            ]
            + ["merlin.llvmlower.host_builder.FnBuilder.for_logical_kernel"],
            "usage": [
                "Parse the selected public ABI declaration, then call FnBuilder.for_logical_kernel(cb, abi=abi).",
                "Use the returned source-name to SSA-pointer map; semantic lhs/rhs order is not ABI order.",
                "Implicit COMMIT and pooled result shapes share the runner's declaration-only derivation.",
            ],
            "requires": [
                "original command buffer and selected public ABI declaration; no inferred private pointer contract",
                "independent runtime ownership and numerical implementation; no harness/reference code is exported",
            ],
        },
        {
            "name": "shape_resource_schedule_selection",
            "symbols": [
                "merlin.transforms.shape_policy." + name
                for name in (
                    "PerformanceDomain",
                    "ShapeScheduleClass",
                    "ShapeCaseFeedback",
                    "assess_shape_policy",
                    "AssessedShapePolicy.execute",
                    "ShapePolicyProposal",
                    "propose_shape_policies",
                )
            ]
            + [
                "merlin.transforms.pipeline_policy." + name
                for name in (
                    "DimensionRef",
                    "ResourceRef",
                    "RatioFeature",
                    "TileFeature",
                    "FeaturePredicate",
                )
            ]
            + [
                "merlin.transforms.evaluated_exploration.explore_evaluated",
                "merlin.transforms.pass_exploration.ExplorationRefused",
                "merlin.transforms.pass_exploration.MappingProposalKey",
                "merlin.transforms.pass_exploration.MappingProposalCache",
            ],
            "usage": [
                "Register bounded real pass paths and an original fallback, with separate legality predicates.",
                "Freeze distinct training/heldout geometries and supply exact paired completed observations.",
                "Declare shape/resource performance guards; include exact cutoffs and adjacent outside cases.",
                "Propose bounded dimension cutoffs and two-axis regions from paired TRAINING cycles only.",
                "Use proposal.requests for missing pairs/boundaries; freeze new boundary geometries before timing.",
                "Express exact ratios, tile counts and tails from current shape/resource facts; unknown stays unknown.",
                "Training selects; heldout and boundary cases veto. UNKNOWN/regression keeps each class incumbent.",
                "Execute on current IR with its selected performance domain; no cached training kernel is reused.",
                "Reuse only option-search proposals in an explicit bounded MappingProposalCache keyed by "
                "semantics/shapes/layouts/dtypes/epilogue and hardware/library/numerics; current factories, "
                "legality and emission still run.",
                "Use existing evaluate_variants(prefix=...) for preparation shared by every alternative; "
                "it discards prefix analyses before current-source branch checks.",
                "Exploration outcomes distinguish duplicates, construction refusals, "
                "pending/UNKNOWN timing and regressions.",
            ],
            "requires": [
                "complete numerical-pass same-source/input/reference/provider/ROI cycle observations",
                "original registered pass contracts and independent source/resource/effect legality checks",
                "new normal inventory22 selection; empirical panel support is not universal or physical qualification",
            ],
        },
        {
            "name": "bounded_declared_settings_cycle_selection",
            "symbols": [
                "merlin.compile.tuning_knobs.KnobSettings",
                "merlin.compile.tuning_knobs.bounded_settings",
                "merlin.transforms.tuning_selection.KnobTrial",
                "merlin.transforms.tuning_selection.unique_emissions",
                "merlin.transforms.tuning_selection.select_settings",
            ],
            "usage": [
                "Enumerate a declared knob subset: exact baseline, single changes and optional bounded pairs.",
                "Emit and revalidate current choices; unique_emissions deduplicates exact same-domain bytes.",
                "Use the caller-owned ordinary evaluator and identity-bound CompletedObservation cycle feedback.",
                "select_settings returns an exact-shape proposal; use shape_policy for validated range rules.",
            ],
            "requires": [
                "original declaration, complete settings and semantic/shape/layout/dtype/numerical/hardware pins",
                "same-source/input/reference/provider/ROI observations; complete passing cycles alone rank",
                "UNKNOWN, failure, conflicts or no demonstrated improvement retain the unchanged baseline",
            ],
        },
        {
            "name": "factored_immutable_tensor_layouts",
            "symbols": [
                prefix + "tensor_storage." + name
                for name in (
                    "FactoredTensorStorage",
                    "column_panel_storage",
                )
            ],
            "usage": [
                "Factor original logical axes and select their physical order from the target's "
                "transfer/storage contract.",
                "Carry the same checked map through actual constant preparation, call-buffer types, target "
                "addresses and readback.",
                "Prepare source-owned immutable bytes once; include preparation in comparisons for mutable or "
                "invocation-local operands.",
                "Inspect emitted transfer addresses and strides; a layout declaration alone does not change "
                "target code.",
            ],
            "requires": [
                "exact positive source factors, a complete dense permutation and original byte extent",
                "independent immutable-source authorization and an executable consumer of the declared physical map",
                "original arithmetic, column coefficients and observers; unsupported tails or affine-only "
                "consumers keep their original layout",
                "compare whole-region cycles and traffic with the actual prepared constant; no preferred panel "
                "width is supplied",
            ],
        },
        {
            "name": "source_checked_integer_observers",
            "symbols": [
                "merlin.llvmlower.late_quant_rne.rewrite",
                "merlin.llvmlower.source_expression_interval.close_scalar_i8_observer",
                "merlin.llvmlower.source_expression_interval.build_source_interval_table",
                "merlin.llvmlower.source_expression_interval.emit_source_interval_i8_lookup",
                "merlin.llvmlower.source_expression_interval_llvm.rewrite_source_interval_i8_lookup",
                "merlin.llvmlower.bounded_rne_word_cells.prepare_bounded_rne_word_cells",
                "merlin.llvmlower.scaled_integer_finite.prove_broadcast_scaled_integer_finite",
                "merlin.llvmlower.scaled_integer_finite_llvm.bind_finite_scale_helpers",
            ],
            "usage": [
                "Bind exact typed scalar arithmetic and complete saturated signed-byte observations.",
                "Retain ordinary bounded roundeven representation and the existing late host legalizer.",
                "Derive finite broadcast-scale scans and immutable tables from the current source only.",
                "Consume returned LLVM/C in the ordinary compiler before object identity; "
                "validate the complete output.",
            ],
            "requires": [
                "explicit RNE, gradual underflow, nontrapping and unobserved floating effects",
                "original source fallback, actual incoming-mode guard and exact original coefficients/order",
                "immutable input scales and fresh disjoint outputs for finite preparation",
                "unknown source operations such as exp require their own numerical contract; no approximation granted",
                "no target/provider, reference output, protected evaluator or timing authority is exported",
            ],
        },
        {
            "name": "counted_projection_pointer_binding",
            "symbols": [
                prefix + "projection_dense." + name
                for name in (
                    "ProjectionDenseBinding",
                    "bind_dense_projection",
                )
            ],
            "usage": [
                "Keep original sibling projection_views, independent pointers and exact scalar readouts.",
                "Retain the common input with the existing dense scheduler; choose panels dividing every member.",
                "Partition complete virtual-N panel loops with ordinary ordered loop transforms.",
                "Bind actual member/local-column/row-width operands to original target ABI pointers and pitches.",
            ],
            "requires": [
                "current typed source group and an independently checked original dense schedule",
                "complete original per-column readout words; physical rectangles cannot cross members",
                "original source aliases, private storage, actual target effects and real completion",
                "no tensor concatenation, dynamic command expansion or execution grant follows from the mapping",
            ],
        },
        {
            "name": "completion_bound_operand_slots",
            "symbols": [
                prefix + "operand_slots." + name
                for name in (
                    "PipelineOperand",
                    "PipelineSlotResult",
                    "allocate_pipeline_operand_slots",
                    "verify_pipeline_operand_slots",
                )
            ],
            "usage": [
                "Group original immutable TemporaryObject generations by operand, retaining actual views.",
                "Supply StorageAdapter projections for initialization, reads, launches and real waits.",
                "Use the returned Kernel: existing allocation chooses bases and checked phase loops rotate them.",
                "Materialize/emit the actual returned SCF and compare whole-call timing independently.",
            ],
            "requires": [
                "one bounded source counted loop with integer-decoded completion operands and owned byte banks",
                "complete original effects, aliases, reservations, alignments and external source/output windows",
                "already legal issue order; no new overlap permission or hardware slots follow from allocation",
                "nested or SSA-carried asynchronous boundaries and irregular rotations currently refuse",
            ],
        },
        {
            "name": "sole_observer_contraction_rows",
            "symbols": [
                "merlin.llvmlower.contraction_row_slice.pushback_contraction_row_slices",
                "merlin.llvmlower.contraction_row_slice.prepare_contraction_row_slices",
            ],
            "usage": [
                "Before source/device binding, push back a sole static contiguous row observer.",
                "Supply original immutable argument bytes for stored epilogue coefficients when needed.",
                "Retain the original ordered integer dot and each selected scalar epilogue instruction.",
            ],
            "requires": [
                "closed zero-seeded signed integer product; complete reduction and output columns",
                "static row-major views, fixed coefficients, pure parallel single-observer epilogue",
                "new ordinary source/interface/target qualification; no old device-ABI transfer",
            ],
        },
        {
            "name": "source_proven_shared_rhs_row_views",
            "symbols": [
                "merlin.llvmlower.weight_prepack_runtime.fixed_arguments",
                "merlin.llvmlower.repeated_pointwise.hoist_repeated_pointwise",
                "merlin.llvmlower.repeated_pointwise.prepare_repeated_pointwise",
                "merlin.transforms.shared_rhs_views.plan_shared_rhs_row_views",
                "merlin.transforms.dense_row_batching.DenseRowBatchingBudgets",
                "merlin.transforms.dense_row_batching.plan_source_dense_row_batching",
            ],
            "usage": [
                "Before device grouping, use exact broadcast/view/scalar source to prepare repeated pointwise work.",
                "Stored coefficients require fixed_arguments from the original generated caller "
                "plus exact bound bytes.",
                "Expose complete matrix groups with plan_shared_rhs_row_views, then use checked dense row batching.",
                "Retain every original result and dense observer; "
                "revalidate actual source/SSA before consuming output.",
            ],
            "requires": [
                "source-proven coordinates and unchanged scalar programs; equal runtime values grant no sharing",
                "fresh source/session/manifest ownership for immutable specialization; literals need no loader",
                "new ordinary source/interface/target/effects and execution qualification after regrouping",
                "late host preparation after device grouping cannot revise an already selected device ABI",
            ],
        },
    )
