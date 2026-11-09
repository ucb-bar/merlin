"""Convolution, rolling-window and spatial readout discovery contracts.

Pure discovery metadata; no admission or execution authority.
"""

from __future__ import annotations


def records(prefix):
    """Build fresh records in their original family order."""
    return (
        {
            "name": "source_exact_quantized_spatial_readout_cache",
            "symbols": [
                prefix + "spatial_readout_cache." + name
                for name in (
                    "SpatialReadoutCache",
                    "propose_spatial_readout_cache",
                    "emit_cached_spatial_max",
                )
            ],
            "usage": [
                "Propose retention from the original RollingPackedSpatialPlan and disjoint caller-owned windows.",
                "Keep original Q1 stores, Q2 conversion, clipped halos and all source liveouts.",
                "Emit counted initial/frontier reloads through the selected synchronous lower callback.",
            ],
            "requires": [
                "supported 3x3 stride-2 pad-1 signed-i8 source with original producer ReLU",
                "exact unchanged i8-to-i32 reload and original pooled readout binding",
                "producer restricted to its assigned disjoint accumulator window",
                "actual reload and readout completion; storage capacity supplies no target capability",
            ],
        },
        {
            "name": "capacity_phased_convolution_rhs_residency",
            "symbols": [
                prefix + "convolution_exploration." + name
                for name in (
                    "ConvolutionRhsPhaseLaw",
                    "ConvolutionRhsPhases",
                    "plan_convolution_rhs_phases",
                    "emit_convolution_schedule",
                    "convolution_pass_space",
                    "ConvolutionStorageAlternative",
                    "plan_convolution_storage_alternative",
                    "convolution_storage_pass_space",
                )
            ],
            "usage": [
                "Keep full-fit RHS residency, else derive complete-K output-column phases from assigned capacity.",
                "Pass the complete owner and exact resident input_rows to emit_convolution_schedule; never emit "
                "phases alone.",
                "The registered rhs_modes=(False, True, 'phased') route retains original column tails and actual-byte"
                " dedup.",
                "Check phase_emissions against the requested owner, retaining real preparation and phase-last-use "
                "drains.",
                "Jointly explore caller-qualified resident input maps and output group bounds before freezing RHS "
                "phase capacity; rank actual emitted complete-call cycles.",
                "Supply rhs_column_limits=(None, ...) to convolution_storage_pass_space for explicit narrower "
                "RHS phases; None retains the maximal-capacity default. plan_convolution_rhs_phases uses "
                "max_column_blocks and plan_convolution_storage_alternative uses rhs_column_blocks.",
                "Derive a useful finite column bound from actual ACC rows divided by requested live spatial "
                "tiles times primitive rows; keep the real operand capacity and exact RHS/input placements.",
                "Retain structured spatial coordinates outside K/N loops; resident input emit_row_coordinates "
                "maps them directly without repeated flattened-coordinate decoding.",
            ],
            "requires": [
                "original independent output columns, immutable inputs and no intermediate output observers",
                "exclusive assigned RHS rows disjoint from the actual resident input footprint",
                "separate RHS preservation and ConvolutionRhsPhaseLaw; capacity alone grants no reordering",
                "real full completion before private RHS replacement; source/effect/numerical checks remain required",
            ],
        },
        {
            "name": "source_qualified_convolution_readout_epochs",
            "symbols": [
                prefix + "window_transfer.ConvolutionReadoutEpochLaw",
                prefix + "convolution_exploration.emit_convolution_schedule",
            ],
            "usage": [
                "Bind a separate selected source readout law; compute-epoch or dense readout laws do not imply it.",
                "Pass readout_law explicitly to the convolution consumer with original source/target callbacks.",
                "Verify the returned readout_epoch, source/schedule and final block drain match the requested law.",
                "Keep intermediate compute, rolling input-ready/last-use and whole-call completion drains.",
                "Inspect actual emitted effects, full outputs and same-domain complete-call cycles; UNKNOWN keeps the"
                " default.",
            ],
            "requires": [
                "counted convolution with issued compute epochs and disjoint original output/ACC intervals",
                "selected ACC RAW/result visibility, ordered store-configuration snapshots and finite queue progress",
                "separate finite pending store capacity and bounded compiler-issued readouts; no extra hardware slots",
                "private ownership and a real final drain before input/ACC/RHS reuse",
                "target effect/completion adapter; consumer receipts do not independently prove emitted instruction "
                "semantics",
            ],
        },
        {
            "name": "pointwise_convolution_dense_view",
            "symbols": [prefix + "pointwise_convolution.pointwise_dense_view"],
            "usage": [
                "Derive a zero-copy matrix view from actual unit-stride unpadded 1x1 NHWC source geometry.",
                "Pass the returned dimensions to shared dense planning, retention and scheduling.",
                "Use the existing target encoder and preserve bias, quantization and all output observers.",
                "Retain spatial lowering when geometry or storage is incompatible; "
                "compare emitted alternatives by cycles.",
            ],
            "requires": [
                "original Window2D geometry and exact result shape",
                "physical operand layout and ABI agreement; target effects and numerical checks remain required",
            ],
        },
        {
            "name": "registered_source_convolution_exploration",
            "symbols": [
                prefix + "convolution_exploration." + name
                for name in (
                    "ConvolutionExplorationSource",
                    "ConvolutionScheduleChoice",
                    "plan_convolution_schedule",
                    "ConvolutionRhsResidencyLaw",
                    "ConvolutionRhsResidency",
                    "plan_convolution_rhs_residency",
                    "emit_convolution_schedule",
                    "convolution_pass_space",
                    "explore_convolution_schedules",
                )
            ],
            "usage": [
                "Bind original counted convolution, complete scalar-safe column intervals and selected private rows.",
                "Register bounded choices with original-source verification and target realization callbacks.",
                "Enable compute_pair_limits with a separate checked epoch law; stores still fully drain.",
                "Select rhs_modes with an independent immutable-weight law and explicit private operand rows.",
                "Emit invocation-local RHS preparation once, or retain original transfers when capacity declines.",
                "Consume the checked schedule through actual target emission; deduplicate exact program bytes.",
                "Compare complete same-source/model/window cycle feedback; no count-based default ranking.",
            ],
            "requires": [
                "separate original input-retention, delayed-load and compute-epoch laws for enabled transformations",
                "actual completion callbacks, private ACC ownership and original source custody",
                "ordinary source/dependency admission; registry/identities confer no hardware or numerical authority",
            ],
        },
        {
            "name": "source_exact_spatial_matrix_gathers",
            "symbols": [
                prefix + "affine_gather_runs." + name
                for name in (
                    "ConvolutionGatherRuns",
                    "emit_convolution_gather_runs",
                    "emit_convolution_matrix_gather",
                )
            ],
            "usage": [
                "Map logical contraction rows and reduction tiles to original valid-window NHWC addresses.",
                "Use reduction tiles dividing the source channel dimension; preserve source tap/channel order.",
                "Pass exact fragments to the ordinary target consumer without allocating host im2col payloads.",
            ],
            "requires": [
                "original checked window geometry and independent dtype, storage, alias and completion admission",
                "fragment construction supplies no permission to overlap transfers or change arithmetic",
            ],
        },
        {
            "name": "source_bound_rolling_convolution_rows",
            "symbols": [
                prefix + name
                for name in (
                    "window_transfer.ConvolutionResidentInputRows",
                    "convolution_exploration.emit_convolution_schedule",
                    "rolling_window.materialize_rolling_window",
                    "rolling_window.emit_rolling_window",
                    "rolling_window.materialize_grouped_rolling_window",
                    "rolling_window.verify_grouped_rolling_window",
                    "rolling_window.emit_grouped_rolling_window",
                )
            ],
            "usage": [
                "Bind original NHWC source cells to caller-owned tiled operand rows and a checked ring lifetime.",
                "Pass input_rows to the ordinary convolution schedule emitter; consume all staging effects.",
                "Derive transfer widths, row pitch and private storage from selected hardware/source contracts.",
                "Padded integer convolution requires explicit original zero halos, not a host padded tensor.",
                "Keep real input-ready and last-use completion; compare whole-call cycles and resource traffic.",
            ],
            "requires": [
                "exact original source identity, source stride and disjoint live operand storage",
                "nonunit compute row steps require an explicitly selected source/target row-step contract",
                "target implementations of actual stage/fill effects and completion; mapping grants no overlap",
                "nonzero affine padding values require a separately supported source and lowering contract",
            ],
        },
        {
            "name": "checked_cross_spatial_array_reuse",
            "symbols": [
                prefix + name
                for name in (
                    "convolution_output_groups.plan_rolling_convolution_output_group",
                    "convolution_output_groups.emit_rolling_convolution_output_group",
                    "dense_array_retention.DenseArrayRetentionAdmission",
                    "convolution_exploration.emit_convolution_schedule",
                )
            ],
            "usage": [
                "Derive source-row/M bounds from the original schedule, resident RHS or complete phase owner, and"
                " live ring.",
                "Pass the group or phase-aligned tuple and separate array_law to emit_convolution_schedule; None "
                "retains an ordinary phase.",
                "Consume actual fresh_weights through the same target encoder; preserve each original N readout.",
                "Verify spatial_output_group and its source/law hashes; keep the incumbent when planning declines.",
                "With a checked multirow ring union, grouped_transport=True completes staging once before the "
                "actual consumer group and drains before reuse.",
            ],
            "requires": [
                "resident full-K RHS per active original phase, checked source-row union in actual ring slots and"
                " disjoint M/N ACC cells",
                "separate independent-output-row/no-intermediate-observer source permissions",
                "selected fresh/stay/output-descriptor behavior and actual bounded compute/readout completion",
                "default one-row path retained; real group/ring drains and full phase-last-use before RHS replacement",
                "grouped transport separately checks actual row/halo/channel command count against the selected "
                "issue bound",
            ],
        },
        {
            "name": "source_exact_convolution_readout_transfers",
            "symbols": [
                prefix + name
                for name in (
                    "window_transfer.ConvolutionReadoutTransferLaw",
                    "window_transfer.plan_convolution_readout_transfers",
                    "convolution_exploration.emit_convolution_schedule",
                )
            ],
            "usage": [
                "Bind a separate selected narrow block-column/row-stride map and original numerical states.",
                "Pass readout_transfer_law with the independent producer/readout law to the original consumer.",
                "Only compatible original adjacent stores merge; inspect actual emitted stores and complete outputs.",
                "Compose with checked spatial groups using each spatial slot's actual ACC block map, including "
                "short source rows and RHS phases.",
            ],
            "requires": [
                "scaled INT8 source exits; exact controller map, output columns and private physical ACC spacing",
                "identical original numerical state within each combined run; full ACC and pooling excluded",
                "separate producer visibility/store snapshot law and real final private-block drain",
                "missing target/model support remains UNKNOWN, not an inferred speedup or qualification",
            ],
        },
        {
            "name": "completed_input_with_rhs_frontier",
            "symbols": [
                prefix + "convolution_input_reuse." + name
                for name in (
                    "ConvolutionInputReuseAdmission",
                    "select_convolution_input_reuse",
                    "InvariantConvolutionInput",
                )
            ]
            + [
                prefix + "rolling_lookahead.RollingRhsFrontierAdmission",
                prefix + "convolution_exploration.emit_convolution_schedule",
            ],
            "usage": [
                "Establish a completed original full-input union through the first actual RHS phase.",
                "Select invariant_input_law and rhs_phase_prefetch_law independently, then pass both to the "
                "same ordinary emitter.",
                "The checked consumer advances complete reduction runs before issuing future RHS writes into "
                "their dead prefix.",
                "Keep all original preparation, group, input-last-use, phase and final drains; inspect actual "
                "reader/compute overlap and cycles.",
            ],
            "requires": [
                "the complete original input union fits its injective assigned rows and remains immutable",
                "every future RHS write is disjoint from retained input under its exact source/phase map",
                "separate target RAW/WAR, ordered configuration, queue progress and real completion contracts",
                "one complete column consumer per phase; unsupported compositions retain separate schedules",
            ],
        },
        {
            "name": "staged_spatial_producer_bands",
            "symbols": [
                prefix + "spatial_producer." + name
                for name in (
                    "SpatialProducerBand",
                    "SpatialProducerBandAdmission",
                    "SpatialProducerBandAlternative",
                    "spatial_producer_band_choices",
                    "emit_spatial_producer_band",
                    "spatial_producer_pass_space",
                    "explore_spatial_producer",
                )
            ]
            + [prefix + "convolution_exploration.emit_convolution_rhs_preparation"],
            "usage": [
                "Match the actual adjacent convolution/pool source and retain both original rounded readouts "
                "and every external observer.",
                "Enumerate band heights from assigned ACC and Q1 scratch capacity; inspect exact halo "
                "recomputation and MAC costs.",
                "Emit the producer union, actual Q1 stores/reloads and a complete multirow Q2 pool rectangle "
                "through selected target callbacks.",
                "Optionally compose an existing complete immutable RHS owner; real preparation and final "
                "completion remain required.",
                "Use the ordinary pass registry to explore independent band/RHS choices and legal planning "
                "orders, deduplicate actual emitted bytes, "
                "then compare exact whole-region cycles. Unknown cycles remain unranked.",
                "The ordinary connected host route uses the original band.source plan and full assigned "
                "scratch; rolling packed/readout-cache "
                "paths retain their separate representation prerequisites.",
            ],
            "requires": [
                "original pure producer with no additional live intermediate observer; original Q1/Q2 scalar "
                "coefficients and signed padding",
                "separate source-bound rectangular pooling address/configuration and command-tracking law with "
                "actual field limits",
                "exclusive assigned ACC/Q1 storage and actual completion of the entire pooled rectangle before reuse",
                "actual original source callback and native execution; this library supplies no host effects "
                "or target qualification",
            ],
        },
        {
            "name": "disjoint_future_row_transport",
            "symbols": [
                prefix + "rolling_lookahead." + name
                for name in (
                    "RollingLookaheadAdmission",
                    "materialize_lookahead_rolling_window",
                    "verify_lookahead_rolling_window",
                    "emit_lookahead_rolling_window",
                )
            ]
            + [prefix + "convolution_output_groups.ConvolutionGroupContinuation"],
            "usage": [
                "Keep the original rolling source, row ownership, and current consumer unchanged.",
                "Use actual ring capacity for the union of current and future live rows.",
                "Emit counted SCF with carried resident handles and real transport/consumer completion.",
                "Optionally derive consumer_runs = kernel_y * kernel_x * ceil(input_channels/channel_tile) "
                "from the original consumer; pass it unchanged to materialize, verify and emit.",
                "Bind start_consumer to initialization and a retained continuation; advance_consumer "
                "emits first_run/run_count complete reductions; finish_consumer performs final readout.",
                "For rolling convolution use emit_rolling_convolution_output_group(..., defer_consumer=True); "
                "its returned continuation checks advance(first, count) coverage before finish().",
                "For a complete invocation pass input_prefetch_law=RollingLookaheadAdmission(...) to "
                "emit_convolution_schedule with its checked output group and grouped transport; original K runs "
                "are derived from source. The separately selected RHS-frontier mode remains an alternative.",
                "Compare complete-call cycles; earlier issue alone does not establish hidden latency.",
            ],
            "requires": [
                "disjoint current/future writes and an immutable original input",
                "independently bound ordered configuration, finite queue progress, and completion callbacks",
                "split consumers additionally require retained_consumer_continuations=True, live ACC/input "
                "storage, complete fresh/stay runs and exactly one original initialization/readout",
                "no arbitrary nested asynchronous schedule or fixed-cycle completion is implied",
            ],
        },
        {
            "name": "direct_ring_convolution_row_fragments",
            "symbols": [
                prefix + "convolution_output_groups." + name
                for name in (
                    "ConvolutionRowFragmentGroup",
                    "plan_rolling_convolution_row_fragments",
                    "emit_rolling_convolution_output_group",
                )
            ],
            "usage": [
                "Request a capacity-safe source-row band with plan_rolling_convolution_row_fragments.",
                "Pass its checked group to emit_convolution_schedule with the original rolling input, "
                "array/readout laws and real completions; the default whole-row planner is unchanged.",
                "Existing ConvolutionGatherRuns splits direct ring reads at original row boundaries. "
                "Fragments compute into disjoint subrows of packed ACC tiles; readout covers each complete "
                "original contiguous interval.",
                "Compare actual fragment/preload count, padded work, bytes and complete-call cycles; "
                "fewer ACC/readout tiles can require more compute fragments.",
            ],
            "requires": [
                "primitive-sized original row tiles, complete source-row bands and the actual staged ring horizon",
                "original immutable input/RHS, independent output rows and separately selected array/readout laws",
                "all original products, per-output reduction order, scalar exits, real pair boundaries and "
                "final drains",
                "no extra DMA bytes, local copy instruction or packed LHS allocation is inferred",
            ],
        },
        {
            "name": "last_use_rhs_phase_transport",
            "symbols": [
                prefix + "rolling_lookahead." + name
                for name in (
                    "RollingRhsFrontierAdmission",
                    "RollingRhsFrontier",
                )
            ]
            + [prefix + "convolution_exploration.emit_convolution_schedule"],
            "usage": [
                "Plan complete original RHS phases and their checked rolling output groups using actual SP/ACC "
                "capacity.",
                "Supply rhs_phase_prefetch_law=RollingRhsFrontierAdmission(...) to emit_convolution_schedule "
                "with grouped_transport=True and the original phase-aligned output_group tuple.",
                "The emitter derives the original reduction run count and stages each next-phase tile after "
                "the complete current K/N/M run in only the final source batch and spatial group.",
                "The next phase uses the same assigned RHS prefix; its column-block count must not grow. "
                "All original preparation, group, phase and final drains remain.",
                "Compare exact outputs and complete-call cycles with the same placement and numerical source.",
            ],
            "requires": [
                "separate selected target law covering exact-range scratchpad WAR/RAW until real source completion",
                "original immutable weights, ordered configuration, finite queue progress and full "
                "consumer/transport drain",
                "complete original batch/spatial domain and original channel tails; no second ACC or RHS window",
            ],
        },
        {
            "name": "bounded_horizontal_convolution_panels",
            "symbols": [
                prefix + "horizontal_convolution." + name
                for name in (
                    "HorizontalConvolutionPanels",
                    "plan_horizontal_convolution_panels",
                    "emit_horizontal_convolution_panels",
                )
            ],
            "usage": [
                "Bind the original CountedConvolutionPlan and explicit private operand/accumulator capacities.",
                "Plan bounded contiguous horizontal reductions and complete output-column intervals.",
                "Materialize the original zero halo; emit counted effects with actual completion before reuse.",
            ],
            "requires": [
                "unchanged source reduction order, output precision, bias and every tail output",
                "separate multiblock transport and fresh/stay array laws; private ownership is not a target grant",
                "complete-call cycles including halo/preparation cost; no speedup follows from transfer counts",
            ],
        },
    )
