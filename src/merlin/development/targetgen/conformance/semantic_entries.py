"""Host, composition, epilogue and carried-state obligations for derived profiles."""

from __future__ import annotations

from merlin.development.targetgen.corpus.selection import (
    extents_for,
    narrowest_admitted,
)


def add_semantic_entries(
    *,
    SOURCE_ROLE,
    SYNTH_ACC_SCALE,
    SYNTH_PREFIX,
    _host_probe_op,
    _mark_source,
    _quanta,
    _signed_stimulus_range,
    _writer_for,
    admitted_dtypes,
    application_inventory,
    cells,
    declare_pool_window,
    declare_residual_add_params,
    entries,
    kept,
    op_for_shape,
    pass_requirements_for,
    permute_probe_layout,
    pool,
    probes,
    shape_unwritable,
    spec_doc,
    unexpressable,
):
    _diag = dict(spec_doc.get("diagnostics") or {})
    host_block = spec_doc.get("host_only") or {}
    host_dtypes = dict(host_block.get("dtypes") or {})
    unsized_host: list[str] = []
    # THE FAMILY BEING UNADMITTED IS NOT ENOUGH. A capsule proves the negative lane only if NOTHING in
    # its program may be accelerated, and a composite family DECOMPOSES: `normalization` is a reduction
    # and an elementwise map, so on a target admitting either, a normalization capsule contains regions
    # the hardware legitimately takes. Measured: every `SY_host_only_normalization` this axis emitted
    # classifies `A` under `boundary.profile_capsule`, so the coverage gate correctly declined to count
    # it -- gemmini's negative lane was covered by a HAND-AUTHORED capsule the whole time, and atlas's
    # was not covered at all while a synthesized capsule sat there asserting it.
    #
    # Worse than uncounted: `forbid: [on_mesh]` on such a program is a claim the submission must NOT do
    # something it is entitled to do, so a compiler that correctly accelerates the admitted elementwise
    # regions would be recorded as violating the lane.
    # WIDER THAN `host_only`, AND KEYED ON THE PAIR. `host_only` carries families with no admitted dtype
    # at all; `host_lane` carries every (family, dtype) the captures contain and the manifest does not
    # admit -- which on a target admitting one narrow format is most of a real model. Measured on
    # gemmini: four admitted families, every one of them ALSO present at f32, 10,719 regions of host work
    # the requirement demanded no capsule of.
    #
    # Whether such a capsule can honestly forbid the mesh is NOT decided here. It is a property of the
    # written program -- `rmsnorm` classifies accelerator at both f32 and bf16 where `layernorm`
    # classifies host-only at bf16, so the op decides it and the family does not -- and the generator
    # classifies the capsule and refuses a forbid the classification will not support. A family-level
    # rule tried here first and was wrong in both directions: it blocked `normalization` on gemmini,
    # where a layernorm capsule proves the lane, and passed it on a target whose rmsnorm program
    # contains an eligible region anyway.
    represented_host_pairs = {
        (str(family), str(host_dtypes[family]))
        for family in host_block.get("families") or ()
        if host_dtypes.get(family)
    }
    for _pair in (spec_doc.get("host_lane") or {}).get("required") or ():
        family, dtype = str(_pair.get("family") or ""), str(_pair.get("dtype") or "")
        if not family or not dtype or (family, dtype) in represented_host_pairs:
            continue  # the narrow axis below already carries this exact typed pair
        # `writer="pytorch"` because this is a HOST axis: see `op_for_family`. When the requirement
        # names the operations this pair's host work consists of, the probe is one of THOSE, never a
        # stock op of the same family that no captured model contains.
        op, observed = _host_probe_op(
            _pair,
            dtype,
            pool,
            native_inventory=application_inventory,
            native_inventory_sha256=(spec_doc.get("application_demands") or {}).get("full_inventory_sha256"),
        )
        if op is None:
            unsized_host.append(
                f"{family}/{dtype} ("
                + (
                    "none of the observed host operations "
                    f"{[row.get('op') for row in _pair.get('observed_ops') or ()]} has a PyTorch writer"
                    if _pair.get("observed_ops") is not None
                    else "no materializable op at this dtype"
                )
                + ")"
            )
            continue
        entry = {
            "cat": "model_slices",
            "kind": "model_slice",
            "name": f"{SYNTH_PREFIX}_host_lane_{family}_{dtype}".replace("-", "_").replace(".", "_"),
            "op": op,
            "operand_dtype": dtype,
            "out": "Y0",
            "lhs": "A0",
            "weight": "W",
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for the host lane: real captures contain {_pair.get('n_regions')} "
                f"{family!r} region(s) at {dtype}, and this target's manifest admits {family!r} at no "
                f"such dtype -- so every one of them must be placed on the host. A corpus with no "
                f"capsule here cannot tell a compiler that routes them correctly from one that does not"
            ),
            "label": "public",
            "lanes": {"forbid": ["on_mesh"]},
            "generalization": {
                "must_accelerate": False,
                "eligible": False,
                "generalization_axis": "host_lane",
                "not_asserted_reason": (
                    "the target admits no datapath for this family at this dtype; the capsule "
                    "exists to prove the compiler does NOT accelerate it"
                ),
            },
            **extents_for("aligned", probes, quantum=_quanta.get(dtype)),
        }
        if op == "permute":
            entry.update(permute_probe_layout(observed))
        if observed is not None:
            entry["frontend_op"] = observed.get("frontend_op")
            from merlin.development.targetgen.applications.native_storage import apply_probe_declaration

            apply_probe_declaration(entry, observed)
            entry["source_reference"] += (
                f". The probe is the observed host operation {op!r} ({observed.get('frontend_op')}, "
                f"{observed.get('n_regions')} captured region(s))"
            )
        # AN OP WITH A `merlin_iface` BUILDER CANNOT SERVE THIS AXIS, whatever dtype it declares.
        # `merlin_iface` is the ACCELERATOR's interface dialect -- every program expressible in it is
        # accelerator work by construction -- and the capsule writer derives an iface interface for any
        # op that has a builder, even on the frontend path. The boundary classifier reads that interface,
        # so the capsule comes out `A`. Measured: a matmul capsule declaring genuine f32 operands
        # (tensor<16x32xf32>) still classified accelerator, while `gelu` and `reduce_sum` -- which have
        # no iface builder, so their interface stays linalg and eligibility is judged dtype-aware --
        # classified host-only.
        #
        # That is what `_writer_for` already means by "pytorch": an op with no builder. Reported rather
        # than written another way, because there IS no other way: covering host-lane contraction needs a
        # contraction op the interface dialect cannot express, which the corpus does not yet have.
        if _writer_for(entry) != "pytorch":
            unsized_host.append(
                f"{family}/{dtype} (op {op!r} has a merlin_iface builder, so its capsule carries an "
                f"accelerator-dialect interface and classifies as accelerator work whatever dtype it "
                f"declares; this axis needs an op of this family the interface dialect cannot express)"
            )
            continue
        entry["source"] = "pytorch"
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    for family in sorted(str(f) for f in (host_block.get("families") or ())):
        dtype = host_dtypes.get(family)
        # `writer="pytorch"` because this is a HOST axis: see `op_for_family`. Observed operations
        # first, exactly as for the host-lane pairs above.
        _observed_rows = (host_block.get("observed_ops") or {}).get(family) if "observed_ops" in host_block else None
        op, observed = (
            _host_probe_op(
                {"observed_ops": _observed_rows},
                dtype,
                pool,
                native_inventory=application_inventory,
                native_inventory_sha256=(spec_doc.get("application_demands") or {}).get("full_inventory_sha256"),
            )
            if dtype
            else (None, None)
        )
        if op is None or not dtype:
            _why = (
                "no dtype for it is observed in any capture"
                if not dtype
                else "no op of this family is one the interface dialect cannot express, so every "
                "candidate classifies as accelerator work and the forbid could never be proven"
            )
            unsized_host.append(f"{family} ({_why})")
            continue
        entry = {
            "cat": "model_slices",
            "kind": "model_slice",
            "name": f"{SYNTH_PREFIX}_host_only_{family}".replace("-", "_"),
            "op": op,
            "operand_dtype": dtype,
            "out": "Y0",
            "lhs": "A0",
            "weight": "W",
            # PyTorch-sourced regardless of whether a direct-MLIR builder exists for the op: a host-only
            # capsule is a model slice at a float dtype, and the frontend-faithful lowering is what makes
            # it the program a real model would hand the host.
            "source": "pytorch",
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for host-only family {family!r}: real captures contain it and this "
                f"target's capability manifest admits no capability for it, so the compiler must leave "
                f"it on the host lane. dtype {dtype} is the one the captures carry for this family"
            ),
            "label": "public",
            "lanes": {"forbid": ["on_mesh"]},
            "generalization": {
                "must_accelerate": False,
                "eligible": False,
                "not_asserted_reason": (
                    "the target declares no capability for this family; the capsule exists to "
                    "prove the compiler does NOT accelerate it"
                ),
            },
            **extents_for("aligned", probes, quantum=_quanta.get(dtype)),
        }
        if observed is not None:
            entry["frontend_op"] = observed.get("frontend_op")
            from merlin.development.targetgen.applications.native_storage import apply_probe_declaration

            apply_probe_declaration(entry, observed)
            entry["source_reference"] += (
                f". The probe is the observed host operation {op!r} ({observed.get('frontend_op')}, "
                f"{observed.get('n_regions')} captured region(s))"
            )
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    # ---- the COMPOSITION axis -----------------------------------------------------------------------
    # The requirement derives which compositions real captures CONTAIN -- an isolated dispatch, adjacent
    # accelerator pairs, a host island, a full routing alternation -- and nothing consumed the set, so no
    # capsule anywhere demanded a host round trip. That is the shape a whole-model compiler actually gets
    # wrong, because it is the one where keeping an intermediate resident and paying to move it out
    # differ.
    #
    # ONE entry, not one per shape. The micro model's composition is whatever its derived layer inventory
    # produces once host layers are interleaved into the interior; emitting a capsule per declared shape
    # would mean inventing inventories no capture supports. The shapes this does not reach are reported.
    composition = dict((spec_doc.get("composition") or {}).get("required") or {})
    if composition:
        entry = {
            "cat": "model",
            "kind": "model",
            "name": f"{SYNTH_PREFIX}_micro_model",
            "micro_model": True,
            "source_role": SOURCE_ROLE,
            "source_reference": (
                "synthesized for the composition axis: one layer per admitted family, one per family a "
                "real capture contains that the manifest does not admit, sized to the target's own tile "
                "edge, with host layers interleaved into the interior. Inventory and extents come from "
                f"micro_model.spec; the shapes the requirement names are {sorted(composition)}"
            ),
            "label": "public",
            "lanes": {"require": ["on_mesh"]},
            "generalization": {"generalization_axis": "composition"},
        }
        _mark_source(entry)
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    # ---- the SHAPE-GENERALIZATION axis --------------------------------------------------------------
    # Rank and operand layout: the two things a `(family, dtype, alignment)` cell cannot say. A target's
    # manifest DECLARES that a unit batches, or transposes, or accepts a packed layout; `capability_probes`
    # has always turned those declarations into region descriptors, and nothing outside the fuzzer read
    # them -- so a target could claim batching and never be asked for a batched region by anything that
    # grades it.
    #
    # Reported, not raised, when no writer exists. A batched contraction is writable only through the
    # block-scaled builder today, and a transposed RHS needs a builder that does not exist (the interface
    # declares the weight [K, N] while a quantized `nn.Linear` stores it [N, K], which is a real feature
    # rather than a check to loosen). Those are capability gaps to argue about with the probe NAMED, and
    # the alternative -- leaving the axis out entirely, as before -- is the one that hides them.
    for req in (spec_doc.get("shape_generalization") or {}).get("required") or ():
        axis = str(req.get("axis") or "")
        family = str(req.get("family") or "")
        dtype = str(req.get("dtype") or "")
        probe = str(req.get("probe") or f"{family}.{axis}")
        # THE PROBE'S DTYPE FIRST, THEN THE TARGET'S OTHER ADMITTED ONES. This axis is about the SHAPE
        # capability -- can the unit be asked for a batched region at all -- so settling for a different
        # admitted format is a fair way to exercise it, where giving up would report a hole the target
        # does not have. `capability_probes` names the family's LEAD dtype, and on a multi-format target
        # that is not always the one a writer covers: mx-gemmini leads with mxfp4 while the batched
        # golden grades mxfp8 only, so the region is reachable and the lead dtype alone said it was not.
        _family_dtypes = [dtype] + sorted(
            {
                str(c.get("dtype"))
                for c in cells
                if c.get("dtype") and str(c.get("family")) == family and str(c.get("dtype")) != dtype
            }
        )
        op, chosen_dtype = None, dtype
        for _dt in _family_dtypes:
            op = op_for_shape(
                family, admitted_ops=pool, dtype=_dt, rank=int(req.get("rank") or 2), layout=req.get("layout")
            )
            if op is not None:
                chosen_dtype = _dt
                break
        dtype = chosen_dtype
        if op is None:
            shape_unwritable.append(
                f"{probe} ({axis} axis, family {family!r}, dtype {dtype!r}): no builder materializes a "
                f"{'rank-' + str(req.get('rank')) if axis == 'rank' else str(req.get('layout')) + '-layout'} "
                f"region for this family at this dtype"
            )
            continue
        entry = {
            "cat": "model_slices",
            "kind": "model_slice",
            "name": f"{SYNTH_PREFIX}_{axis}_{probe.replace('.', '_')}",
            "op": op,
            "operand_dtype": dtype,
            "out": "Y0",
            "lhs": "A0",
            "weight": "W",
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for the {axis} axis: this target's capability manifest declares "
                f"{family!r} handles a {probe.split('.')[-1]} region, and a (family, dtype, alignment) "
                f"cell cannot demand one. Shape and layout come from capability_probes"
            ),
            "label": "public",
            "generalization": {"generalization_axis": axis},
        }
        # EXTENTS COME FROM THE BUILDER, NOT THE PROBE. This axis asks whether the unit can be asked for
        # a batched or transposed region at all; WHICH extent is the alignment axis's question, and the
        # cells already carry it. The probe's tile-relative shape is a reasonable default for a plain
        # 2-D region and not necessarily a legal one for the op that expresses this region: the batched
        # MX golden needs its contraction dim a multiple of 32 where the probe offers one tile, so
        # passing the probe's extents through produced an entry that built its interface and then failed
        # in the golden. The builder derives a legal shape from the same tile edge the probe used.
        if int(req.get("batch") or 1) > 1:
            # `B` is the key the batched builder reads. Spelling it `batch` produced an entry that built
            # at the builder's DEFAULT batch instead of the probe's, so the capsule would have carried a
            # batch nobody derived while looking like it carried the declared one.
            entry["B"] = int(req["batch"])
        if req.get("layout"):
            entry["layout"] = str(req["layout"])
        _mark_source(entry)
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    # ---- the EPILOGUE axis --------------------------------------------------------------------------
    # WHICH stage rides the contraction. The cells already say that a fused-only family must be carried
    # as an epilogue rather than standalone -- but they cannot say WHICH one, so a corpus derived from
    # cells alone picks a single representative (gemmini got `relu` for elementwise_map and `acc_scale`
    # for reduction) and reports the fusion capability covered. The hand-authored corpus tests four
    # stages and their combinations, which is the level this axis is for.
    #
    # The requirement side evidences each stage from the manifest or from the target's own instruction
    # taxonomy; here we only have to write one contraction per stage.
    for _st in (spec_doc.get("epilogue") or {}).get("required") or ():
        stage = str(_st.get("stage") or "")
        if not stage:
            continue
        dtype = kept[0] if kept else narrowest_admitted(admitted_dtypes)
        if not dtype:
            unexpressable.append(f"epilogue axis: no admitted dtype to build a {stage!r} stage at")
            continue
        entry = {
            "cat": "layers",
            "kind": "layer",
            "name": f"{SYNTH_PREFIX}_epilogue_{stage}",
            "op": "matmul",
            "operand_dtype": dtype,
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
            "epilogue": [stage],
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for the epilogue axis: this target can fuse a {stage!r} stage onto a "
                "contraction (evidenced by the derived requirement), and a (family, dtype, "
                f"alignment) cell cannot demand a particular stage -- so without this the capability is "
                f"reported covered by whichever single stage the cell axis happened to pick"
            ),
            "label": "public",
            "generalization": {"generalization_axis": "epilogue"},
            **extents_for("aligned", probes, quantum=_quanta.get(dtype)),
        }
        declare_pool_window(entry)
        declare_residual_add_params(entry)
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    # ---- the MUST-REFUSE side of the epilogue axis -----------------------------------------------
    # A stage the requirement REJECTS on evidence (a declared readout that does not apply it, a derived
    # taxonomy with no class for it) is a computation this hardware does not perform. A backend that
    # emits it anyway either computes something the unit cannot do or silently moves pinned work, and
    # nothing would catch either: no capsule asked. One member per such stage asks, and its only
    # correct answer is the backend's stated decline (merlin.development.targetgen.grading.boundary). A stage left
    # UNRESOLVED is not evidence of anything and gets no member.
    from merlin.development.targetgen.corpus.binding import BUILDER_EPILOGUE_STAGES as _refusable

    for _st in (spec_doc.get("epilogue") or {}).get("rejected") or ():
        stage = str(_st.get("stage") or "")
        if not stage or stage not in set(_refusable):
            continue
        dtype = kept[0] if kept else narrowest_admitted(admitted_dtypes)
        if not dtype:
            continue
        entry = {
            "cat": "layers",
            "kind": "layer",
            "name": f"{SYNTH_PREFIX}_refuse_{stage}",
            "op": "matmul",
            "operand_dtype": dtype,
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
            "epilogue": [stage],
            "outcome": "refuse",
            "refusal": {
                "stage": stage,
                "reason": f"no readout or contraction-scoped route this target declares performs {stage!r}",
                "evidence": str(_st.get("why") or "rejected by the derived requirement's epilogue axis"),
            },
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for the epilogue axis: the requirement rejects {stage!r} on this target's "
                "own evidence, so the correct answer is a stated refusal and an emitted program fails"
            ),
            "label": "public",
            "stimulus_range": list(_signed_stimulus_range()),
            "generalization": {"generalization_axis": "epilogue_refusal"},
            **extents_for("aligned", probes, quantum=_quanta.get(dtype)),
        }
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    # ---- the GROUP axis -----------------------------------------------------------------------------
    # Complete readouts and column-dependent gains need hardware-width and tail cases on each
    # publicly observed carrier: scalar-only tests cannot distinguish an ignored gain vector.
    from merlin.development.targetgen.corpus.binding import BUILDER_EPILOGUE_STAGES as _builder_stages

    _stage_names = set(_builder_stages)
    # ONE MEMBER PER (STAGES, DTYPE). Signatures that differ only in their carrier (a contraction and
    # a convolution with the same readout) are written as the same matmul program, so they share one
    # member: two entries with the same name and the same program would be the same capsule twice.
    _group_members: dict[str, dict] = {}
    for _group in (spec_doc.get("groups") or {}).get("required") or ():
        if _group.get("column_readout"):
            from merlin.development.targetgen.corpus.derived_models import entry as column_entry

            member = column_entry(_group)
            member["pass_requirements"] = pass_requirements_for(member, spec_doc)
            entries.append(member)
            continue
        stages = [str(x) for x in (_group.get("epilogue") or ())]
        if len(stages) < 2:
            continue  # a single stage is the epilogue axis's member already
        unknown = [x for x in stages if x not in _stage_names]
        if unknown:
            unexpressable.append(
                f"group axis: {_group.get('signature')} carries {unknown}, which no capsule builder can state"
            )
            continue
        dtype = str(_group.get("dtype") or "") or (kept[0] if kept else narrowest_admitted(admitted_dtypes))
        if not dtype:
            unexpressable.append(f"group axis: no dtype to build {_group.get('signature')} at")
            continue
        name = f"{SYNTH_PREFIX}_group_{'_'.join(stages)}_{dtype}"
        if name in _group_members:
            member = _group_members[name]
            member["_group_count"] += int(_group.get("groups") or 0)
            member["source_reference"] = (
                f"synthesized for the group axis: selected captures form {member['_group_count']} "
                "compute group(s) with this stage combination on this "
                f"target, and no per-stage member demands a combination"
            )
            continue
        entry = {
            "cat": "layers",
            "kind": "layer",
            "name": name,
            "op": "matmul",
            "operand_dtype": dtype,
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
            "epilogue": stages,
            # A scale stage with no multiplier is read as 1.0 by every engine, which makes the
            # requantization a bare saturating cast that a backend ignoring the scale passes.
            **({"acc_scale": SYNTH_ACC_SCALE} if "acc_scale" in stages else {}),
            "source_role": SOURCE_ROLE,
            "source_reference": (
                "synthesized for the group axis: this target's derived output path applies this stage "
                "combination to a contraction on move-out, and no per-stage member demands a combination"
                if "derived_output_epilogue" in (_group.get("evidenced_by") or ())
                else f"synthesized for the group axis: selected captures form {_group.get('groups')} "
                "compute group(s) with this stage combination on this "
                f"target, and no per-stage member demands a combination"
            ),
            "label": "public",
            # SIGNED. Every stage in a combination treats a negative accumulator differently from a
            # positive one, and on the non-negative default none of that can be observed.
            "stimulus_range": list(_signed_stimulus_range()),
            "generalization": {"generalization_axis": "groups"},
            **extents_for("aligned", probes, quantum=_quanta.get(dtype)),
        }
        declare_pool_window(entry)
        declare_residual_add_params(entry)
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entry["_group_count"] = int(_group.get("groups") or 0)
        _group_members[name] = entry
        entries.append(entry)
    for entry in _group_members.values():
        entry.pop("_group_count", None)

    # ---- the CARRIED-STATE axis ---------------------------------------------------------------------
    # Two commands over one resident weight: the configuration-only stage on the FIRST, nothing on the
    # second. The hand-authored two-command capsule runs them in the other order, on a stimulus that
    # cannot go negative, so it cannot show a stage left switched on: the second command WANTS it.
    for _carried in (spec_doc.get("carried_state") or {}).get("required") or ():
        stage = str(_carried.get("stage") or "")
        dtype = kept[0] if kept else narrowest_admitted(admitted_dtypes)
        if not stage or not dtype:
            unexpressable.append(f"carried-state axis: no admitted dtype to build a {stage!r} program at")
            continue
        extents = extents_for("aligned", probes, quantum=_quanta.get(dtype))
        rows = {key: value for key, value in extents.items() if key.startswith("M")}
        entry = {
            "cat": "layers",
            "kind": "layer",
            "name": f"{SYNTH_PREFIX}_carried_{stage}_{dtype}",
            "op": "resident_reuse",
            "operand_dtype": dtype,
            "weight": "W",
            "matmuls": [
                {"lhs": "A0", "out": "Y0", "epilogue": [stage], **rows},
                {"lhs": "A1", "out": "Y1", "epilogue": [], **rows},
            ],
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for the carried-state axis: {stage!r} is configuration a unit stays in "
                "(evidenced by the derived requirement), so the command after it, which does "
                f"not ask for it, is where a backend that does not restore its configuration shows"
            ),
            "label": "public",
            # SIGNED, or the second result has nothing the leaked stage would change.
            "stimulus_range": list(_signed_stimulus_range()),
            "generalization": {"generalization_axis": "carried_state"},
            **{key: value for key, value in extents.items() if not key.startswith("M")},
        }
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    return composition, unsized_host
