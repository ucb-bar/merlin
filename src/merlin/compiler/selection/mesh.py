"""Existing package, numerical binding and tile-request selection for mesh consumers."""

from __future__ import annotations


def _default_oot_package(target: str) -> str | None:
    """The conventional OOT backend package for ``target``, or None when the target ships no default (a
    bespoke-layout target must name it via ``--package``). Prefers the agent-submission package
    (``.../targets/<target>/agent_spec_v1_mlir_oot`` — the SAME default ``compile_oot`` resolves, so on-mesh
    tile verification reuses it verbatim) and falls back to a hand-curated REFERENCE backend package
    (``.../reference_v0``) when a target ships one instead of an agent submission (e.g. a SIMT core whose
    deterministic reference lowering is the package). Target-agnostic — both are directory conventions, not
    target literals."""
    from merlin.common.artifacts import artifacts_dir

    base = artifacts_dir() / "targets" / target
    # Selected by KIND, not merely by the presence of a manifest. ``out/artifacts/targets/<target>/`` also
    # holds CODEGEN packages (schedules/knobs/dialects, e.g. a hand-curated ``hand_v0``), whose manifest is
    # a different artifact entirely. Returning one of those as an OOT backend does not fail cleanly: it
    # gets as far as the package loader and dies on "manifest schema violation: 'artifact_type' is a
    # required property", which reads like a corrupt backend rather than a directory that was never one.
    for pkg_id in ("agent_spec_v1_mlir_oot", "reference_v0"):
        cand = base / pkg_id
        mf = cand / "manifest.yaml"
        if not mf.is_file():
            continue
        try:
            from merlin.common.yaml import load_yaml

            if not str((load_yaml(mf) or {}).get("artifact_type", "")).strip():
                continue  # not an OOT backend manifest — keep looking
        except Exception:  # noqa: BLE001 — unreadable manifest: let the loader report it
            pass
        return str(cand)
    return None


def _mesh_tile_binding(
    target: str,
    operand_dtype: str | None,
    accum_dtype: str | None,
    requant_output_dtype: str | None = None,
    *,
    numeric_policy: dict | None = None,
):
    """A ``corpus_spec.CorpusBinding`` for synthesizing a single systolic mesh tile of ``target``, with the
    operand/accumulate datapath pinned to what the routed op actually needs (derived, never assumed). The
    tile dim and instruction-class deriver come from the selected target contract. The caller supplies
    the complete numerical policy as declared assumptions, not hardware-verified evidence.

    ``requant_output_dtype`` (the narrow dtype an ``acc_scale`` epilogue requants the i32 accumulator to,
    e.g. i8) is passed only when the caller needs a requant handoff — an int8 matmul chain commits each
    layer's accumulator back to i8 so it can feed the next mesh layer.

    Operation dtype pins overlay that policy without dropping subnormal handling, comparison, scaling,
    or future numerical fields. Corpus-generation controls are excluded; requantization is requested
    by this operation, never inherited from a profile's corpus defaults. No profile path is discovered
    here. A floating datapath without a declared policy refuses explicitly."""
    from types import SimpleNamespace

    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.oracle_policy import selected_sim_via as _bespoke_sim_via

    te = SimpleNamespace(target=target, sim_via=_bespoke_sim_via(target))
    datapath = CS.profile_datapath({"datapath": numeric_policy}, numeric_only=True)
    if operand_dtype:
        datapath["operand_dtype"] = operand_dtype
    if accum_dtype:
        datapath["accum_dtype"] = accum_dtype
    if requant_output_dtype:
        datapath["requant_output_dtype"] = requant_output_dtype
    binding = CS.derive_binding(te, datapath)
    if not binding.integer and numeric_policy is None:
        raise ValueError("floating mesh execution requires an explicit declared numeric_policy")
    return binding


def tile_builder_op(demand) -> tuple[str | None, str]:
    """``(builder name, how it was chosen)`` for a mesh demand, or ``(None, "unsynthesizable")``.

    A tile is synthesized from the demand's EXTENT, never from how the op is spelled -- the certifier
    below reads only ``m/k/n`` and the tile binding. So an op the router placed on the mesh AS a
    contraction, carrying a shape, has a matmul tile whatever it is called.

    Matching on the name alone is the name-vs-family mismatch :attr:`routing.Demand.family` exists for:
    torch-MLIR spells a convolution ``convolution_im2col_matmul`` while a contract spells the unit's ops
    ``[matmul]``. Measured: ``M2_microvit_gemmini`` routed 3 convolutions to the mesh under that
    spelling and ran the whole model bit-exact with 12 of 12 certifiable tiles passing on RTL; the three
    were counted ``n_unsynthesizable`` and failed the capsule at its cert tier, for a builder table the
    submission does not own.

    ``family`` is UNKNOWN-safe and never widens, so an op with no derivable family is still matched by
    name exactly as before, and a family match is recorded so the weaker provenance stays visible.
    """
    from merlin.targetgen import corpus_spec as CS

    name = "matmul" if demand.op in ("matmul", "linear") else demand.op
    if name in CS.BUILDERS:
        return name, "name"
    if demand.family == "contraction" and demand.has_shape:
        return "matmul", "family"
    return None, "unsynthesizable"


for _legacy_function in (
    _default_oot_package,
    _mesh_tile_binding,
    tile_builder_op,
):
    _legacy_function.__module__ = "merlin.compile.mesh"
del _legacy_function
