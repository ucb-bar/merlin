"""Disclosed public architecture envelopes to ordinary independent source shapes.

Config fields and explicit operator operating points select development shapes;
no model execution, private graph, captured values or latency labels are read.
Occurrence counts guide shape sampling only, never performance weighting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import development_patterns as P
from . import development_sources as D
from . import shape_spectrum as S


def public_configs(root):
    root = Path(root)
    provenance = json.loads((root / "provenance.json").read_bytes())
    documents = {}
    for row in provenance:
        if set(row) != {"name", "url", "revision", "sha256", "bytes"} or not row["revision"]:
            raise ValueError("public architecture configuration needs exact public source attribution")
        path = root / (row["name"] + ".json")
        raw = path.read_bytes() if not path.is_symlink() else b""
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("public architecture config bytes changed")
        documents[row["name"]] = json.loads(raw)
    if set(documents) != {"decoder", "vision_action", "vision_language"}:
        raise ValueError("select the complete explicit public configuration roster")
    envelope_path = root / "operating-envelope.json"
    envelope_raw = envelope_path.read_bytes() if not envelope_path.is_symlink() else b""
    envelope = json.loads(envelope_raw)
    if (
        envelope.get("scope") != "public_development"
        or envelope.get("schema") != "merlin.phase0.public_architecture_operating_envelope.v1"
    ):
        raise ValueError("architecture operating points require explicit public development disclosure")
    documents["operating_envelope"] = envelope
    provenance.append(
        {
            "name": "operating_envelope",
            "sha256": hashlib.sha256(envelope_raw).hexdigest(),
            "bytes": len(envelope_raw),
            "source_kind": "operator_declared_public_development_envelope",
            "upstream_sources": envelope["sources"],
        }
    )
    return documents, provenance


def declaration(pattern, geometry, *, ordinal):
    return {
        "schema": P.SCHEMA,
        "scope": "public_development",
        "name": f"public_{ordinal:04d}",
        "pattern": pattern,
        "geometry": geometry,
        "arithmetic": {"contract": D.CONTRACT, "readout_shift": 3},
        "max_logical_elements": 2**32,
    }


def derive(config_root, *, axis_granules, resource_sha256, max_spectrum=32):
    documents, provenance = public_configs(config_root)
    decoder = documents["decoder"]
    language, vision = documents["vision_language"]["text_config"], documents["vision_language"]["vision_config"]
    action = documents["vision_action"]
    points = documents["operating_envelope"]["operator_points"]
    prefill, decode, calls, reduced = (
        points[key] for key in ("decoder_prefill", "decoder_decode", "decoder_decode_calls", "reduced_steps")
    )
    rows = []
    # These operating points are the disclosed development task, not imported
    # validation trajectories. Reduced depth retains original logical widths.
    for config, tokens, depth, readout in (
        (decoder, [decode, prefill], decoder["num_hidden_layers"], True),
        (language, [decode, prefill, points["action_chunk"]], action["num_vlm_layers"], False),
    ):
        width, hidden = config["hidden_size"], config["intermediate_size"]
        kv = config["num_key_value_heads"] * (width // config["num_attention_heads"])
        for m in tokens:
            for k, n, occurrence in [(width, width, 2), (width, kv, 2), (width, hidden, 2), (hidden, width, 1)]:
                rows.append(
                    {
                        "family": "projection",
                        "contract": D.CONTRACT,
                        "shape": {"M": m, "K": k, "N": n},
                        "occurrences": depth * occurrence * (calls if m == decode else 1),
                    }
                )
            if readout:
                rows.append(
                    {
                        "family": "readout",
                        "contract": D.CONTRACT,
                        "shape": {"M": m, "K": width, "N": config["vocab_size"]},
                        "occurrences": 1,
                    }
                )
    # A small source-control lower envelope is retained beside architecture
    # widths. It is not represented as an actual model layer or transfer case.
    for shape in documents["operating_envelope"]["kernel_controls"]:
        rows.append({"family": "engineering_control", "contract": D.CONTRACT, "shape": shape, "occurrences": 1})
    summary = {
        "schema": S.SUMMARY_SCHEMA,
        "scope": "public_development",
        "source_sha256": [row["sha256"] for row in provenance],
        "members": rows,
    }
    spectrum = S.plan(
        summary,
        axis_granules=axis_granules,
        resource_sha256=resource_sha256,
        max_members=max_spectrum,
        extra_density_points=min(8, max_spectrum),
    )
    requests = []

    def add(pattern, geometry, basis, **metadata):
        requests.append(
            {"declaration": declaration(pattern, geometry, ordinal=len(requests)), "basis": basis, **metadata}
        )

    for row in spectrum["members"]:
        for layout in ("matmul", "matmul_rhs_nk"):
            add(
                layout,
                row["shape"],
                "public_joint_shape_spectrum",
                spectrum_member=row["id"],
                purposes=row["purposes"],
                occurrence_proxy=row["occurrence_proxy"],
                family=row["family"],
                role="functional_guard" if row["family"] == "engineering_control" else "development_performance",
            )
    for row in spectrum["deferred"]:
        for layout in ("matmul", "matmul_rhs_nk"):
            requests.append(
                dict(
                    basis="public_joint_shape_spectrum",
                    family=row["family"],
                    layout=layout,
                    spectrum_member=row["id"],
                    shape=row["shape"],
                    purposes=row["purposes"],
                    occurrence_proxy=row["occurrence_proxy"],
                    status="source_selection_budget_denied",
                    reason="joint spectrum source-selection ceiling; remains an uncovered required candidate",
                )
            )
    # Disclosed convolution transitions use public source geometry, retaining
    # kernel/stride/padding independently instead of relabeling them as GEMMs.
    conv = documents["operating_envelope"]["convolution_transitions"] + [
        [
            vision["image_size"],
            3,
            vision["hidden_size"],
            language["hidden_size"],
            vision["patch_size"],
            vision["patch_size"],
            0,
        ]
    ]
    for h, ci, co, out, kernel, stride, pad in conv:
        add(
            "conv_transition_nhwc",
            dict(H=h, W=h, CI=ci, CO=co, OUT=out, KH=kernel, KW=kernel, SH=stride, SW=stride, PH=pad, PW=pad),
            "disclosed_public_convolution_transition",
        )
    for h, ci, co, out, kernel, stride, pad in [(5, 3, 5, 7, 3, 1, 1), (9, 3, 7, 11, 7, 2, 3), (7, 5, 3, 9, 1, 1, 0)]:
        add(
            "conv_transition_nhwc",
            dict(H=h, W=h + 2, CI=ci, CO=co, OUT=out, KH=kernel, KW=kernel, SH=stride, SW=stride, PH=pad, PW=pad),
            "independent_small_spatial_tail_control",
        )
    for config, prefix, tokens in (
        (decoder, prefill, decode),
        (decoder, prefill, prefill),
        (language, prefill, decode),
        (language, 1024, points["action_chunk"]),
    ):
        heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
        geometry = dict(
            steps=reduced,
            prefix_tokens=prefix,
            query_tokens=tokens,
            kv_heads=kv,
            query_heads=heads,
            head_dim=config["hidden_size"] // heads,
        )
        row = declaration(D.PATTERN, geometry, ordinal=len(requests))
        # The original explicit grouped-head source has unchanged V1 limits.
        row["schema"] = D.SCHEMA
        row["max_logical_elements"] = 2**30
        requests.append({"declaration": row, "basis": "disclosed_public_grouped_head_layout"})
    for width, prefix, chunk in [
        (7, 5, 3),
        (language["hidden_size"], prefill, points["action_chunk"]),
        (int(language["hidden_size"] * action["expert_width_multiplier"]), 48, points["action_chunk"]),
    ]:
        add(
            "prefix_action_recurrence",
            dict(
                prefix=prefix,
                hidden=width,
                action=action["max_action_dim"],
                chunk=chunk,
                steps=reduced,
                decode=min(chunk, action["n_action_steps"]),
            ),
            "public_prefix_reuse_action_state_envelope",
            omitted_steps=points["denoising_steps"] - reduced,
        )
    small_conv = dict(H=5, W=7, CI=3, CO=5, OUT=7, KH=3, KW=3, SH=1, SW=1, PH=1, PW=1)
    small_action = dict(prefix=35, hidden=7, action=5, chunk=3, steps=2, decode=2)
    add(
        "vision_prefix_action",
        {"vision": small_conv, "action": small_action},
        "independent_connected_vision_prefix_state",
    )
    return {
        "scope": "public_architecture_informed_development",
        "sources": provenance,
        "summary": summary,
        "spectrum": spectrum,
        "requests": requests,
        "missing": [
            "complete_reduced_models",
            "numerical_software_admission",
            "private_exclusion",
            "accelerator_execution",
            "qualified_timing",
        ],
    }
