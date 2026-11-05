"""Additive public projection envelope; never rewrite the frozen source cohort."""

from __future__ import annotations

import ast
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from . import development_envelopes as E


def _public_sources(root):
    provenance = json.loads((root / "defaults-provenance.json").read_bytes())
    trees = {}
    for row in provenance:
        path = root / row["path"]
        raw = path.read_bytes() if not path.is_symlink() else b""
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("selected public default implementation bytes changed")
        trees[path.name.removesuffix(".source")] = ast.parse(raw)
    return trees, provenance


def _class(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _init(cls):
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")


def _linear_pairs(cls):
    return {
        ast.unparse(node.targets[0]): [ast.unparse(arg) for arg in node.value.args[:2]]
        for node in ast.walk(_init(cls))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "nn.Linear"
    }


def derive(config_root, *, base_manifest_sha256):
    root = Path(config_root)
    configs, config_sources = E.public_configs(root)
    trees, default_sources = _public_sources(root)
    vision = configs["vision_language"]["vision_config"]
    language, action = configs["vision_language"]["text_config"], configs["vision_action"]
    points = configs["operating_envelope"]["operator_points"]
    cfg = trees["configuration_smolvlm.py"]
    constructor = _init(_class(cfg, "SmolVLMVisionConfig"))
    defaults = dict(
        zip(
            (arg.arg for arg in constructor.args.args[-len(constructor.args.defaults) :]),
            (ast.literal_eval(node) for node in constructor.args.defaults),
            strict=True,
        )
    )
    route = _init(_class(cfg, "SmolVLMConfig"))
    if not any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "SmolVLMVisionConfig"
        and any(
            keyword.arg is None and isinstance(keyword.value, ast.Name) and keyword.value.id == "vision_config"
            for keyword in node.keywords
        )
        for node in ast.walk(route)
    ):
        raise ValueError("selected public configuration no longer proves the vision dictionary constructor route")
    effective = {}
    for field in ("intermediate_size", "num_hidden_layers"):
        if not any(
            isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "self." + field
            and isinstance(node.value, ast.Name)
            and node.value.id == field
            for node in ast.walk(constructor)
        ):
            effective[field] = None
        else:
            effective[field] = vision.get(field, defaults.get(field))
        if type(effective[field]) is not int or effective[field] < 1:
            effective[field] = None
    model = trees["modeling_smolvlm.py"]
    attention = _linear_pairs(_class(model, "SmolVLMVisionAttention"))
    if any(
        attention.get("self." + name) != ["self.embed_dim", "self.embed_dim"]
        for name in ("q_proj", "k_proj", "v_proj", "out_proj")
    ):
        raise ValueError("selected public source no longer proves four equal-width vision projections")
    mlp = _linear_pairs(_class(model, "SmolVLMVisionMLP"))
    if mlp != {
        "self.fc1": ["config.hidden_size", "config.intermediate_size"],
        "self.fc2": ["config.intermediate_size", "config.hidden_size"],
    }:
        effective["intermediate_size"] = None
    connector = _init(_class(model, "SmolVLMSimpleMLP"))
    assignments = {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in ast.walk(connector)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
    }
    if (
        assignments.get("input_size") != "config.vision_config.hidden_size * config.scale_factor ** 2"
        or assignments.get("output_size") != "config.text_config.hidden_size"
        or _linear_pairs(_class(model, "SmolVLMSimpleMLP")).get("self.proj") != ["input_size", "output_size"]
    ):
        raise ValueError("selected public connector width equations are unsupported")
    shuffle = next(
        node
        for node in _class(model, "SmolVLMConnector").body
        if isinstance(node, ast.FunctionDef) and node.name == "pixel_shuffle"
    )
    if not any(
        isinstance(node, ast.Call)
        and ast.unparse(node) == "x.reshape(bsz, int(seq / scale_factor ** 2), embed_dim * scale_factor ** 2)"
        for node in ast.walk(shuffle)
    ):
        raise ValueError("selected public connector token equation is unsupported")
    width, hidden = vision["hidden_size"], language["hidden_size"]
    factor = configs["vision_language"]["scale_factor"]
    tokens = points["camera_count"] * (points["camera_pixels"] // vision["patch_size"]) ** 2
    image_tokens = tokens // factor**2
    text_tokens = action["tokenizer_max_length"]
    expert_fraction = Fraction(str(action["expert_width_multiplier"])) * hidden
    if expert_fraction.denominator != 1:
        raise ValueError("public expert width multiplier does not prove an exact integral width")
    expert = expert_fraction.numerator
    requests, gaps = [], []

    def add(m, k, n, purpose, *, invocations, layers, multiplicity=1, upper=False, nk=False):
        metadata = dict(
            purpose=purpose,
            invocations=invocations,
            layer_count=layers,
            structural_multiplicity=multiplicity,
            occurrence_proxy=(None if upper or layers is None else invocations * layers * multiplicity),
            occurrence_scope="public architecture sampling proxy; not latency or objective weight",
            role="upper_geometry_control" if upper else "development_performance",
        )
        for pattern in ["matmul", "matmul_rhs_nk"] if nk else ["matmul"]:
            declaration = E.declaration(pattern, dict(M=m, K=k, N=n), ordinal=len(requests))
            declaration["name"] = f"shape_additive_{len(requests):02d}"
            requests.append(
                {"declaration": declaration, "basis": "additive_public_implementation_shape_audit", **metadata}
            )

    depth, intermediate = effective["num_hidden_layers"], effective["intermediate_size"]
    add(tokens, width, width, "vision_Q_K_V_output", invocations=1, layers=depth, multiplicity=4)
    if type(intermediate) is int and intermediate > 0:
        add(tokens, width, intermediate, "vision_MLP_up", invocations=1, layers=depth, nk=True)
        add(tokens, intermediate, width, "vision_MLP_down", invocations=1, layers=depth)
        add((5 * tokens + 3) // 4, width, intermediate, "vision_above_upper_M", invocations=1, layers=depth, upper=True)
    else:
        gaps.append(
            dict(member="vision_MLP_up_down", status="UNKNOWN", reason="effective intermediate default unavailable")
        )
    add(image_tokens, width * factor**2, hidden, "pixel_shuffle_connector", invocations=1, layers=1, nk=True)
    prefix = image_tokens + text_tokens
    language_depth = action["num_vlm_layers"]
    kv_width = language["num_key_value_heads"] * (hidden // language["num_attention_heads"])
    for k, n, count, purpose in [
        (hidden, hidden, 2, "Q_output"),
        (hidden, kv_width, 2, "K_V"),
        (hidden, language["intermediate_size"], 2, "MLP_up_gate"),
        (language["intermediate_size"], hidden, 1, "MLP_down"),
    ]:
        add(
            prefix,
            k,
            n,
            "language_postconnector_image_plus_text_" + purpose,
            invocations=1,
            layers=language_depth,
            multiplicity=count,
        )
    add(
        tokens,
        hidden,
        language["intermediate_size"],
        "language_preconnector_upper_control",
        invocations=1,
        layers=language_depth,
        upper=True,
    )
    add(
        tokens + text_tokens,
        hidden,
        hidden,
        "language_preconnector_plus_text_upper_control",
        invocations=1,
        layers=language_depth,
        upper=True,
    )
    add(
        points["action_chunk"],
        expert,
        expert,
        "expert_width_envelope",
        invocations=points["denoising_steps"],
        layers=None,
        nk=True,
    )
    return {
        "schema": "merlin.phase0.additive_public_shape_supplement.v1",
        "base_manifest_sha256": base_manifest_sha256,
        "original_members_removed": [],
        "sources": config_sources + default_sources,
        "requests": requests,
        "effective_vision_defaults": effective,
        "default_scope": "selected Transformers public implementation; not proof about every historical version",
        "geometry": dict(
            vision_tokens=tokens,
            post_shuffle_image_tokens=image_tokens,
            connector_reduction_width=width * factor**2,
            image_plus_text_prefix=prefix,
            expert_width=expert,
        ),
        "invocation_audit": dict(
            decoder_prefill=1,
            decoder_decode=points["decoder_decode_calls"],
            reused_vision_prefix=1,
            reused_language_prefix=1,
            denoising=points["denoising_steps"],
            original_language_M1_times32="engineering sampling proxy; not SmolVLA invocation census",
        ),
        "gaps": gaps
        + [
            dict(member="extra_state_and_special_prefix_tokens", status="UNKNOWN"),
            dict(
                member="expert_depth_and_intermediate_size",
                status="UNKNOWN",
                reason="num_expert_layers=0 interpretation requires its public expert implementation",
            ),
            dict(
                member="complete_reduced_models",
                status="UNKNOWN",
                reason="projection sources omit residual, bias, activation and normalization context",
            ),
        ],
        "phase0_admitted": False,
        "performance_objective_changed": False,
    }
