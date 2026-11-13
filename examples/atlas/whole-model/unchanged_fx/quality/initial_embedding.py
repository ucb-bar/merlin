"""Authenticate a quality input by gathering unchanged source BF16 words."""
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np


def checked(path):
    path = Path(path)
    return path


def file_hash(path):
    digest = sha256()
    with checked(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_info(value):
    return {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": sha256(value.tobytes()).hexdigest()}


def authenticate(initial_inputs, preparation, tokens_path, source_bundle, manifest, first_capsule,
                 selection_manifest=None):
    """Return one verified raw BF16 embedding and its independent provenance.

    Source FX must bind the declared embedding node directly to the declared
    BF16 parameter. A preparation receipt alone cannot authorize arbitrary
    activation bytes. No source graph numerical execution is invoked.
    """
    import torch

    initial_raw = checked(initial_inputs).read_bytes()
    preparation_raw = checked(preparation).read_bytes()
    tokens_raw = checked(tokens_path).read_bytes()
    record = json.loads(preparation_raw)
    expected_graph = manifest["source_model_sha256"]
    if (record.get("schema") != "atlas.unchanged_embedding_input.v1" or record.get("status") != "PREPARED"
            or record.get("source_model_sha256") != expected_graph
            or record.get("source_boundary") != "embedding"
            or record.get("operation") != "raw BF16 word gather by token index"
            or record.get("weight_cast_or_requantization") is not False
            or first_capsule["source_boundaries"][0] != "embedding"
            or first_capsule["input_keys"][0] != "input_0"):
        raise ValueError("initial override is not the declared first source embedding")
    with np.load(io.BytesIO(initial_raw), allow_pickle=False) as archive:
        if archive.files != ["input_0"]:
            raise ValueError("initial override may contain only layer0 input_0")
        activation = archive["input_0"].copy()
    tokens = np.load(io.BytesIO(tokens_raw), allow_pickle=False)
    if (activation.dtype != np.uint16 or activation.ndim != 3
            or tokens.dtype != np.int64 or tokens.ndim != 2 or tokens.size == 0
            or activation.shape[:2] != tokens.shape
            or list(activation.shape) != first_capsule["tensors"]["input_0"]["shape"]
            or first_capsule["tensors"]["input_0"]["dtype"] != "torch.bfloat16"
            or array_info(activation) != record.get("input_0")):
        raise ValueError("initial override raw shape, dtype, or bytes differ from declared embedding")
    tokens_digest = sha256(tokens.tobytes()).hexdigest()
    if (tokens_digest != record.get("tokens_sha256") or tokens_digest != record.get("window", {}).get("input_sha256")
            or record["window"].get("input_length") != tokens.shape[1]):
        raise ValueError("initial tokens differ from the declared evaluation window")
    if selection_manifest is not None:
        selection_raw = checked(selection_manifest).read_bytes()
        selection = json.loads(selection_raw)
        if (sha256(selection_raw).hexdigest() != record.get("selection_manifest_sha256")
                or record["window"] not in selection.get("windows", ())):
            raise ValueError("initial tokens are not bound to the selected quality manifest")
    bundle = checked(source_bundle)
    graph_file = bundle / "model.pt2"
    source_manifest = json.loads((bundle / "fx_manifest.json").read_text())
    if (source_manifest["files"]["model.pt2"]["sha256"] != expected_graph
            or file_hash(graph_file) != expected_graph):
        raise ValueError("initial embedding source graph archive differs from capsule source")
    program = torch.export.load(graph_file)
    nodes = {node.name: node for node in program.graph.nodes}
    embedding = nodes.get("embedding")
    specs = {spec.arg.name: spec for spec in program.graph_signature.input_specs}
    parameter_name = record.get("source_parameter")
    if (embedding is None or embedding.op != "call_function" or str(embedding.target) != "aten.embedding.default"
            or len(embedding.args) < 2 or not isinstance(embedding.args[0], torch.fx.Node)
            or embedding.args[0].op != "placeholder"):
        raise ValueError("source embedding is not a direct declared parameter gather")
    weight_spec = specs.get(embedding.args[0].target)
    if weight_spec is None or weight_spec.kind.name != "PARAMETER" or weight_spec.target != parameter_name:
        raise ValueError("initial receipt names a different source parameter")
    token_node = embedding.args[1]
    token_spec = specs.get(token_node.target) if isinstance(token_node, torch.fx.Node) and token_node.op == "placeholder" else None
    if (token_spec is None or token_spec.kind.name != "USER_INPUT"
            or getattr(token_node.meta.get("val"), "dtype", None) != torch.int64
            or tuple(token_node.meta["val"].shape) != tokens.shape
            or getattr(embedding.meta.get("val"), "dtype", None) != torch.bfloat16
            or tuple(embedding.meta["val"].shape) != activation.shape):
        raise ValueError("initial token IDs must bind directly to the original source embedding input")
    dependency_cache = {}
    def depends_on_tokens(node):
        if node not in dependency_cache:
            dependency_cache[node] = node is token_node or any(depends_on_tokens(parent) for parent in node.all_input_nodes)
        return dependency_cache[node]
    for boundary in first_capsule["source_boundaries"][1:]:
        if boundary not in nodes or depends_on_tokens(nodes[boundary]):
            raise ValueError("unchanged preparation boundary depends on the overridden source token values")
    weight = program.state_dict[parameter_name]
    if weight.dtype != torch.bfloat16 or record.get("source_parameter_dtype") != "bf16" or weight.ndim != 2:
        raise ValueError("initial source embedding table must retain its BF16 dtype")
    words = weight.detach().contiguous().view(torch.uint16).numpy()
    table_digest = sha256(words.tobytes()).hexdigest()
    if (list(words.shape) != record.get("source_parameter_shape") or table_digest != record.get("source_parameter_sha256")
            or tokens.min() < 0 or tokens.max() >= words.shape[0]):
        raise ValueError("initial source table identity or token indices differ")
    gathered = np.ascontiguousarray(words[tokens])
    if (gathered.shape != activation.shape or gathered.tobytes() != activation.tobytes()
            or sha256(words.tobytes()).hexdigest() != table_digest or file_hash(graph_file) != expected_graph):
        raise ValueError("initial bytes are not an unchanged source embedding gather")
    receipt = {"schema": "atlas.independent_embedding_override.v1", "status": "PASS",
               "input": "input_0", "source_boundary": "embedding", "source_model_sha256": expected_graph,
               "preparation_sha256": sha256(preparation_raw).hexdigest(),
               "initial_inputs_file_sha256": sha256(initial_raw).hexdigest(),
               "tokens_file_sha256": sha256(tokens_raw).hexdigest(), "tokens_sha256": tokens_digest,
               "window": record["window"], "source_parameter": parameter_name,
               "source_parameter_sha256": table_digest, "source_parameter_dtype": "bf16",
               "source_parameter_shape": list(words.shape), "input_0": array_info(activation),
               "independent_raw_gather_verified": True, "source_graph_numerical_execution": False,
               "other_preparation_boundaries_token_independent": True,
               "weight_cast_or_requantization": False,
               "selection_manifest_verified": selection_manifest is not None,
               "selection_manifest_sha256": record.get("selection_manifest_sha256"),
               "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    return activation, receipt
