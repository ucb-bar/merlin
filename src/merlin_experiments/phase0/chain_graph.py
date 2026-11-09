"""Producer-to-consumer chains found by structure in a captured program's frontend graph.

A single-operation test case cannot ask a compiler to fuse anything: the work it would fuse is not in
the program. This module finds, in a captured model's ORIGINAL (pre-quantization) frontend graph, the
connected patterns whose fusion or shared loading a compiler could exploit, and describes each by its
geometry only:

* ``conv_requant_pool`` -- a convolution whose result, through an optional per-channel affine map (the
  normalization a quantized model folds into the readout) and an optional unary map, is max-pooled;
* ``contraction_residual`` -- a matrix contraction (a linear layer, or a pointwise stride-1
  convolution, which is the same contraction over the image positions) whose result, through an
  optional per-channel affine map, is one operand of a two-tensor add of identical shape;
* ``shared_input_siblings`` -- two or more such contractions that read the same value;
* ``sibling_feed`` -- two equally wide siblings whose results meet in one binary elementwise node
  whose result feeds a further contraction (a gated feed-forward block);
* ``shared_kv_heads`` -- an attention over more query heads than key/value heads, so each key/value
  head is read by several query heads.

Operations are classified by their semantic family (:func:`merlin.targetgen.semantic_families.from_op`)
and arity, never by a model, module or case name. The reader is value free: it keeps operation names,
tensor shapes and dtypes, integer geometry arguments (stride, padding, kernel) and whether a value is a
parameter. Weights, captured activations, float arguments (scales, epsilons) and module paths are not
read. Nothing here writes a test case or decides a placement.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

FAMILIES = ("conv_requant_pool", "contraction_residual", "shared_input_siblings", "sibling_feed", "shared_kv_heads")
_PARAMETER_KINDS = {"PARAMETER", "BUFFER", "CONSTANT_TENSOR", "CONSTANT"}
#: How far a pattern may pass through single-operand nodes between its producer and consumer.
_MAX_PASSTHROUGH = 6


def _base(target: str) -> str:
    """``aten.conv2d.default`` -> ``conv2d``: the operator's own name, without namespace or overload."""
    parts = str(target).split(".")
    return parts[1] if len(parts) >= 3 else parts[-1]


def _structural(value: Any) -> Any:
    """Keep integer geometry (ints, bools, int lists); a tensor reference or float becomes ``None``."""
    if isinstance(value, bool) or type(value) is int:
        return value
    if isinstance(value, list) and value and all(type(item) is int for item in value):
        return list(value)
    return None


def read(trace: Mapping[str, Any], *, stage: str = "original") -> dict[str, Any]:
    """The value-free structural projection of one stage of a frontend trace."""
    from merlin.targetgen.semantic_families import from_op

    graphs = trace.get("graphs")
    if not isinstance(graphs, Mapping) or stage not in graphs:
        raise ValueError(f"frontend trace has no {stage!r} graph")
    graph = graphs[stage]
    parameters, nodes = set(), {}
    for node in graph["nodes"]:
        if node.get("op") == "placeholder":
            if node.get("input_kind") in _PARAMETER_KINDS:
                parameters.add(node["id"])
            continue
        if node.get("op") != "call_function":
            continue
        base = _base(node["target"])
        nodes[node["id"]] = {
            "id": node["id"],
            "ordinal": node.get("ordinal"),
            "op": base,
            "family": from_op(base),
            "results": [{"shape": list(r.get("shape") or []), "dtype": r.get("dtype")}
                        for r in node.get("results") or () if r.get("kind") == "tensor"],
            "args": [_structural(arg) for arg in node.get("args") or ()],
        }
    inputs, consumers = defaultdict(dict), defaultdict(list)
    for edge in graph.get("edges") or ():
        consumer, producer = edge.get("consumer_node_id"), edge.get("producer_node_id")
        if consumer not in nodes or edge.get("value_kind", "tensor") != "tensor":
            continue
        inputs[consumer][edge["argument_path"]] = {
            "producer": producer, "shape": list(edge.get("shape") or []), "dtype": edge.get("dtype"),
            "parameter": producer in parameters,
        }
        # A node that produces no tensor (a metadata assertion) is not a data consumer.
        if producer in nodes and nodes[consumer]["results"]:
            consumers[producer].append((consumer, edge["argument_path"]))
    return {"stage": stage, "graph_sha256": graph.get("sha256"), "nodes": nodes,
            "inputs": dict(inputs), "consumers": dict(consumers)}


def _shape(node: Mapping[str, Any]) -> list[int] | None:
    results = node["results"]
    return results[0]["shape"] if len(results) == 1 else None


def _activations(graph, node_id) -> list[tuple[str, dict]]:
    """A node's non-parameter tensor operands in argument order."""
    return sorted(((path, edge) for path, edge in graph["inputs"].get(node_id, {}).items() if not edge["parameter"]),
                  key=lambda item: item[0])


def _only_consumer(graph, node_id) -> str | None:
    users = graph["consumers"].get(node_id) or []
    return users[0][0] if len(users) == 1 else None


def _passthrough(graph, node_id) -> str | None:
    """``affine`` or ``unary`` when a node maps one activation elementwise to an equal shape."""
    node = graph["nodes"][node_id]
    acts = _activations(graph, node_id)
    shape = _shape(node)
    if len(acts) != 1 or shape is None or acts[0][1]["shape"] != shape:
        return None
    params = [edge for path, edge in graph["inputs"].get(node_id, {}).items() if edge["parameter"]]
    channels = shape[1] if len(shape) >= 2 else None
    if params and all(len(edge["shape"]) == 1 and edge["shape"][0] in (channels, shape[-1]) for edge in params):
        return "affine"  # per-channel scale/shift: what a quantized readout folds as bias and scale
    if not params and node["family"] == "elementwise_map":
        return "unary"
    return None


def contraction(graph, node_id) -> dict[str, Any] | None:
    """``{"kind": "linear"|"pointwise"|"convolution", M, K, N, ...}`` of a parameter-weighted contraction."""
    node = graph["nodes"][node_id]
    if node["family"] != "contraction":
        return None
    edges = graph["inputs"].get(node_id, {})
    lhs, weight = edges.get("args/0"), edges.get("args/1")
    out = _shape(node)
    if lhs is None or weight is None or lhs["parameter"] or not weight["parameter"] or out is None:
        return None
    if len(weight["shape"]) == 2:
        k, n = weight["shape"][1], weight["shape"][0]
        if lhs["shape"][-1] != k or out[-1] != n:
            return None
        return {"kind": "linear", "M": math.prod(lhs["shape"][:-1]), "K": k, "N": n, "bias": "args/2" in edges}
    if len(weight["shape"]) == 4 and len(lhs["shape"]) == 4 and len(out) == 4:
        cout, cin_group, kh, kw = weight["shape"]
        batch, cin, height, width = lhs["shape"]
        args = node["args"] + [None] * 7
        stride = args[3] if isinstance(args[3], list) else [1, 1]
        padding = args[4] if isinstance(args[4], list) else [0, 0]
        dilation = args[5] if isinstance(args[5], list) else [1, 1]
        groups = args[6] if type(args[6]) is int else 1
        if len(stride) == 1:
            stride = stride * 2
        if len(padding) == 1:
            padding = padding * 2
        if len(dilation) == 1:
            dilation = dilation * 2
        expected = [(extent + 2 * pad - dil * (k - 1) - 1) // step + 1
                    for extent, pad, dil, k, step
                    in zip((height, width), padding, dilation, (kh, kw), stride, strict=True)]
        if groups != 1 or cin_group != cin or out[:2] != [batch, cout] or out[2:] != expected:
            return None  # a grouped or unexplained convolution is not described here
        geometry = {"batch": batch, "cin": cin, "height": height, "width": width, "cout": cout, "kh": kh, "kw": kw,
                    "stride": list(stride), "padding": list(padding), "dilation": list(dilation),
                    "out_height": out[2], "out_width": out[3], "bias": "args/2" in edges}
        pointwise = kh == kw == 1 and stride == [1, 1] and padding == [0, 0]
        return {"kind": "pointwise" if pointwise else "convolution",
                "M": batch * out[2] * out[3], "K": cin * kh * kw, "N": cout, **geometry}
    return None


def _walk(graph, node_id, limit=_MAX_PASSTHROUGH) -> tuple[str, list[str]]:
    """Follow a node's sole consumer through elementwise passthrough nodes; ``(last, kinds)``."""
    kinds, current = [], node_id
    for _ in range(limit):
        nxt = _only_consumer(graph, current)
        if nxt is None:
            break
        kind = _passthrough(graph, nxt)
        if kind is None:
            break
        kinds.append(kind)
        current = nxt
    return current, kinds


def _pool(graph, node_id) -> dict[str, Any] | None:
    node = graph["nodes"][node_id]
    acts = _activations(graph, node_id)
    out = _shape(node)
    if node["family"] != "reduction" or not node["op"].startswith("max_pool") or len(acts) != 1 or out is None:
        return None
    source = acts[0][1]["shape"]
    args = node["args"] + [None] * 6
    kernel = args[1]
    if not isinstance(kernel, list) or len(source) != 4 or len(out) != 4 or out[:2] != source[:2]:
        return None
    kernel = kernel * 2 if len(kernel) == 1 else kernel
    stride = args[2] if isinstance(args[2], list) and args[2] else kernel
    stride = stride * 2 if len(stride) == 1 else stride
    padding = args[3] if isinstance(args[3], list) else [0, 0]
    padding = padding * 2 if len(padding) == 1 else padding
    dilation = args[4] if isinstance(args[4], list) else [1, 1]
    if args[5] is True or dilation not in ([1, 1], [1]):
        return None  # ceil-mode or dilated pooling is outside the declared readout pool
    expected = [(extent + 2 * pad - k) // step + 1
                for extent, pad, k, step in zip(source[2:], padding, kernel, stride, strict=True)]
    if out[2:] != expected:
        return None
    return {"size": list(kernel), "stride": list(stride), "padding": list(padding), "out": out[2:]}


def instances(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every chain instance of one structural graph, each with its family, geometry and node ids."""
    found: list[dict[str, Any]] = []
    nodes = graph["nodes"]
    contractions = {nid: c for nid in nodes if (c := contraction(graph, nid)) is not None}
    # Producer -> (affine/unary) -> max pool.
    for nid, c in contractions.items():
        if c["kind"] not in {"convolution", "pointwise"}:
            continue
        last, kinds = _walk(graph, nid)
        pool_id = _only_consumer(graph, last)
        pool = _pool(graph, pool_id) if pool_id is not None else None
        if pool is not None:
            found.append({"family": "conv_requant_pool", "nodes": [nid, pool_id], "producer": c,
                          "affine": "affine" in kinds, "relu": "unary" in kinds, "pool": pool})
    # Contraction -> (affine) -> add with an independent operand of identical shape.
    for nid, c in contractions.items():
        if c["kind"] not in {"linear", "pointwise"}:
            continue
        last, kinds = _walk(graph, nid)
        if "unary" in kinds:
            continue  # an activation between the contraction and the add is not a residual readout
        add_id = _only_consumer(graph, last)
        if add_id is None or nodes[add_id]["op"] not in {"add", "add_"}:
            continue
        acts = _activations(graph, add_id)
        shape = _shape(nodes[add_id])
        if len(acts) != 2 or any(edge["shape"] != shape for _, edge in acts):
            continue
        after_add, after_kinds = _walk(graph, add_id, limit=1)
        found.append({"family": "contraction_residual", "nodes": [nid, add_id], "producer": c,
                      "affine": "affine" in kinds, "relu_after_add": "unary" in after_kinds})
    # Contractions that read the same value.
    readers = defaultdict(list)
    for nid, c in contractions.items():
        if c["kind"] in {"linear", "pointwise"}:
            readers[graph["inputs"][nid]["args/0"]["producer"]].append(nid)
    for value, group in readers.items():
        if len(group) < 2 or value is None:
            continue
        group = sorted(group, key=lambda n: nodes[n]["ordinal"])
        shapes = [contractions[n] for n in group]
        if len({(c["M"], c["K"]) for c in shapes}) != 1:
            continue
        feed = _feed(graph, group, contractions) if len(group) == 2 and shapes[0]["N"] == shapes[1]["N"] else None
        if feed is not None:
            found.append({"family": "sibling_feed", "nodes": [*group, feed["merge"], feed["down"]],
                          "M": shapes[0]["M"], "K": shapes[0]["K"], "H": shapes[0]["N"],
                          "N": contractions[feed["down"]]["N"], "merge": feed["merge_op"],
                          "gated": feed["gated"]})
        else:
            found.append({"family": "shared_input_siblings", "nodes": group, "M": shapes[0]["M"],
                          "K": shapes[0]["K"], "N": [c["N"] for c in shapes],
                          "kinds": sorted({c["kind"] for c in shapes})})
    found.extend(_attention(graph, contractions))
    return found


def _feed(graph, group, contractions) -> dict[str, Any] | None:
    """The merge node and down contraction when two siblings' results meet and feed a contraction."""
    nodes = graph["nodes"]
    ends = []
    for nid in group:
        last, kinds = _walk(graph, nid)
        merge = _only_consumer(graph, last)
        if merge is None:
            return None
        ends.append((merge, kinds))
    merge = ends[0][0]
    if ends[1][0] != merge or nodes[merge]["family"] != "elementwise_map" or len(_activations(graph, merge)) != 2:
        return None
    last, _ = _walk(graph, merge)
    down = _only_consumer(graph, last)
    if down not in contractions or contractions[down]["kind"] not in {"linear", "pointwise"}:
        return None
    if graph["inputs"][down]["args/0"]["producer"] != last:
        return None
    return {"merge": merge, "down": down, "merge_op": nodes[merge]["op"],
            "gated": any("unary" in kinds for _, kinds in ends)}


def _broadcast_factor(graph, node_id, path, limit=12) -> int | None:
    """How many times an attention operand's elements were replicated before it was read.

    Walk back from the operand through single-activation nodes; each data-movement node
    (:func:`merlin.targetgen.semantic_families.from_op` ``movement``: expand, reshape, transpose,
    repeat) whose result holds more elements than its input replicated them by that integer
    ratio. A padded or otherwise computed node is passed through without counting. Stops at the
    first node with two activations (a rotary embedding, a cache concatenation) or at the
    contraction that produced the operand.
    """
    factor, current = 1, graph["inputs"].get(node_id, {}).get(path)
    for _ in range(limit):
        if current is None or current["producer"] not in graph["nodes"]:
            break
        producer = graph["nodes"][current["producer"]]
        acts = _activations(graph, producer["id"])
        if len(acts) != 1 or producer["family"] == "contraction":
            break  # the projection that produced the operand ends its broadcast history
        before, after = math.prod(acts[0][1]["shape"] or [1]), math.prod(current["shape"] or [1])
        if producer["family"] == "movement" and after > before:
            if after % before:
                return None
            factor *= after // before
        current = acts[0][1]
    return factor


def _attention(graph, contractions) -> list[dict[str, Any]]:
    """Attentions whose key/value heads are fewer than their query heads."""
    nodes, found = graph["nodes"], []
    for nid, node in nodes.items():
        edges = graph["inputs"].get(nid, {})
        if node["family"] == "attention":
            q, k, v = (edges.get(f"args/{i}") for i in range(3))
            if None in (q, k, v) or len(q["shape"]) != 4 or len(k["shape"]) != 4 or len(v["shape"]) != 4:
                continue
            _, query_heads, seq, head_dim = q["shape"]
            kv_len, value_dim = k["shape"][-2], v["shape"][-1]
            share = _broadcast_factor(graph, nid, "args/1")
            kv_heads = query_heads // share if share and query_heads % share == 0 else None
            if kv_heads and share > 1 and _broadcast_factor(graph, nid, "args/2") == share:
                found.append({"family": "shared_kv_heads", "nodes": [nid], "query_heads": query_heads,
                              "kv_heads": kv_heads, "S": seq, "L": kv_len, "d": head_dim, "dv": value_dim})
        elif node["family"] == "contraction" and nid not in contractions:
            lhs, rhs = edges.get("args/0"), edges.get("args/1")
            out = _shape(node)
            if (lhs is None or rhs is None or out is None or len(out) != 4 or lhs["parameter"] or rhs["parameter"]
                    or len(lhs["shape"]) != 4 or len(rhs["shape"]) != 4):
                continue
            # A score contraction: [b, h, S, d] x [b, h, d, L]; its result reaches a context contraction
            # [b, h, S, L] x [b, h, L, dv] through single-activation nodes (scaling, softmax).
            query_heads, seq, head_dim = lhs["shape"][1], lhs["shape"][2], lhs["shape"][3]
            if rhs["shape"][2] != head_dim:
                continue
            context = next((user for user, path in _users_through(graph, nid) if path == "args/0"
                            and nodes[user]["family"] == "contraction" and user not in contractions), None)
            if context is None:
                continue
            share = _broadcast_factor(graph, nid, "args/1")
            if not share or share == 1 or query_heads % share or _broadcast_factor(graph, context, "args/1") != share:
                continue
            kv_heads = query_heads // share
            found.append({"family": "shared_kv_heads", "nodes": [nid, context], "query_heads": query_heads,
                          "kv_heads": kv_heads, "S": seq, "L": out[-1], "d": head_dim,
                          "dv": _shape(nodes[context])[-1]})
    return found


def _users_through(graph, node_id, limit=8) -> list[tuple[str, str]]:
    """Consumers reached from a node along its value, with the consuming argument.

    The walk continues through a consumer whose other activations are smaller broadcasts (an
    attention mask added to scores, a scale), so it follows the value, not every operand.
    """
    out, current = [], node_id
    for _ in range(limit):
        users = graph["consumers"].get(current) or []
        if len(users) != 1:
            return out + list(users)
        user, path = users[0]
        out.append((user, path))
        acts = _activations(graph, user)
        own = math.prod((graph["inputs"][user][path]["shape"]) or [1])
        if any(math.prod(edge["shape"] or [1]) >= own for other, edge in acts if other != path):
            return out
        current = user
    return out


def signature(instance: Mapping[str, Any]) -> tuple:
    """The geometry that identifies an instance's test case; node ids and positions excluded."""
    family = instance["family"]
    if family == "conv_requant_pool":
        p = instance["producer"]
        return (family, p["cin"], p["height"], p["width"], p["cout"], p["kh"], p["kw"], *p["stride"],
                *p["padding"], *p["dilation"], instance["affine"], instance["relu"],
                *instance["pool"]["size"], *instance["pool"]["stride"], *instance["pool"]["padding"])
    if family == "contraction_residual":
        p = instance["producer"]
        return (family, p["kind"], p["M"], p["K"], p["N"], instance["relu_after_add"])
    if family == "shared_input_siblings":
        return (family, instance["M"], instance["K"], *instance["N"])
    if family == "sibling_feed":
        return (family, instance["M"], instance["K"], instance["H"], instance["N"])
    return (family, instance["query_heads"], instance["kv_heads"], instance["S"], instance["L"], instance["d"],
            instance["dv"])


def macs(instance: Mapping[str, Any]) -> int:
    """The instance's contraction multiply-accumulates (pooling and adds excluded)."""
    family = instance["family"]
    if family in {"conv_requant_pool", "contraction_residual"}:
        p = instance["producer"]
        return p["M"] * p["K"] * p["N"]
    if family == "shared_input_siblings":
        return instance["M"] * instance["K"] * sum(instance["N"])
    if family == "sibling_feed":
        return instance["M"] * (2 * instance["K"] * instance["H"] + instance["H"] * instance["N"])
    return instance["query_heads"] * instance["S"] * instance["L"] * (instance["d"] + instance["dv"])


def inventory(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Instances grouped by signature with occurrence counts and MAC weight, largest first."""
    groups: dict[tuple, dict[str, Any]] = {}
    for item in instances(graph):
        key = signature(item)
        row = groups.setdefault(key, {"family": item["family"], "signature": list(key), "instance": item,
                                      "occurrences": 0, "macs": 0, "source_node_ids": []})
        row["occurrences"] += 1
        row["macs"] += macs(item)
        row["source_node_ids"].append(list(item["nodes"]))
    return sorted(groups.values(), key=lambda row: (row["family"], -row["macs"], str(row["signature"])))


def families_present(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [family for family in FAMILIES if any(row["family"] == family for row in rows)]
