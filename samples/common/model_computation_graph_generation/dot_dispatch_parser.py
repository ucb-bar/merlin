#!/usr/bin/env python3
"""
Post-process IREE dispatch graph (.dot) to extract dependencies between async dispatches.
Uses pydot to read the dot file and keeps only nodes that correspond to @...$async_dispatch_<n>.
Outputs a JSON summary and prints a readable dependency list.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import networkx as nx
import pydot


def load_dot_graph(dot_path: Path):
    graphs = pydot.graph_from_dot_file(str(dot_path))
    if not graphs:
        raise RuntimeError(f"No graphs found in {dot_path}")
    return graphs[0]


def _collect_nodes_recursive(graph) -> List[pydot.Node]:
    """Collect all nodes from graph and nested subgraphs."""
    nodes = list(graph.get_nodes())
    for sub in graph.get_subgraphs():
        nodes.extend(_collect_nodes_recursive(sub))
    return nodes


def build_dispatch_mapping(graph) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Return mapping from node name (v42) to unique dispatch key, including subgraphs.
    If multiple nodes share the same async_dispatch_<id>, they receive suffixes _1, _2, ...
    Example: dispatch_13_1, dispatch_13_2.
    """
    matches: List[Tuple[str, int]] = []
    pattern = re.compile(r"@[^@]*\$async_dispatch_(\d+)")

    for node in _collect_nodes_recursive(graph):
        name = (node.get_name() or "").strip('"')
        label = node.get_label() or ""
        label = label.strip('"')
        m = pattern.search(label)
        if m:
            dispatch_id = int(m.group(1))
            matches.append((name, dispatch_id))

    # Count occurrences per dispatch id
    total_per_id: Dict[int, int] = {}
    for _, did in matches:
        total_per_id[did] = total_per_id.get(did, 0) + 1

    node_to_key: Dict[str, str] = {}
    key_meta: Dict[str, Any] = {}
    current_count: Dict[int, int] = {}

    for name, did in matches:
        current_count[did] = current_count.get(did, 0) + 1
        ordinal = current_count[did]
        total = total_per_id[did]
        if total > 1:
            key = f"dispatch_{did}_{ordinal}"
        else:
            key = f"dispatch_{did}"
        node_to_key[name] = key
        key_meta[key] = {"id": did, "ordinal": ordinal, "total": total}

    return node_to_key, key_meta


def extract_dependencies(graph, node_to_key: Dict[str, str]):
    """Return adjacency list of dispatch dependencies (including paths via non-dispatch nodes)."""
    deps: Dict[str, Set[str]] = {}

    def _edges_recursive(g) -> List[pydot.Edge]:
        edges = list(g.get_edges())
        for sub in g.get_subgraphs():
            edges.extend(_edges_recursive(sub))
        return edges

    # Build adjacency over all nodes (dispatch and non-dispatch)
    adj: Dict[str, Set[str]] = {}
    for edge in _edges_recursive(graph):
        src = (edge.get_source() or "").strip('"')
        dst = (edge.get_destination() or "").strip('"')
        if not src or not dst:
            continue
        adj.setdefault(src, set()).add(dst)

    dispatch_nodes = set(node_to_key.keys())

    # For each dispatch, traverse forward until hitting downstream dispatches
    for src_node, src_key in node_to_key.items():
        deps.setdefault(src_key, set())  # ensure presence
        visited: Set[str] = set()
        queue: List[str] = list(adj.get(src_node, []))
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            if cur in dispatch_nodes:
                dst_key = node_to_key[cur]
                deps.setdefault(dst_key, set()).add(src_key)
                deps.setdefault(src_key, set())
                # Do not traverse past another dispatch to avoid skipping intermediate dispatch boundaries
                continue
            # non-dispatch: keep traversing
            queue.extend(adj.get(cur, []))

    return deps


def to_json(dot_path: Path, deps: Dict[str, Set[str]], key_meta: Dict[str, Any]):
    return {
        "dot_file": str(dot_path),
        "dispatches": {
            k: {
                "id": key_meta.get(k, {}).get("id"),
                "ordinal": key_meta.get(k, {}).get("ordinal"),
                "total": key_meta.get(k, {}).get("total"),
                "dependencies": sorted(list(parents)),
            }
            for k, parents in sorted(deps.items())
        },
        "edges": [
            {"from": src, "to": dst}
            for dst, parents in deps.items()
            for src in parents
        ],
    }


def _sort_key(keys: List[str]) -> List[str]:
    def keyer(k: str):
        m = re.match(r"dispatch_(\d+)(?:_(\d+))?", k)
        if m:
            base = int(m.group(1))
            ordv = int(m.group(2)) if m.group(2) else 0
            return (base, ordv, k)
        return (1e9, 1e9, k)
    return sorted(keys, key=keyer)


def print_summary(deps: Dict[str, Set[str]]):
    print("Dispatch dependency summary (filtered to async_dispatch nodes):")
    for key in _sort_key(list(deps.keys())):
        parents = _sort_key(list(deps[key]))
        parents_str = ", ".join(parents) if parents else "None"
        print(f"  {key} <- [{parents_str}]")


def render_png(deps: Dict[str, Set[str]], png_path: Path):
    """Render a DAG PNG from dependency dict."""
    G = nx.DiGraph()
    for dst, parents in deps.items():
        G.add_node(dst)
        for src in parents:
            G.add_edge(src, dst)

    # simple layered layout using topological generations
    layers: List[List[int]] = []
    try:
        layers = list(nx.topological_generations(G))
    except nx.NetworkXUnfeasible:
        # fallback: all nodes in one layer
        layers = [list(G.nodes())]

    pos: Dict[str, Tuple[float, float]] = {}
    y_gap = 2.0
    for yi, layer in enumerate(layers):
        x_gap = 2.5
        start_x = -0.5 * x_gap * (len(layer) - 1) if layer else 0
        for xi, node in enumerate(layer):
            pos[node] = (start_x + xi * x_gap, -yi * y_gap)

    # ensure all nodes have positions
    for n in G.nodes():
        if n not in pos:
            pos[n] = (0, -len(layers) * y_gap - 2)

    plt_fig = None
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt_fig = plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels={n: n for n in G.nodes()},
            node_size=2000,
            node_color="lightblue",
            arrows=True,
            arrowsize=20,
            arrowstyle="->",
            font_size=10,
            edge_color="gray",
        )
        plt.title("Dispatch Dependency DAG (from .dot)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
        print(f"PNG written to {png_path}")
    except Exception as exc:  # pragma: no cover
        if plt_fig:
            try:
                import matplotlib.pyplot as plt
                plt.close()
            except Exception:
                pass
        print(f"Warning: failed to render PNG: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Extract dispatch dependencies from an IREE dispatch .dot graph")
    parser.add_argument("dot_file", type=Path, help="Path to dispatch graph .dot file")
    parser.add_argument("--json-out", type=Path, help="Optional path to write JSON summary")
    parser.add_argument("--png", type=Path, help="Optional path to write DAG PNG rendered from deps")
    args = parser.parse_args()

    graph = load_dot_graph(args.dot_file)
    node_to_dispatch, key_meta = build_dispatch_mapping(graph)
    if not node_to_dispatch:
        # Debug help: list a few node labels
        sample_labels = []
        for node in graph.get_nodes()[:5]:
            label = (node.get_label() or "").strip('"')
            name = (node.get_name() or "").strip('"')
            sample_labels.append({"name": name, "label": label})
        raise RuntimeError(f"No async_dispatch nodes found in the dot file. Sample labels: {sample_labels}")

    deps = extract_dependencies(graph, node_to_dispatch)
    print_summary(deps)

    if args.json_out:
        data = to_json(args.dot_file, deps, key_meta)
        with open(args.json_out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nJSON written to {args.json_out}")

    if args.png:
        render_png(deps, args.png)


if __name__ == "__main__":
    main()
