//! General-purpose e-graph execution for Merlin's typed semantic rules.
//! Algorithmic rule generation, extraction and allocation live in Merlin Python.

use egg::{EGraph, Id, Language, Pattern, Rewrite, Runner, StopReason, SymbolLang};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};

#[derive(Deserialize)]
struct Node {
    symbol: String,
    children: Vec<usize>,
}

#[derive(Deserialize)]
struct Rule {
    name: String,
    lhs: String,
    rhs: String,
}

#[derive(Deserialize)]
struct Request {
    schema: String,
    nodes: Vec<Node>,
    roots: Vec<usize>,
    rewrites: Vec<Rule>,
    iterations: usize,
    node_limit: usize,
}

#[derive(Serialize)]
struct ENode {
    symbol: String,
    children: Vec<usize>,
}

#[derive(Serialize)]
struct EClass {
    id: usize,
    nodes: Vec<ENode>,
}

#[derive(Serialize)]
struct Response {
    schema: &'static str,
    roots: Vec<usize>,
    class_by_node: Vec<usize>,
    classes: Vec<EClass>,
    stop_reason: String,
    iterations: usize,
    egraph_nodes: usize,
}

fn run(request: Request) -> Result<Response, String> {
    if request.schema != "merlin.egg_request.v1" {
        return Err("unsupported e-graph request schema".into());
    }
    if request.iterations == 0
        || request.iterations > 64
        || request.node_limit == 0
        || request.node_limit > 100_000
    {
        return Err("invalid e-graph work limits".into());
    }
    let mut graph: EGraph<SymbolLang, ()> = EGraph::default();
    let mut ids: Vec<Id> = Vec::with_capacity(request.nodes.len());
    for node in &request.nodes {
        let children: Vec<Id> = node
            .children
            .iter()
            .map(|&idx| {
                ids.get(idx)
                    .copied()
                    .ok_or_else(|| "node operands must precede their users".to_string())
            })
            .collect::<Result<_, _>>()?;
        ids.push(graph.add(SymbolLang::new(node.symbol.as_str(), children)));
    }
    let mut rules: Vec<Rewrite<SymbolLang, ()>> = Vec::with_capacity(request.rewrites.len());
    for rule in &request.rewrites {
        let lhs: Pattern<SymbolLang> = rule.lhs.parse().map_err(|e| format!("invalid lhs: {e}"))?;
        let rhs: Pattern<SymbolLang> = rule.rhs.parse().map_err(|e| format!("invalid rhs: {e}"))?;
        rules.push(
            Rewrite::new(rule.name.as_str(), lhs, rhs)
                .map_err(|e| format!("invalid rewrite: {e}"))?,
        );
    }
    let runner = Runner::default()
        .with_egraph(graph)
        .with_iter_limit(request.iterations)
        .with_node_limit(request.node_limit)
        .run(&rules);
    let graph = runner.egraph;
    let class_by_node: Vec<usize> = ids.iter().map(|&id| usize::from(graph.find(id))).collect();
    let roots = request
        .roots
        .iter()
        .map(|&idx| {
            class_by_node
                .get(idx)
                .copied()
                .ok_or_else(|| "invalid root index".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut classes: Vec<EClass> = graph
        .classes()
        .map(|class| {
            let mut nodes: Vec<ENode> = class
                .nodes
                .iter()
                .map(|node| ENode {
                    symbol: node.op.to_string(),
                    children: node
                        .children()
                        .iter()
                        .map(|&id| usize::from(graph.find(id)))
                        .collect(),
                })
                .collect();
            nodes.sort_by(|a, b| (&a.symbol, &a.children).cmp(&(&b.symbol, &b.children)));
            EClass {
                id: usize::from(class.id),
                nodes,
            }
        })
        .collect();
    classes.sort_by_key(|class| class.id);
    Ok(Response {
        schema: "merlin.egg_result.v1",
        roots,
        class_by_node,
        classes,
        stop_reason: match runner.stop_reason {
            Some(StopReason::Saturated) => "saturated",
            Some(StopReason::IterationLimit(_)) => "iteration_limit",
            Some(StopReason::NodeLimit(_)) => "node_limit",
            Some(StopReason::TimeLimit(_)) => "time_limit",
            Some(StopReason::Other(_)) => "other",
            None => "unknown",
        }
        .into(),
        iterations: runner.iterations.len(),
        egraph_nodes: graph.total_size(),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let request: Request = serde_json::from_str(&input)?;
    let result = run(request).map_err(io::Error::other)?;
    serde_json::to_writer(io::stdout(), &result)?;
    Ok(())
}
