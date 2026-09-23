"""Every probe the self-check offers is reachable from inside an agent's sandbox."""

from __future__ import annotations

import ast

from phase1_feedback import feedback_source

from merlin.common.paths import merlin_dir, module_source_path

_HARNESS = merlin_dir() / "experiments" / "capsule_bench" / "harness"


def _store_true_flags(path) -> set[str]:
    """Flags a script declares with ``action="store_true"``, read from its syntax tree."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args):
            continue
        first = node.args[0]
        store_true = any(
            k.arg == "action" and isinstance(k.value, ast.Constant) and k.value.value == "store_true"
            for k in node.keywords
        )
        if store_true and isinstance(first, ast.Constant) and str(first.value).startswith("--"):
            found.add(str(first.value))
    return found


def _forwarded() -> dict[str, str]:
    tree = ast.parse((feedback_source("selfcheck_broker")).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "FORWARDED_PROBES":
            return ast.literal_eval(node.value)
    raise AssertionError("the broker declares no FORWARDED_PROBES table")


def test_a_probe_the_selfcheck_offers_is_forwarded_by_the_shim_and_the_broker() -> None:
    # The measured defect: a probe was added to the real self-check and never to the shim, so every
    # sandboxed run -- which is every real run -- got "unrecognized arguments" for it, for weeks.
    probes = {"--shape-coverage", "--offload-census", "--model-layers"}
    real = _store_true_flags(feedback_source("agent_selfcheck"))
    shim = _store_true_flags(module_source_path("merlin_experiments.phase1.tools.selfcheck"))
    assert probes <= real, "this test's probe list has drifted from the self-check"
    assert probes <= shim, f"the sandbox shim does not accept {sorted(probes - shim)}"
    assert set(_forwarded().values()) == probes


def test_the_shim_writes_the_keys_the_broker_reads() -> None:
    shim = module_source_path("merlin_experiments.phase1.tools.selfcheck").read_text(encoding="utf-8")
    for key in _forwarded():
        assert f'"{key}": bool(a.{key})' in shim, f"the shim never writes request key {key!r}"


def test_the_model_layer_probe_honors_the_capsule_names_it_is_given() -> None:
    """A probe that widens a one-capsule request back to the whole corpus wastes the agent's only
    self-check slot, and says nothing about having done it. Held by source, because running the
    probe needs a package and an oracle; its behaviour end to end is exercised by the campaign."""
    import ast

    source = (feedback_source("agent_selfcheck")).read_text(encoding="utf-8")
    tree = ast.parse(source)
    probe = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_model_layers")
    assert "capsules" in {argument.arg for argument in probe.args.args + probe.args.kwonlyargs}
    # The dispatcher hands it the parsed flag rather than dropping it.
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_model_layers"
    )
    passed = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    assert passed.get("capsules") == "a.capsules"
    body = ast.unparse(probe)
    # An unknown name is refused, and what was graded is reported back.
    assert "no model-layer capsule named" in body and "'graded'" in body  # unparse normalizes quotes


def test_the_model_layer_probe_can_report_a_pass_at_all() -> None:
    """A check that cannot pass is as useless as one that cannot fail.

    The probe withholds the deeper adapters on purpose and then counted every tier in the row, so a
    layer that passed every tier that RAN still read as not passed: `all_pass` was unreachable and
    the probe could never say a model layer works, while the operator errata tells the agent a round
    is not converged until one does. Measured 2026-09-17 on a backend that computes two real layers
    exactly: 0 of 2 reported.
    """
    import ast

    source = (feedback_source("agent_selfcheck")).read_text(encoding="utf-8")
    probe = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_model_layers"
    )
    rule = next(node for node in ast.walk(probe) if isinstance(node, ast.FunctionDef) and node.name == "_passed_here")
    # The tiers it withheld are excluded before the row is judged.
    assert "withheld" in ast.unparse(rule)
    namespace: dict = {"withheld": {"L3"}}
    exec(ast.unparse(rule), namespace)  # noqa: S102 -- the rule under test, parsed from the source
    passed = namespace["_passed_here"]
    assert passed({"tiers": {"L0": "pass", "L1": "pass", "L2": "pass", "L3": "unavailable"}})
    assert passed({"tiers": {"L0": "skipped", "L1": "skipped", "L2": "pass", "L3": "unavailable"}})
    # THE MUTATIONS: a tier that ran and failed, a tier we did not withhold that never ran, and a
    # row where nothing ran at all, are each still not a pass.
    assert not passed({"tiers": {"L0": "pass", "L1": "pass", "L2": "fail", "L3": "unavailable"}})
    assert not passed({"tiers": {"L0": "pass", "L1": "unavailable", "L2": "pass"}})
    assert not passed({"tiers": {"L3": "unavailable"}})
