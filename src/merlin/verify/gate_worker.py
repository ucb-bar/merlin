"""Run one speculative gate check in this process and write its verdict.

``python -m merlin.verify.gate_worker <job-dir>`` reads ``<job-dir>/request.json``, resolves the
named check (``module:function``, or a built-in name), calls it with the request, and writes
``<job-dir>/verdict.json``. Only a check that returns normally is ``accepted``; every exception
is classified, and an unclassified one is ``unavailable``, never a pass.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import sys
import traceback
from pathlib import Path

BUILTIN_CHECKS = (
    "tensor_refinement",
    "compiler_loop_refinement",
    "compiler_loop_generic_refinement",
    "compiler_buffer_refinement",
    "compiler_memory_dce_refinement",
    "compiler_macc_contract",
    "compiler_qdq_refinement",
    "compiler_configuration_refinement",
    "compiler_configuration_hoist_refinement",
    "compiler_host_graph_fusion",
)


def resolve(name: str):
    if name in BUILTIN_CHECKS:
        from . import gate_checks

        return getattr(gate_checks, name)
    module, _, function = name.partition(":")
    if not module or not function:
        raise LookupError(f"check {name!r} is neither a built-in name nor module:function")
    return getattr(importlib.import_module(module), function)


def classify(error: BaseException) -> str:
    """Map a check's exception onto the four verdict kinds; anything unknown is unavailable."""
    from .gate_errors import GateRejected, GateUnavailable, GateUnsupported

    if isinstance(error, GateUnsupported):
        return "unsupported"
    if isinstance(error, GateRejected):
        return "not_established"
    if isinstance(error, GateUnavailable):
        return "unavailable"
    return "unavailable"


def write_verdict(directory: Path, verdict: dict) -> None:
    temporary = directory / "verdict.json.tmp"
    temporary.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str))
    os.replace(temporary, directory / "verdict.json")


def _stop_group(signum, frame) -> None:
    """A stop request (from the gate, or the parent-death signal when the compile dies) ends the
    whole check: the worker leads its own group, and the checker processes it runs are in it."""
    os.killpg(os.getpgid(0), signal.SIGKILL)


def _require_builtin_result(name, detail):
    if name == "compiler_loop_generic_refinement":
        from .compiler_loop_generic import DOMAIN, RELATION, ROOT, TILE, UNROLL, adjacent
        from .gate_errors import GateRejected

        if (
            type(detail) is not dict
            or detail.get("schema") != "merlin.compiler-generic-loop.gate.v1"
            or detail.get("status") != "checked"
            or detail.get("domain") != DOMAIN
            or detail.get("relation") != RELATION
            or detail.get("kind") not in (UNROLL, TILE)
            or detail.get("compiler_enforcement") is not True
            or detail.get("native_fidelity_proved") is not False
            or type(detail.get("theorems")) is not list
            or ROOT not in detail["theorems"]
            or adjacent(detail["kind"]) not in detail["theorems"]
            or "Merlin.GraphStatePipelineInstance.completeData" not in detail["theorems"]
            or len(set(detail["theorems"])) != len(detail["theorems"])
            or type(detail.get("axioms")) is not list
            or not set(detail["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"}
            or type(detail.get("entries")) is not list
            or not detail["entries"]
            or any(type(value) is not str or not value for value in detail["entries"])
            or len(set(detail["entries"])) != len(detail["entries"])
        ):
            raise GateRejected("generic handler did not establish its complete typed operational entry relation")
        return
    if name == "compiler_host_graph_fusion":
        from .gate_errors import GateRejected
        from .host_graph_fusion_intake import DOMAIN, RELATION, ROOT

        if (
            type(detail) is not dict
            or detail.get("schema") != "merlin.host-graph-fusion.gate.v1"
            or detail.get("status") != "checked"
            or detail.get("domain") != DOMAIN
            or detail.get("relation") != RELATION
            or detail.get("compiler_enforcement") is not True
            or detail.get("native_fidelity") is not False
            or type(detail.get("theorems")) is not list
            or ROOT not in detail["theorems"]
            or type(detail.get("entry")) is not str
            or not detail["entry"]
            or type(detail.get("fused_stages")) is not int
            or not 2 <= detail["fused_stages"] <= 128
        ):
            raise GateRejected("host fusion handler did not establish complete public-result preservation")
        return
    if name == "compiler_memory_dce_refinement":
        from .compiler_memory_dce import DOMAIN, RELATION, ROOTS
        from .gate_errors import GateRejected

        if (
            type(detail) is not dict
            or detail.get("schema") != "merlin.compiler-memory-dce.gate.v1"
            or detail.get("status") != "checked"
            or detail.get("domain") != DOMAIN
            or detail.get("relation") != RELATION
            or detail.get("theorems") != list(ROOTS)
            or detail.get("compiler_enforcement") is not True
            or detail.get("ordinary_same_mode") is not True
            or detail.get("normalization_mode_used") is not False
            or detail.get("full_error_or_world_equivalence") is not False
            or detail.get("native_fidelity_proved") is not False
            or detail.get("allocator_policy_required") is not False
            or detail.get("assurance") != "lean_kernel_std3_no_native_reflection"
            or type(detail.get("axioms")) is not list
            or not set(detail["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"}
            or type(detail.get("entry_points")) is not list
            or not detail["entry_points"]
            or any(type(entry) is not str or not entry for entry in detail["entry_points"])
            or len(set(detail["entry_points"])) != len(detail["entry_points"])
            or type(detail.get("changed_sites")) is not int
            or detail["changed_sites"] < 0
        ):
            raise GateRejected("memory DCE handler did not establish its exact complete graph outcome relation")
        return
    if name == "compiler_configuration_hoist_refinement":
        from .compiler_configuration_hoist_intake import DOMAINS, RELATION, WIDE_DOMAIN
        from .configuration_hoist_emit import ROOT
        from .gate_errors import GateRejected

        if (
            type(detail) is not dict
            or detail.get("schema")
            not in {"merlin.compiler-configuration-hoist.gate.v1", "merlin.compiler-configuration-hoist.gate.v2"}
            or detail.get("status") != "checked"
            or type(detail.get("domain")) is not str
            or detail["domain"] not in DOMAINS
            or detail.get("relation") != RELATION
            or detail.get("representation") != "schedule_kernel"
            or detail.get("compiler_enforcement") is not True
            or detail.get("hardware_model_approved") is not False
            or detail.get("independently_qualified") is not False
            or detail.get("native_fidelity") is not False
            or type(detail.get("theorems")) is not list
            or ROOT not in detail["theorems"]
            or type(detail.get("entries")) is not list
            or not detail["entries"]
            or any(type(entry) is not str or not entry for entry in detail["entries"])
            or len(set(detail["entries"])) != len(detail["entries"])
        ):
            raise GateRejected("hoisting handler did not establish complete Kernel error/state preservation")
        if detail["schema"] == "merlin.compiler-configuration-hoist.gate.v2":
            from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer

            try:
                ConfigurationCConsumer.reopen(detail.get("c_consumer"))
            except (OSError, ValueError, TypeError) as error:
                raise GateRejected("hoisting handler did not bind its actual index64 C consumer") from error
        elif detail["domain"] == WIDE_DOMAIN or "c_consumer" in detail:
            raise GateRejected("wide C-consumer hoisting requires its exact versioned result schema")
        return
    if name == "compiler_configuration_refinement":
        from .compiler_configuration_intake import DOMAIN, RELATION
        from .configuration_elision_emit import FINAL_ROOT
        from .gate_errors import GateRejected

        if (
            type(detail) is not dict
            or detail.get("schema") != "merlin.compiler-configuration.gate.v1"
            or detail.get("status") != "checked"
            or detail.get("domain") != DOMAIN
            or detail.get("relation") != RELATION
            or detail.get("compiler_enforcement") is not True
            or detail.get("hardware_model_approved") is not False
            or detail.get("independently_qualified") is not False
            or detail.get("native_fidelity") is not False
            or detail.get("representation") not in {"instruction_mlir", "schedule_kernel"}
            or type(detail.get("theorems")) is not list
            or FINAL_ROOT not in detail["theorems"]
            or type(detail.get("entries")) is not list
            or not detail["entries"]
            or any(type(entry) is not str or not entry for entry in detail["entries"])
            or len(set(detail["entries"])) != len(detail["entries"])
        ):
            raise GateRejected("configuration handler did not establish complete entry error/state preservation")
        return
    if name == "compiler_qdq_refinement":
        from .gate_checks import THEOREMS
        from .gate_errors import GateRejected

        if (
            type(detail) is not dict
            or detail.get("status") != "checked_proposed_qdq_refinement"
            or detail.get("claim_type") != "proposed_qdq_model_refinement"
            or detail.get("refinement_verdict") != "verified"
            or detail.get("theorems") != list(THEOREMS)
            or type(detail.get("functions")) is not list
            or not detail["functions"]
            or type(detail.get("source_sites")) is not list
            or not detail["source_sites"]
            or detail.get("source_model_status") != "proposed"
            or detail.get("source_spec_approved") is not False
            or type(detail.get("source_correspondence")) is not dict
            or detail["source_correspondence"].get("status") not in {"matched", "not_requested"}
        ):
            raise GateRejected("Q/DQ handler did not establish its complete proposed-model claim")
        source = detail["source_correspondence"]
        if source["status"] == "matched" and (
            source.get("original_program_bound") is not True or not source.get("sites")
        ):
            raise GateRejected("Q/DQ source qualification returned an incomplete binding")
        if source["status"] == "not_requested" and source.get("original_program_bound") is not False:
            raise GateRejected("Q/DQ source qualification was not requested")
        return
    if name == "tensor_refinement":
        from .gate_checks import THEOREMS
        from .gate_errors import GateRejected

        # The optional strengthenings stay optional, but a result may neither
        # drop the per-function chains nor claim definedness it did not verify.
        if (
            type(detail) is not dict
            or type(detail.get("theorems")) not in (list, tuple)
            or tuple(detail["theorems"]) != THEOREMS
            or type(detail.get("functions")) is not list
            or any(type(function) is not str or not function for function in detail["functions"])
            or len(set(detail["functions"])) != len(detail["functions"])
            or type(detail.get("axioms")) is not list
            or not set(detail["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"}
            or type(detail.get("receipt")) is not str
            or not detail["receipt"]
            or type(detail.get("chains")) is not dict
            or set(detail["chains"]) != set(detail["functions"])
            or any(type(path) is not str or not path for path in detail["chains"].values())
            or type(detail.get("strengthenings")) is not dict
            or set(detail["strengthenings"]) != {"universal_boundary", "definedness"}
            or type(detail["strengthenings"]["universal_boundary"]) is not str
            or not detail["strengthenings"]["universal_boundary"]
            or detail["strengthenings"]["definedness"] not in ("verified", "abstracted")
            or type(detail.get("established")) is not list
            or set(detail["established"])
            != {key for key, value in detail["strengthenings"].items() if value == "verified"}
            or len(set(detail["established"])) != len(detail["established"])
            or detail.get("claim")
            != "strict refinement on inputs of the declared types; "
            + (
                "defined on typed, nonpoison inputs (no UB or poison)"
                if "definedness" in detail["established"]
                else "definedness not established"
            )
        ):
            raise GateRejected("tensor handler did not establish complete typed refinement and source support")
        return
    if name == "compiler_loop_refinement":
        from .compiler_loop_intake import TILE, UNROLL
        from .gate_errors import GateRejected
        from .loop_transform_gate import PREMISES, RELATION

        if (
            type(detail) is not dict
            or detail.get("claim") != RELATION
            or detail.get("premises") != PREMISES
            or type(detail.get("receipt")) is not str
            or not detail["receipt"]
            or type(detail.get("axioms")) is not list
            or not set(detail["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"}
            or type(detail.get("boundaries")) is not dict
            or not detail["boundaries"]
            or not set(detail["boundaries"]) <= {UNROLL, TILE}
            or any(
                root != "Merlin.GraphGateProfileAdjacent.complete" + ("2" if kind == UNROLL else "3")
                for kind, root in detail["boundaries"].items()
            )
            or detail.get("native_fidelity") is not False
            or detail.get("general_tiling") is not False
        ):
            raise GateRejected("loop handler did not establish complete modeled return/error/world preservation")
        return
    if name == "compiler_buffer_refinement":
        from .buffer_reuse_intake import RELATION, ROOT
        from .compiler_transforms import ALLOCATOR, BUFFER, TILE
        from .gate_errors import GateRejected

        chain = ("shape_view_normalization", BUFFER, TILE)
        if (
            type(detail) is not dict
            or detail.get("claim") != RELATION
            or detail.get("premises") != {"allocator": ALLOCATOR}
            or detail.get("theorems") != [ROOT]
            or type(detail.get("receipt")) is not str
            or not detail["receipt"]
            or type(detail.get("axioms")) is not list
            or not set(detail["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"}
            or type(detail.get("covered_chain")) is not list
            or not detail["covered_chain"]
            or detail["covered_chain"] != [kind for kind in chain if kind in detail["covered_chain"]]
            or detail.get("compiler_enforcement") is not True
            or detail.get("native_allocator_fidelity") is not False
            or detail.get("full_error_or_world_equivalence") is not False
        ):
            raise GateRejected("buffer handler did not establish complete conditional possible-success preservation")
        return
    if name != "compiler_macc_contract":
        return
    from .compiler_macc import THEOREM
    from .gate_errors import GateRejected

    # The canonical handler must complete its live proof/delivery path. An
    # accidental pending/empty return cannot stand in for that claim. This
    # shape check is not independent proof or numerical approval authority.
    if (
        type(detail) is not dict
        or detail.get("status") != "checked_draft_contract"
        or detail.get("claim_type") != "draft_policy_conformance"
        or detail.get("theorem") != THEOREM
        or detail.get("policy_reviewed") is not False
        or detail.get("numerical_policy_verdict") != "abstracted"
        or detail.get("original_source_equivalence") is not False
        or type(detail.get("sites")) is not list
        or not detail["sites"]
    ):
        raise GateRejected("draft MACC handler did not establish its complete conditional claim")
    seen = set()
    for site in detail["sites"]:
        if (
            type(site) is not dict
            or set(site) != {"function", "source_operation", "source_results", "qualification_passed"}
            or type(site["function"]) is not str
            or not site["function"]
            or type(site["source_operation"]) is not int
            or site["source_operation"] < 0
            or site["qualification_passed"] is not True
            or type(site["source_results"]) is not list
            or not site["source_results"]
            or any(type(value) is not int or value < 0 for value in site["source_results"])
            or len(set(site["source_results"])) != len(site["source_results"])
        ):
            raise GateRejected("draft MACC handler returned a malformed or unqualified site")
        key = (site["function"], site["source_operation"])
        if key in seen:
            raise GateRejected("draft MACC handler returned duplicate site coverage")
        seen.add(key)


def main(argv: list[str]) -> int:
    if os.getpgid(0) == os.getpid():  # only as the leader of its own group (how the gate starts it)
        signal.signal(signal.SIGTERM, _stop_group)
        parent = os.environ.get("MERLIN_GATE_PARENT_PID")
        if parent is not None and str(os.getppid()) != parent:
            # Covers death before setpriv armed the kernel signal. No checker has
            # been started yet; never run one for an already vanished compile.
            _stop_group(signal.SIGTERM, None)
    directory = Path(argv[1])
    request = json.loads((directory / "request.json").read_text())
    try:
        detail = resolve(request["check"])(request)
        _require_builtin_result(request["check"], detail)
        detail = detail or {}
        verdict = {"verdict": "accepted", "reason": "", "detail": detail}
    except BaseException as error:  # noqa: BLE001 -- every outcome must be recorded as a verdict
        verdict = {
            "verdict": classify(error),
            "reason": str(error) or type(error).__name__,
            "detail": {"error_type": type(error).__name__, "traceback": traceback.format_exc()[-4000:]},
        }
    write_verdict(directory, verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
