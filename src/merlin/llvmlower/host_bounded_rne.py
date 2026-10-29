"""Selection and original-recipe binding for the existing late host legalizer."""

from __future__ import annotations

import os
from pathlib import Path

PASS_NAME = "host-bounded-rne"
PREPARATION = "host_bounded_rne"


def selected(fallback=False):
    """An explicit optional-pass choice wins over the existing API/default."""
    from .optional_passes import switched

    return switched(PASS_NAME, fallback or os.environ.get("MERLIN_HOST_BOUNDED_RNE", "0") == "1")


def require_report(document, *, require_combined=False):
    """Require the original supported scalar policy, not an arbitrary marker."""
    from merlin.common.digest import sha256_file

    from . import late_quant_rne

    identity = (isinstance(document, dict)
                and document.get("routes") == []
                and document.get("identity") == "no_matching_bounded_conversions"
                and "refusal" not in document
                and isinstance(document.get("source_sha256"), str)
                and len(document["source_sha256"]) == 64
                and document["source_sha256"] == document.get("rewritten_sha256")
                == document.get("native_oracle_sha256"))
    if (not isinstance(document, dict)
            or document.get("schema") != "merlin_late_bounded_rne_v1"
            or document.get("host_isa") not in ("rv64gc", "portable")
            or type(document.get("combine_clamp")) is not bool
            or (document["host_isa"] == "portable" and document["combine_clamp"])
            or (require_combined and (document["host_isa"] != "rv64gc" or document["combine_clamp"] is not True))
            or document.get("legalizer_sha256") != sha256_file(late_quant_rne.__file__)
            or not isinstance(document.get("routes"), list)
            or not document["routes"] and not identity
            or document["routes"] and "identity" in document):
        raise ValueError("original host bounded RNE preparation is absent or changed")


def bind_preparation(compilation, work, observation):
    """Bind the actual existing callback receipt to its exact emitted LLVM."""
    from merlin.common.digest import sha256_file
    from merlin.common.strict_json import loads

    if isinstance(observation, dict) and isinstance(chain := observation.get("chain"), dict):
        from .host_transform_chain import recheck_host_transform_chain

        if sha256_file(chain["path"]) != chain["sha256"]:
            raise ValueError("host bounded RNE transform chain changed")
        terminal = recheck_host_transform_chain(Path(chain["path"])).get("legacy_terminal", {}).get("receipt")
        if isinstance(terminal, dict):
            observation = terminal
            work = Path(terminal["selected_path"]).parent
    report = Path(work) / "receipt.json"
    if not report.is_file():
        if selected():
            raise ValueError("selected host bounded RNE produced no original preparation report")
        return
    document = loads(report.read_bytes())
    if not isinstance(document, dict) or document.get("schema") != "merlin_late_bounded_rne_v1":
        if selected():
            raise ValueError("selected host bounded RNE produced another preparation report")
        return
    require_report(document, require_combined=selected())
    if (not isinstance(observation, dict)
            or document.get("source_sha256") != observation.get("source_sha256")
            or document.get("rewritten_sha256") != observation.get("selected_sha256")):
        raise ValueError("host bounded RNE report differs from actual emitted LLVM")
    compilation.bind_preparation(PREPARATION, report)
