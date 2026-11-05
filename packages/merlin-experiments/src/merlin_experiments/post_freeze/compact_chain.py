"""Explicit compact all-element chain transport, separate from full raw arrays."""

from pathlib import Path

from merlin.model_check import chain_image_readback as readback


def selected(chain_pin):
    """Reopen the pinned ordinary image, references and actual producer records."""
    image = readback.verify_image(chain_pin)
    if image.get("transport") != "all_element_comparison_v1":
        raise ValueError("compact intake requires an explicitly selected compact guest")
    return image


def collect(model, console: Path):
    chains = [asset for asset in model["assets"] if asset["role"] == "chain_manifest"]
    if len(chains) != 1:
        raise ValueError("compact transport requires one selected complete image manifest")
    image = selected(chains[0])
    if image["output_roster"] != model["output_roster"] or image["chain_steps"] != model["chain_steps"]:
        raise ValueError("compact guest differs from the complete workload roster")
    report = readback.decode(image, console.read_bytes())
    if report["numerical_status"] != "pass" or report["runtime_status"] != "pass":
        raise ValueError("compact guest reported a numerical/runtime mismatch")
    return report
