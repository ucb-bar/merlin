"""Real current SDK source-release and bundle producers for custody tests."""

import hashlib
import json

from merlin.common.paths import repo_root
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.tool_profile import CURRENT_COMMON_SUBSTRATE, attach_common_substrate
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release


def prepare(root, *, tools=()):
    prepare_shared_source_release(
        root, python_source_root=repo_root() / "src", license_directory=repo_root(), version=27, tools=tools
    )
    return root


def bundle(root, *, arm="raw_baseline", host_index_bits=64, **options):
    selected = dict(arm=arm, tools=list(TR.ARM_TOOLS[arm]), allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(
        selected, version=CURRENT_COMMON_SUBSTRATE, python_source_root=root, host_index_bits=host_index_bits, **options
    )
    for row in selected["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(root / row["path"].removeprefix("merlin/python/"))
    return selected


def reseal(selected):
    profile = selected["compiler_tool_profile"]
    profile["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in profile.items() if k != "sha256"}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
