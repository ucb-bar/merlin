"""Read diagnostic address constants from the runtime's literal C header."""

from pathlib import Path
import re

import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_tooling import sha, sha256

HEADER = checked(Path(__file__).absolute().parents[1] / "runtime/atlas_host_layout.h")
_raw = HEADER.read_bytes()
HEADER_SHA256 = sha(_raw)
CONSTANTS = {}
for _line in _raw.decode("ascii").splitlines():
    _match = re.fullmatch(r"#define (ATLAS_[A-Z0-9_]+) (0x[0-9a-fA-F]+|[0-9]+)", _line)
    if _match:
        _name, _value = _match.groups()
        if _name in CONSTANTS:
            raise ValueError("duplicate host layout constant: " + _name)
        CONSTANTS[_name] = int(_value, 0)
    elif _line.startswith("#define ATLAS_") and _line != "#define ATLAS_HOST_LAYOUT_H":
        raise ValueError("host layout constants must be integer literals")
if not CONSTANTS:
    raise ValueError("empty host layout header")
globals().update(CONSTANTS)


def recheck():
    if sha256(HEADER) != HEADER_SHA256:
        raise ValueError("host layout changed during use")


def fp8_output_offset(m, k, n):
    """Output offset after packed FP8 operands and the 256-byte input guard."""
    return m * k + k * n + 256
