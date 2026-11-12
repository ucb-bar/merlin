"""Compatibility script for existing application ingestion invocations.

Run by an isolated capture interpreter that need not have Merlin installed.
The implementation belongs to the compiler's PyTorch frontend.
"""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent.parent.parent / "compiler/frontend/torch/capture.py"), run_name="__main__")
else:
    from merlin.compiler.frontend.torch.capture import _convert_export, _example_inputs, main  # noqa: F401
