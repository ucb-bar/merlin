import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects import llvm

from merlin.compiler.dialects.xdsl_context import text
from merlin.compiler.lower.host import borrowed_buffer_calls
from merlin.compiler.lower.host import writers as fresh_tensor_writer
from merlin.compiler.lower.host.borrowed_buffer_calls import (
    CHECKPOINT,
    REPORT,
    RESULTS_SCHEMA,
    from_writers,
    verify_native,
)
from merlin.compiler.lower.host.writers import prove_lowered_writer_results
from merlin.compiler.lower.model import oracle as abi
from merlin.compiler.lower.native import lifetime as dealloc_lifetime
from merlin.compiler.lower.native import pipeline
from merlin.compiler.lower.native import recipe as llvm_dialect_product
from merlin.compiler.lower.native.recipe import verify_llvm_dialect_product
from merlin.verification.host import llvm_symbols as host_call_alignment
from merlin.verification.host.llvm_symbols import import_host_entry
from verification_witnesses.plural_fresh_tensor_writer import contract
from verification_witnesses.plural_fresh_tensor_writer import fixture as tensor_fixture
from verification_witnesses.plural_fresh_tensor_writer import (
    run_plural_results_native_against_separate_calls_and_reference as run_pair,
)


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "CHECKPOINT",
    "Path",
    "REPORT",
    "RESULTS_SCHEMA",
    "abi",
    "borrowed_buffer_calls",
    "contract",
    "ctypes",
    "dealloc_lifetime",
    "fresh_tensor_writer",
    "from_writers",
    "hashlib",
    "host_call_alignment",
    "import_host_entry",
    "json",
    "llvm",
    "llvm_dialect_product",
    "np",
    "pin",
    "pipeline",
    "prove_lowered_writer_results",
    "pytest",
    "run_pair",
    "tensor_fixture",
    "text",
    "verify_llvm_dialect_product",
    "verify_native",
]
