"""Reviewed neutral integer readout, using the existing byte-addressed MAC engine."""

import hashlib
import json
from dataclasses import replace

import numpy as np
from test_dense_integer_readout import QUALIFICATION
from test_dense_source_provider import DenseProvider

from merlin.common.digest import sha256_file

# Independent C semantics: compare the complete signed i32 accumulator before
# narrowing. No floating-point operation, approximation or per-tile conversion.
READOUT_BODY = "if(value<0)value=0; if(value>127)value=127;"


class ReadoutProvider(DenseProvider):
    def __init__(self, work, model, trace, *, enabled, mutation=None, qualified=True):
        super().__init__(work, model, trace, mutation=mutation)
        self.enabled, self.qualified = enabled, qualified
        self.original_trace = tuple(trace)
        # This reviewed contract precedes source discovery. It qualifies the
        # neutral raw-byte engine's narrow-store API, not physical hardware.
        self.qualification = dict(
            schema="neutral.integer_readout_qualification.v1",
            identity=QUALIFICATION,
            implementation=READOUT_BODY,
            implementation_sha256=hashlib.sha256(READOUT_BODY.encode()).hexdigest(),
            facts_sha256=hashlib.sha256(self.selected.facts_bytes).hexdigest(),
            software_sha256=hashlib.sha256(self.selected.software_bytes).hexdigest(),
            accumulator="signed i32, complete source-order modular reduction",
            output="i8; signed clamp[0,127] before exact truncation",
            envelope="all signed-i32 values and legal selected tile/address/tail operands",
            effects="read accumulator; fully write logical output; synchronous; no retained aliases/releases",
            scope="reviewed neutral C template only; no target support or timing qualification",
        )
        self.qualification_path = work / "original_readout_qualification.json"
        self.qualification_path.write_text(json.dumps(self.qualification, indent=2))
        self.qualification_sha256 = sha256_file(self.qualification_path)

    def selection(self):
        from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects

        return replace(
            super().selection(),
            fuse_output_readout=self.enabled,
            readout_qualification_identity=QUALIFICATION if self.qualified else None,
            borrowed_effects=BorrowedBufferEffects("neutral-reviewed-integer-readout-strided-ABI-v1", True, True, True),
        )

    def prepare(self, bindings, private_directory):
        assert sha256_file(self.qualification_path) == self.qualification_sha256
        assert json.loads(self.qualification_path.read_text()) == self.qualification
        widths = [b.compilation.compiled.kernel.args[-1].dtype for b in bindings]
        self.trace = [
            (index, left, right, np.clip(raw, 0, 127).astype("i1") if widths[index] == "i8" else raw)
            for index, left, right, raw in self.original_trace
        ]
        super().prepare(bindings, private_directory)

    def implementation_options(self, result):
        if result.compiled.kernel.args[-1].dtype == "i32":
            return {}
        assert result.output_readout.instruction_name == "store_i8"
        body = READOUT_BODY
        if self.mutation == "readout_order":
            body = "value=(int8_t)value;" + body
        elif self.mutation == "readout_bound":
            body = body.replace("127", "126")
        return dict(readout_body=body)

    @staticmethod
    def wrapper(index, symbol, bridge, problem, *, output_width=4):
        return DenseProvider.wrapper(index, symbol, bridge, problem, output_width=output_width, typed_scratch=True)
