"""The audit path and the mining path must decode a stream the SAME way.

They did not. `asm_audit` grew a full endpoint-decode dispatch and `mining.mine` grew a shorter copy,
and the copy was missing four things: the ISA triple/mattr the endpoint declares, `roles_of`, the
`stream_width` override, and the intrinsics custom table. Every omission produced one symptom - an
expert kernel lifting to None, reported as "decoded nothing of this endpoint". That reads as an empty
expert corpus, when what happened is that the decoder was never handed what it needed.

The cost of that confusion is specific: the mining loop compares an expert CCA against ours, and an
expert side that silently lifts to nothing yields NO divergence, so the loop reports agreement with
an expert it never actually read.
"""
from __future__ import annotations

import collections

import pytest

from merlin.kernels import decode as D
from merlin.kernels import endpoints as _ep


def _roles(decoded) -> collections.Counter:
    return collections.Counter(r for d in decoded for r in (getattr(d, "roles", ()) or ()))


class TestTheDispatcherCoversEveryDeclaredEncodingSource:
    def test_every_source_a_declared_endpoint_uses_is_handled(self):
        """A source the dispatcher does not know returns [], which a caller reads as "nothing here".

        funct_header was exactly this: radiance's MX endpoint declares it, the dispatcher had no
        branch for it, and radiance's expert CCA came back None while its audit read 157 roled
        instructions from the same bytes.
        """
        handled = {"rtl_facts", "isa_encoding", "funct_header", "matrix_units", "mnemonic_grammar"}
        declared = set()
        for name, block in (_ep._spec().get("endpoints") or {}).items():
            src = str(((block or {}).get("encoding") or {}).get("source") or "")
            if src:
                declared.add(src)
        # `isa_model` endpoints are assembled from text, never disassembled from an object.
        unhandled = declared - handled - {"isa_model"}
        assert not unhandled, (
            f"endpoint encoding source(s) {sorted(unhandled)} have no branch in "
            "decode.decode_for_endpoint; a caller will read the empty result as an empty corpus")

    def test_an_unknown_source_yields_empty_not_a_wrong_decode(self):
        """NEGATIVE CASE: refusing is correct; guessing a decoder would invent semantics."""
        class _Fake:
            name = "__no_such_endpoint__"
            engine = "vector"

            @staticmethod
            def roles_of(_):
                return ()

        assert D.decode_for_endpoint([], "gemmini", _Fake()) == []


class TestDisasmSettingsComeFromTheEndpoint:
    @pytest.mark.parametrize("target,expect_triple", [("radiance", "riscv32"), ("gemmini", "riscv64")])
    def test_the_declared_triple_is_used(self, target, expect_triple):
        """A probe that does not pin its ISA settings reports the TOOL's ignorance as the CORPUS's
        nature -- the omission that once put radiance's unknown-word rate at 76% where pinning
        --mattr put it at 15%."""
        eps = [e for e in _ep.endpoints_for(target) if e.roles]
        if not eps:
            pytest.skip(f"{target} declares no roled endpoint")
        assert D.disasm_settings(target, eps[0])["triple"] == expect_triple

    def test_radiance_declares_the_compressed_extension(self):
        eps = [e for e in _ep.endpoints_for("radiance") if e.roles]
        if not eps:
            pytest.skip("radiance declares no roled endpoint")
        mattr = D.disasm_settings("radiance", eps[0])["mattr"] or ""
        assert "c" in mattr, f"compressed must be pinned or half the stream decodes as data: {mattr!r}"
