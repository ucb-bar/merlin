"""The public starter-kit verifier must not encode one accelerator's instruction protocol."""

from merlin.targetgen.oot_starterkit.verify import legal_functs, structural_checks, validate


def test_two_unrelated_public_isa_vocabularies_share_the_same_verifier(tmp_path):
    systolic = tmp_path / "matrix.h"
    simt = tmp_path / "threads.h"
    systolic.write_text("#define MATRIX_PREPARE 0\n#define MATRIX_RUN 4\n")
    simt.write_text("#define THREAD_BARRIER 0x1\n#define THREAD_LAUNCH 0x8\n")

    for header, prefix, names, codes in (
        (systolic, "MATRIX_", ("PREPARE", "RUN"), (0, 4)),
        (simt, "THREAD_", ("BARRIER", "LAUNCH"), (1, 8)),
    ):
        legal = legal_functs(header, define_prefix=prefix)
        assert tuple(legal) == names
        trace = {"instructions": [{"name": names[1], "funct": codes[1]}, {"name": names[0], "funct": codes[0]}]}
        assert structural_checks(trace, legal) == []  # ordering is target-owned
        assert validate(trace=trace, isa_header=header, define_prefix=prefix)["ok"]
        trace["instructions"].append({"name": "OUTSIDE", "funct": 99})
        assert not validate(trace=trace, isa_header=header, define_prefix=prefix)["ok"]

    assert structural_checks({"instructions": "malformed"})
