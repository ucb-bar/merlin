"""Actual ELF alias definitions; synthetic listings grant no runtime authority."""

import shutil
import subprocess

import pytest
from merlin_experiments.phase1 import annotated_spike_evidence as E
from merlin_experiments.phase2 import _counter_symbol_aliases as A


def rows(entry="entry_alias"):
    return E.parse_disassembly(f'''00000100 <main>:
      100: csrr s0,mcycle
      104: jalr ra # 400 <{entry}>
      108: jalr ra # 500 <complete>
      10c: csrr a0,mcycle
    ''')


def actual_elf(tmp_path, *, other=False):
    cc = shutil.which("cc")
    assert cc, "independent static alias check requires the existing host compiler"
    source = tmp_path / "aliases.c"
    source.write_text('void entry(void){}\nvoid other(void){}\nvoid complete(void){}\n'
                      f'extern __typeof(entry) entry_alias __attribute__((alias("{"other" if other else "entry"}")));\n'
                      'int main(void){return 0;}\n')
    elf = tmp_path / "aliases.elf"
    subprocess.run([cc, "-O0", "-no-pie", str(source), "-o", str(elf)], check=True, capture_output=True)
    return elf.read_bytes()


def review(rows, raw):
    return A.window(rows, elf_bytes=raw, counter_asm="csrr %0, mcycle", entry_symbol="entry",
                    completion_symbol="complete")


def test_original_exact_symbol_review_stays_v1_without_an_alias_interpretation():
    window, schema = review(rows("entry"), b"not an ELF; original reader requires no alias bridge")
    assert schema == E.REVIEW_SCHEMA and "symbol_aliases" not in window
    assert window["begin_pc"] == 0x100 and window["end_pc"] == 0x10C


def test_exact_positive_function_extent_recovers_only_the_selected_symbol_spelling(tmp_path):
    raw = actual_elf(tmp_path)
    with pytest.raises(E.EvidenceError, match="expected one function"):
        E.counter_window(rows(), counter_asm="csrr %0, mcycle", entry_symbol="entry", completion_symbol="complete")
    window, schema = review(rows(), raw)
    assert schema == A.REVIEW_SCHEMA
    assert window["begin_pc"] == 0x100 and window["end_pc"] == 0x10C
    assert window["symbol_aliases"][0]["selected_symbol"] == "entry"
    assert window["symbol_aliases"][0]["alias"] == "entry_alias"
    assert window["symbol_aliases"][0]["size_bytes"] > 0
    assert "no counter, completion or timing qualification" in window["alias_authority"]


def test_same_function_bytes_at_another_address_cannot_supply_an_alias(tmp_path):
    with pytest.raises(E.EvidenceError):
        review(rows(), actual_elf(tmp_path, other=True))


@pytest.mark.parametrize("damage", ["counter_order", "two_callers", "absent", "too_many"])
def test_alias_bridge_preserves_all_original_scope_and_bounded_intake_refusals(tmp_path, damage):
    raw, listing = actual_elf(tmp_path), rows()
    if damage == "counter_order":
        listing[0]["pc"] = 0x106
    elif damage == "two_callers":
        listing += [{**row, "function": "second"} for row in listing]
    elif damage == "absent":
        listing[1]["symbols"] = ["unrelated"]
    else:
        listing[1]["symbols"] += ["unrelated_" + str(i) for i in range(129)]
    with pytest.raises(E.EvidenceError):
        review(listing, raw)


def test_missing_positive_extent_cannot_be_accepted_by_zero_size_or_name(monkeypatch):
    monkeypatch.setattr(A.SYMBOLS, "linked_entry_extent", lambda _raw, name: dict(
        symbol=name, address=0x400, size_bytes=0, section=".text"))
    with pytest.raises(E.EvidenceError):
        review(rows(), b"bounded fixture; decoder substituted only for this refusal check")
