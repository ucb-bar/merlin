"""Muon-owned pre-spend smoke of the production fork-free compiler pipeline.

This is a narrow executable smoke, not compiler certification or RTL qualification.
True means it ran and passed; False means failure; None means it did not run.
"""

from __future__ import annotations

import tempfile

from merlin.targetgen.isa_model import isa_model_for_target
from merlin.targetgen.oracle_policy import selected_sim_via

from . import muon


def preflight_codegen_smoke(*, target: str) -> tuple[bool | None, str]:
    """Check the compile prerequisite before any inapplicable-smoke shortcut."""
    if selected_sim_via(target) == "cyclotron":
        try:
            muon._model_for(target)
        except Exception as e:  # noqa: BLE001 — missing prerequisite is NO_GO, never n/a
            return False, (
                f"the fork-free emit path cannot build its ISA model for {target!r}: "
                f"{str(e)[-160:]}. Every capsule would fail to compile and the run would "
                f"grade only capsules needing no oracle. Derive the encoding fact / set "
                f"MERLIN_MLC_DIR before spending."
            )
    try:
        if not isa_model_for_target(target).is_fixed_format():
            return None, "n/a (ISA is not fixed-format — no fork-free re-encode smoke for this emit path)"
    except Exception as e:  # noqa: BLE001 — no derived model means nothing to smoke
        return None, f"n/a (no fixed-format ISA model: {str(e)[-120:]})"
    if selected_sim_via(target) != "cyclotron":
        return None, "n/a (fixed-format ISA but no cyclotron reference sim declared for the fork-free smoke)"
    if not muon.available("cyclotron"):
        return None, "n/a (reference sim absent — oracle_available reports this separately)"
    # Retained target-owned smoke ABI: one thread, inline MMIO, independent inputs.
    kernel = (
        "#include <stdint.h>\n"
        'static inline uint32_t hid(void){uint32_t r;__asm__ volatile("csrr %0,0xF14":"=r"(r));return r;}\n'
        "static inline void pc(char c){*(volatile char*)0xFF080000u=c;}\n"
        "static inline void ph(uint32_t v){for(int i=7;i>=0;--i){uint32_t n=(v>>(i*4))&0xF;"
        "pc(n<10?(char)('0'+n):(char)('a'+n-10));}}\n"
        "int main(void){volatile uint32_t A[8],B[8],C[8];"
        "for(int i=0;i<8;i++){A[i]=(uint32_t)(i+1);B[i]=(uint32_t)(10*(i+1));}"
        "for(int i=0;i<8;i++)C[i]=A[i]+B[i];"
        "if(hid()==0){for(int i=0;i<8;i++){ph(C[i]);pc('\\n');}}return 0;}\n"
    )
    with tempfile.TemporaryDirectory() as td:
        try:
            elf = muon.compile_kernel_forkfree(kernel, td, target=target)
            console, _cyc, _ = muon.run_elf(str(elf), simulator="cyclotron", timeout=180)
        except Exception as e:  # noqa: BLE001 — broken emit path must fail this gate
            return False, f"fork-free codegen smoke failed: {type(e).__name__}: {str(e)[-200:]}"
    want = ["0000000b", "00000016", "00000021", "0000002c", "00000037", "00000042", "0000004d", "00000058"]
    missing = [w for w in want if w not in console]
    if missing:
        return False, (
            f"fork-free kernel ran but produced the wrong result (missing {missing}); console tail: {console[-200:]!r}"
        )
    return True, f"fork-free codegen emits a runnable kernel with the correct result on the {target!r} reference sim"
