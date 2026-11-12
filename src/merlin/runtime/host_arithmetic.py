"""Explicit shared host CPU implementations of independent source FMA chains.

The selected host ISA/ABI and source arithmetic/effect permissions are required.
These emitters issue no accelerator instruction and select no graph transform,
approximation, workload or schedule. Their presence proves neither the actual
host capability nor the original source permissions; callers establish those
independently and bind emitted bytes into ordinary compilation evidence.
"""

from dataclasses import dataclass

from merlin.compiler.lower.model.source_fma_batch import SourceFmaBatchContract


def _identifier(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name[0].isdigit()
        or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in name)
    ):
        raise ValueError("explicit C identifier required")
    return name


def _profile(isa: str, abi: str) -> None:
    if isa != "rv64gc" or abi != "lp64d":
        raise ValueError("explicit RV64GC/lp64d host arithmetic capability required")


@dataclass(frozen=True)
class SourceFmaPairCapability:
    isa: str
    abi: str
    ieee_f32: bool
    gradual_underflow: bool
    stable_rounding: bool
    nontrapping_unobserved_flags: bool

    def header(self, *, namespace: str = "merlin_host_source") -> str:
        _profile(self.isa, self.abi)
        _identifier(namespace)
        if any(
            value is not True
            for value in (
                self.ieee_f32,
                self.gradual_underflow,
                self.stable_rounding,
                self.nontrapping_unobserved_flags,
            )
        ):
            raise ValueError("explicit source arithmetic and effect contract required")
        return r"""#ifndef __GUARD__
#define __GUARD__
static inline void __NAMESPACE___fma_pair(
 float al,float bl,float cl,float ah,float bh,float ch,float *low,float *high){
 float l,h;
 __asm__("fmadd.s %0,%2,%3,%4\n\tfmadd.s %1,%5,%6,%7"
         : "=&f"(l),"=&f"(h)
         : "f"(al),"f"(bl),"f"(cl),"f"(ah),"f"(bh),"f"(ch));
 *low=l;*high=h;
}
#define MERLIN_SOURCE_F32_FMA_PAIR __NAMESPACE___fma_pair
#endif
""".replace("__GUARD__", namespace.upper() + "_FMA_PAIR_H").replace("__NAMESPACE__", namespace)


@dataclass(frozen=True)
class SourceFmaBatchCapability:
    isa: str
    abi: str
    contract: SourceFmaBatchContract

    def header(self, *, namespace: str = "merlin_host_source") -> str:
        _profile(self.isa, self.abi)
        _identifier(namespace)
        if not isinstance(self.contract, SourceFmaBatchContract):
            raise ValueError("typed source FMA batch contract required")
        self.contract.validate()
        # Each output is read/write and early-clobber, so a live input from
        # another lane cannot share its register. Operand order is unchanged.
        instructions = "\\n\\t".join(f"fmadd.s %{lane},%{lane + 8},%{lane},%16" for lane in range(8))
        outputs = ",".join(f'"+&f"(product[{lane}])' for lane in range(8))
        inputs = ",".join(f'"f"(fraction[{lane}])' for lane in range(8))
        guard = namespace.upper() + "_FMA_BATCH_H"
        return (
            f"#ifndef {guard}\n"
            f"#define {guard}\n"
            f"static inline void {namespace}_fma_eight(\n"
            " const float fraction[8],float product[8],float coefficient) {\n"
            f' __asm__("{instructions}"\n'
            f"         : {outputs}\n"
            f'         : {inputs},"f"(coefficient));\n'
            "}\n"
            f"#define MERLIN_SOURCE_F32_FMA_EIGHT {namespace}_fma_eight\n"
            "#endif\n"
        )
