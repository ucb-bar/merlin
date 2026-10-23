"""Extended rational sentinels whose NaN payload is discarded only by a byte Q.

Finite arithmetic remains Fraction arithmetic. These sentinels perform no
floating computation; final materialization without a quantizer must refuse.
"""
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class BoundarySpecial:
    sign: int  # -1/+1 infinity; 0 NaN.
    word: int | None = None  # Quieted single-input f32 NaN; None means ambiguous.

    def __eq__(self,other):
        return isinstance(other,BoundarySpecial) and bool(self.sign) and self.sign==other.sign

    def __float__(self):
        return float('nan') if not self.sign else float('inf')*self.sign

    def __neg__(self):return BoundarySpecial(-self.sign,self.word^0x80000000 if self.word is not None else None)
    def __abs__(self):return BoundarySpecial(abs(self.sign))
    def __add__(self,other):
        if isinstance(other,BoundarySpecial):
            if not self.sign and not other.sign:return self if self.word==other.word else BoundarySpecial(0)
            if not self.sign:return self
            if not other.sign:return other
            return self if self.sign==other.sign else BoundarySpecial(0)
        return self
    __radd__=__add__
    def __sub__(self,other):return self+(-other)
    def __rsub__(self,other):return -self+other
    def __mul__(self,other):
        if isinstance(other,BoundarySpecial):return BoundarySpecial(self.sign*other.sign)
        if not self.sign:return self if other>=0 else BoundarySpecial(0)
        return BoundarySpecial(0 if not other else self.sign*(1 if other>0 else -1))
    __rmul__=__mul__
    def __truediv__(self,other):
        if isinstance(other,BoundarySpecial):return BoundarySpecial(0)
        return BoundarySpecial(self.sign*(1 if other>=0 else -1))
    def __rtruediv__(self,other):
        return Fraction(0) if self.sign else BoundarySpecial(0)
    def __lt__(self,other):
        if not self.sign:return False
        return self.sign<other.sign if isinstance(other,BoundarySpecial) and other.sign else self.sign<0 if not isinstance(other,BoundarySpecial) else False
    def __le__(self,other):return self==other or self<other if self.sign else False
    def __gt__(self,other):return (-self)<(-other)
    def __ge__(self,other):return self==other or self>other if self.sign else False
