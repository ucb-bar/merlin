"""merlin-rtl-introspect: extract structure-only facts from elaborated RTL (CIRCT/FIRRTL)."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

# Preserve the public RTL ISA import while its implementation lives in the
# selected chip package. Importing the namespace does not resolve that package.
from . import opu_isa as opu_isa
