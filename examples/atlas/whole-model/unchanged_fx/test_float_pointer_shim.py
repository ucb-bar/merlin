"""The default non-opaque float ABI uses typed pointer arrays in its C shim."""
import ctypes as ct
from pathlib import Path
import shutil

import pytest

from compile_capsule import build_host_library


def test_default_nonopaque_all_float_application(tmp_path):
    if not shutil.which('cc'):
        pytest.skip('C compiler required')
    package = tmp_path / 'c-package'
    package.mkdir()
    (package / 'capsule.h').write_text('''#include <stddef.h>
typedef struct {int unused;} capsule_driver_state;
typedef struct {void **contexts; size_t count;} capsule_driver_contexts;
static inline int capsule_invoke(capsule_driver_state *s, void *w, size_t n,
 const float *const *i, float *const *o, capsule_driver_contexts *d) {
 (void)s;(void)w;(void)n;(void)d; o[0][0]=i[0][0];o[0][1]=i[0][1];return 0;
}
''')
    (package/'capsule.c').write_text('#include "capsule.h"\n')
    manifest = {'application_name':'capsule', 'regions':[{}],
                'application_interface':{'header':'capsule.h','source':'capsule.c','workspace_bytes':64,
                    'inputs':['x'],'outputs':['y']}, 'link_sources':[]}
    build_host_library(tmp_path, manifest)
    library = ct.CDLL(str(package/'capsule.so'))
    library.run.argtypes = [ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_void_p)]
    incoming=(ct.c_float*2)(1.25,-0.0);outgoing=(ct.c_float*2)()
    assert library.run((ct.c_void_p*1)(ct.addressof(incoming)),
                       (ct.c_void_p*1)(ct.addressof(outgoing)))==0
    assert bytes(incoming)==bytes(outgoing)
