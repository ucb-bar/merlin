"""The example adapter consumes the selected map and a caller's namespace."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from generate_host_map import generate


class PlatformTest(unittest.TestCase):
    def test_map_and_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'map.json'
            source.write_text(json.dumps(dict(imem_base=0x123000, csr_base=0x456000,
                                             imem_capacity_bytes=0x789000)))
            generate(source, root/'ee290_host_map.h')
            header = (root/'ee290_host_map.h').read_text()
            self.assertIn('0x123000ULL', header)
            (root/'atlas_host.h').write_text('''
#include <stddef.h>
#include <stdint.h>
typedef struct { uintptr_t imem_base, csr_base; size_t imem_capacity_bytes; } atlas_host_map;
typedef struct { int (*read32)(void); int (*write32)(void); } atlas_host_mmio;
typedef struct { int (*write)(void); int (*read)(void); } atlas_tile_memory_io;
typedef struct { int coherent; } atlas_host_memory;
typedef struct { int unused; } atlas_host;
static int atlas_host_direct_read32(void) { return 0; }
static int atlas_host_direct_write32(void) { return 0; }
static int atlas_host_direct_memory_write(void) { return 0; }
static int atlas_host_direct_memory_read(void) { return 0; }
static int atlas_host_init(atlas_host *h, atlas_host_map m, atlas_host_mmio io, atlas_host_memory mem) {
 (void)h; (void)io; return m.imem_base!=0x123000 || m.csr_base!=0x456000 || m.imem_capacity_bytes!=0x789000 || !mem.coherent;
}
''')
            (root/'custom.h').write_text('''
typedef struct { int unused; } custom_context;
static int custom_bind(custom_context *c, atlas_host *h, atlas_tile_memory_io io, size_t p) {
 (void)c; (void)h; (void)io; return p!=100;
}
''')
            (root/'test.c').write_text('''
#define HARNESS_APPLICATION_HEADER "custom.h"
#define HARNESS_APPLICATION_NAMESPACE custom
#include "platform.h"
#include "platform.h"
int main(void) { custom_context c; return harness_platform_bind(&c,100); }
''')
            subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror',
                            '-I', str(root), '-I', str(Path(__file__).parent),
                            str(root/'test.c'), '-o', str(root/'test')], check=True)
            subprocess.run([str(root/'test')], check=True)
