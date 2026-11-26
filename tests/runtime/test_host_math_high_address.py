"""The selected RV64 supplier must link in the upper positive 32-bit range."""
import json
import shutil
import subprocess

from merlin.runtime import host_math_supplier as host_math
from merlin.runtime.host_math_supplier.build import build


def test_selected_supplier_links_above_medlow_range(tmp_path):
    cc = shutil.which('riscv64-unknown-elf-gcc')
    assert cc, 'RV64 supplier link validation requires the bare-metal toolchain'
    build('rv64', tmp_path/'supplier', host_math.POLICY_PATH)
    folder = tmp_path/'supplier/rv64'
    receipt = json.loads((folder/'build-receipt.json').read_bytes())
    archive = folder/'libmerlin-host-math.a'
    assert host_math.sha256(archive) == receipt['archive_sha256']
    caller = tmp_path/'caller.c'
    caller.write_text('extern float merlin_host_sinf(float);\n'
                      'volatile float input=0.75f;\n'
                      'int main(void){return merlin_host_sinf(input)>0.0f;}\n')
    elf = tmp_path/'high.elf'
    # The wrapper references every policy-selected routine, so this link checks
    # the complete supplier, including its local constant pools and support.
    command = [cc, '-O2', '-march=rv64gc', '-mabi=lp64d', '-mcmodel=medany',
               '-nostartfiles', '-Wl,--no-relax', '-Wl,-e,main', '-Wl,-Ttext=0x80000000',
               str(caller), str(archive), '-Wl,--start-group', '-lc', '-lm', '-lgcc',
               '-Wl,--end-group', '-o', str(elf)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert elf.read_bytes()[:6] == b'\x7fELF\x02\x01'
