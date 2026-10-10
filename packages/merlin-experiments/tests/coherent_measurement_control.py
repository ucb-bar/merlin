"""Original complete repeated CPU fixture, never selected as an author seed."""

import hashlib
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.perf.component_coherent_measurement import tensor_values
from merlin.runtime.direct_kernel_harness import render_direct_kernel


def render(cb, *, inputs, readback_policy, plan):
    return render_direct_kernel(
        cb,
        inputs=inputs,
        readback_policy=readback_policy,
        abi=plan.abi,
        invocation_plan=plan.invocation_plan,
        counter_plan=plan.counter_plan,
        phase_plan=plan.phase_plan,
    )


class Reader:
    def __init__(self, plan, symbols):
        self.plan = plan
        self.symbols = symbols
        self.prepared = None

    def prepare(self, *, cb, elf_path, workdir, target, simulator, backend):
        self.cb = cb
        self.elf = elf_path
        result = I.run(
            [str(self.symbols), "-sW", str(elf_path)],
            directory=workdir,
            stage="coherent_elf_symbols",
            inputs=(elf_path,),
            dependencies=(self.symbols, Path(__file__).resolve()),
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            capture_output=True,
            timeout=10,
            check=True,
        )
        roster, found = dict(self.plan.bind(cb)), {}
        for row in result.stdout.decode("ascii").splitlines():
            fields = row.split()
            if len(fields) != 8 or fields[3] != "OBJECT":
                continue
            name = fields[-1]
            if name not in roster and name.startswith("tensor_"):
                name = name.partition(".")[0]
            if name in roster:
                if name in found:
                    raise ValueError("diagnostic ELF repeats original objects")
                found[name] = int(fields[2])
        if found != roster:
            raise ValueError("diagnostic ELF omitted original complete object extents")
        self.prepared = self.plan.prepare(cb=cb, elf_path=elf_path, workdir=workdir)
        return self.prepared

    def decode(self, console):
        if console != "DONE\n":
            raise ValueError("diagnostic runner did not complete")
        output = Path(self.prepared["memory_readback"]["output_path"])
        with output.open("rb") as stream:
            raw = stream.read(self.plan.max_payload_bytes + 1)
        roster = self.plan.bind(self.cb)
        if len(raw) != sum(extent for _, extent in roster):
            raise ValueError("diagnostic readback is incomplete")
        objects, offset = {}, 0
        for name, extent in roster:
            objects[name] = raw[offset : offset + extent]
            offset += extent
        outputs = {
            argument["tensor"]: tensor_values(
                objects["tensor_" + str(index)],
                spec=self.cb["tensors"][argument["tensor"]],
                byte_order=self.plan.abi.byte_order,
            )
            for index, argument in enumerate(self.cb["kernel_abi"]["args"])
            if argument["access"] == "write"
        }
        return outputs, {
            "status": "complete",
            "original_elf": {"path": str(self.elf), "sha256": self.prepared["memory_readback"]["elf_sha256"]},
            "payload": {"path": str(output), "sha256": hashlib.sha256(raw).hexdigest()},
        }


def support_source(plan, cb, *, mode):
    roster = plan.bind(cb)
    history_names = {
        row.symbol
        for row in plan.invocation_plan.bind(
            cb, entry_symbol=plan.abi.entry_symbol, completion_symbol=plan.abi.completion_symbol
        )
    }
    declarations = "".join(
        f"extern {'volatile ' if name not in history_names else ''}unsigned char {name}[{size}];\n"
        for name, size in roster
        if not name.startswith("tensor_")
    )
    rows = [
        ("args[" + str(index) + "]" if name.startswith("tensor_") else name, extent)
        for index, (name, extent) in enumerate(roster)
    ]
    writer = "".join(f"publish({name},{extent});" for name, extent in rows)
    if mode == "partial":
        writer = writer.rsplit("publish(", 1)[0]
    return (
        '#include "htif.h"\n#include <stdint.h>\n'
        + declarations
        + "static unsigned char* args[5];static uint64_t tick,state;"
        + (
            "uint64_t raw_counter(void){return tick++%2?0:UINT64_MAX;}"
            if mode == "wrapping"
            else "uint64_t raw_counter(void){return tick++;}"
        )
        + "uint64_t raw_control(void){return state;}"
        "void __real_control_entry(void*,void*,void*,void*,void*);"
        "void __wrap_control_entry(void*a,void*b,void*y,void*z,void*w){"
        "args[0]=a;args[1]=b;args[2]=y;args[3]=z;args[4]=w;"
        + ("" if mode == "completion" else "__real_control_entry(a,b,y,z,w);")
        + "}void finish_copy(void){state++;}"
        'static void publish(const volatile unsigned char*p,unsigned n){const char*hex="0123456789abcdef";'
        "for(unsigned i=0;i<n;i++){char s[3]={hex[p[i]>>4],hex[p[i]&15],0};htif_puts(s);}}"
        "void __real_htif_exit(int);void __wrap_htif_exit(int code){"
        + ("__real_control_entry(args[0],args[1],args[2],args[3],args[4]);" if mode == "completion" else "")
        + ("args[0][0]^=1;" if mode == "input" else "")
        + (plan.counter_plan.storage_prefix + "_completed[0]=0;" if mode == "counter_count" else "")
        + ('htif_puts("COHERENT_HEX ");' + writer + 'htif_puts("\\n");' if mode != "missing" else "")
        + "__real_htif_exit(code);}"
    )
