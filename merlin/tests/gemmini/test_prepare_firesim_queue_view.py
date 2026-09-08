from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tarfile

SCRIPT = (Path(__file__).parents[2] / "experiments/gemmini_perf_bench/scripts"
          / "prepare_firesim_queue_view.py")
SPEC = importlib.util.spec_from_file_location("prepare_firesim_queue_view", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
prepare = MODULE.prepare


def test_view_binds_workload_to_elf_and_exact_bitstream_driver(tmp_path: Path) -> None:
    chipyard = tmp_path / "chipyard"
    firesim = chipyard / "sims" / "firesim"
    deploy = firesim / "deploy"
    workloads = deploy / "workloads"
    workloads.mkdir(parents=True)
    (chipyard / "generators").mkdir()
    (firesim / "sim").mkdir()
    for path in (chipyard / "env.sh", firesim / "sourceme-manager.sh", deploy / "firesim"):
        path.write_text("fixture\n")
    (workloads / "bare.json").write_text(json.dumps({
        "benchmark_name": "bare",
        "common_bootbinary": "bare.elf",
        "common_rootfs": None,
    }))

    quintuplet = "xilinx_alveo_u250-firesim-FireSim-Fixture-BaseXilinxAlveoU250Config"
    driver = firesim / "sim" / "output" / "xilinx_alveo_u250" / quintuplet / "FireSim-xilinx_alveo_u250"
    driver.parent.mkdir(parents=True)
    driver.write_text("driver\n")
    driver.chmod(driver.stat().st_mode | stat.S_IXUSR)
    bitstream = tmp_path / "firesim.tar.gz"
    metadata = f"firesim-deployquintuplet:{quintuplet},firesim-commit:fixture\n".encode()
    with tarfile.open(bitstream, "w:gz") as archive:
        info = tarfile.TarInfo("xilinx_alveo_u250/metadata")
        info.size = len(metadata)
        archive.addfile(info, io.BytesIO(metadata))
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"exact elf")

    identity = prepare(argparse.Namespace(
        chipyard=chipyard,
        bitstream=bitstream,
        elf=elf,
        workload_prefix="model-cold",
        workload_template="bare",
        view_root=tmp_path / "views",
    ))
    view = Path(identity["view"])
    try:
        workload = json.loads((view / "sims/firesim/deploy/workloads" /
                               f"{identity['workload']}.json").read_text())
        assert identity["workload"].startswith("model-cold-")
        assert workload["common_bootbinary"] == identity["bootbinary"]
        assert identity["deploy_quintuplet"] == quintuplet
        assert identity["driver"] == str(driver)
        assert not os.access(view / "sims/firesim/deploy", os.W_OK)
    finally:
        for directory in (view / "sims/firesim/deploy/workloads", view / "sims/firesim/deploy",
                          view / "sims/firesim", view / "sims", view):
            directory.chmod(stat.S_IRWXU)
