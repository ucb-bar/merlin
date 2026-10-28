"""Run a checked complete-call ELF pair on an explicitly supplied Spike oracle.

This local functional check does not qualify FireSim performance or substitute
an unpinned oracle for the software specification's provider.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from build_shared_chunk_elf import digest


def check(guest, receipt_path, seconds):
    import measure
    guest=guest.resolve(strict=True); receipt_path=receipt_path.resolve(strict=True)
    destination=guest/"spike-results.json"
    if destination.exists():
        raise ValueError("Refusing to overwrite Spike evidence")
    artifact=json.loads((guest/"artifact.json").read_text())
    receipt=json.loads(receipt_path.read_text())
    for kind in ("spike","plugin"):
        if digest(receipt[kind+"_path"])!=receipt[kind+"_sha256"]:
            raise ValueError("Simulator binding changed: "+kind)
    for row in [artifact['package'],artifact['fixture'],artifact['reference_provenance'],*artifact['elfs'].values()]:
        if digest(row['path'])!=row['sha256']:
            raise ValueError("Build input changed: "+row['path'])
    records=[]
    for label in ("valid","bad-oracle"):
        elf=artifact["elfs"][label]
        code,text=measure._spike(elf["path"],SimpleNamespace(spike_receipt=receipt_path),timeout=seconds)
        log=guest/("spike-"+label+".log");log.write_text(text)
        passed=(code==0 and "DSE_PASS" in text.splitlines()) if label=="valid" else (
            code==22 and "TENSORIR_NUMERICAL_FAILURE" in text)
        records.append(dict(label=label,passed=passed,returncode=code,elf=elf,
                            log_sha256=digest(log),wall_timeout_seconds=seconds))
    result=dict(status="passed" if all(r['passed'] for r in records) else "failed",
                functional_only=True,performance_qualified=False,
                oracle_receipt_sha256=digest(receipt_path),oracle_receipt=receipt,
                checks=records,checker_sha256=digest(__file__),artifact_sha256=digest(guest/"artifact.json"))
    destination.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status=result['status'],receipt=str(destination))))
    if result['status']!='passed':raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guest',type=Path,required=True)
    parser.add_argument('--simulator-receipt',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=60)
    args=parser.parse_args()
    if not 1<=args.seconds<=300:parser.error('Use a bounded 1..300 second timeout')
    check(args.guest,args.simulator_receipt,args.seconds)


if __name__=='__main__':main()
