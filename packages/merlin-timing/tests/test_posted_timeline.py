"""Causal posted callbacks and completion ownership, not physical timing labels."""
from pathlib import Path
import shutil
import subprocess

def test_posted_source_timeline(tmp_path):
    root=Path(__file__).resolve().parents[1]
    compiler=shutil.which("c++")
    assert compiler
    for name in ("posted_timeline","posted_engine_stage"):
        check_source(root,compiler,tmp_path,name)

def check_source(root,compiler,tmp_path,name):
    output=tmp_path/name
    subprocess.run([compiler,"-std=c++17","-O1","-Wall","-Wextra","-Werror","-I",str(root/"include"),str(root/("tests/test_"+name+".cpp")),"-o",str(output)],check=True,capture_output=True,timeout=30)
    subprocess.run([str(output)],check=True,capture_output=True,timeout=5)
