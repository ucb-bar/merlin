"""Handoff authenticates multi-image order while preserving old call metadata."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from merlin.compile.application_handoff import application_handoff
from merlin.compile.launch_schedule import make_launch_schedule,expand_launch_schedule
from merlin.semantic_compiler.model import KernelRequest,SemanticNode,TensorType


class ApplicationHandoffScheduleTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        tensor=TensorType((1,),"f32","raw-bit-pattern")
        self.source=KernelRequest((SemanticNode("x","input",(),tensor,effect="input"),
            SemanticNode("y","identity",("x",),tensor)),("y",),("external",),(("x","external"),),"test-target")
        self.interface=dict(source="app.c",header="app.h",entry="app_run",inputs=["x"])
        self.files={"c-package/app.c":b"int app_run(void){return 0;}\n","c-package/app.h":b"int app_run(void);\n"}
        self.paths,self.images=[],[]
        for i in range(4):
            name=f"program{i}";folder=self.root/name;folder.mkdir()
            binary=(i+1).to_bytes(4,"little");digest=sha256(binary).hexdigest()
            (folder/"program.bin").write_bytes(binary)
            (folder/"manifest.json").write_text(json.dumps(dict(binary_sha256=digest)))
            (folder/"execution_plan.json").write_text(json.dumps(dict(schema="test",streaming=dict(launch_count=99))))
            self.paths.append(name);self.images.append(digest)

    def tearDown(self):
        self.temporary.cleanup()

    def region(self,name,repeat):
        schedule=make_launch_schedule(self.images,[dict(repeat=repeat,runs=[
            dict(image=0,count=64),dict(image=1,count=3),dict(image=2,count=1),dict(image=3,count=64)])])
        return dict(id=name,placement="accelerator",native_programs=list(self.paths),
                    description=dict(device_launch_schedule=schedule))

    def emit(self,regions):
        files=application_handoff(self.source,self.interface,regions,self.files,self.root)
        return json.loads(files["handoff/abi.json"])

    def test_same_four_images_have_distinct_per_call_group_repeats(self):
        host=dict(id="host",placement="host",native_programs=[],description={})
        abi=self.emit([self.region("short",1),host,self.region("long",4)])
        self.assertEqual(len(abi["programs"]),4)
        for i,count in ((0,132),(2,528)):
            call=abi["calls"][i]
            self.assertEqual(call["images"],self.images)
            self.assertEqual(call["device_launch_schedule"]["total_launches"],count)
            self.assertEqual(len(expand_launch_schedule(call["device_launch_schedule"],image_sha256s=call["images"])),count)
            self.assertNotIn("launches_per_call",call)
        self.assertNotIn("device_launch_schedule",abi["calls"][1])
        self.assertTrue(all("device_launch_schedule"not in p for p in abi["programs"]))

    def test_legacy_single_image_count_and_default_metadata_are_unchanged(self):
        old=dict(id="old",placement="accelerator",native_programs=self.paths[:1],description={})
        explicit=deepcopy(old);explicit["id"]="explicit";explicit["description"]={"launches_per_call":7}
        abi=self.emit([old,explicit])
        self.assertNotIn("device_launch_schedule",abi["calls"][0])
        self.assertNotIn("launches_per_call",abi["calls"][0])
        self.assertEqual(abi["calls"][1]["launches_per_call"],7)
        self.assertEqual(abi["programs"][0]["execution"]["streaming"]["launch_count"],99)

    def test_schedule_legacy_coexistence_host_or_reordered_images_refuse(self):
        for fault in ("legacy","host","order","missing","extra"):
            region=self.region("bad",1)
            if fault=="legacy":
                region["description"]["launches_per_call"]=132
            elif fault=="host":
                region["placement"]="host"
            elif fault=="order":
                region["native_programs"]=list(reversed(self.paths))
            elif fault=="missing":
                region["native_programs"].pop()
            else:
                region["native_programs"].append(self.paths[0])
            with self.subTest(fault=fault),self.assertRaises(ValueError):
                self.emit([region])

    def test_changed_actual_binary_cannot_keep_old_schedule_hash(self):
        region=self.region("bad",1)
        binary=b"\x05\0\0\0";folder=self.root/self.paths[0]
        (folder/"program.bin").write_bytes(binary)
        (folder/"manifest.json").write_text(json.dumps(dict(binary_sha256=sha256(binary).hexdigest())))
        with self.assertRaisesRegex(ValueError,"ordered native programs"):
            self.emit([region])


if __name__=="__main__":
    unittest.main()
