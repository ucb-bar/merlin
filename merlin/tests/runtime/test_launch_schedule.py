"""Compact launch groups preserve exact order without accepting new images."""

from copy import deepcopy
from itertools import islice
import unittest

from merlin.compile.launch_schedule import (
    MAX_LAUNCHES, expand_launch_schedule, iter_launch_schedule,
    make_launch_schedule, validate_launch_schedule,
)


class LaunchScheduleTest(unittest.TestCase):
    def setUp(self):
        self.images=[f"{i+1:064x}" for i in range(4)]
        self.groups=[dict(repeat=2,runs=[dict(image=0,count=64),dict(image=1,count=3),
                                       dict(image=2,count=1),dict(image=3,count=64)])]

    def validate(self,schedule,**changes):
        return validate_launch_schedule(schedule,image_sha256s=changes.get("images",self.images),
                                        placement=changes.get("placement","accelerator"))

    def test_counted_native_runs_repeat_in_exact_order(self):
        schedule=make_launch_schedule(self.images,self.groups)
        once=(self.images[0],)*64+(self.images[1],)*3+(self.images[2],)+(self.images[3],)*64
        self.assertEqual(schedule["total_launches"],264)
        self.assertEqual(expand_launch_schedule(schedule,image_sha256s=self.images),once*2)
        self.assertEqual(self.validate(schedule),schedule)
        self.assertEqual(self.groups[0]["runs"][0]["count"],64)
        # A consumer holds its validated copy even if the caller later mutates
        # the original metadata while iteration is pending.
        iterator=iter_launch_schedule(schedule,image_sha256s=self.images)
        schedule["groups"][0]["runs"][0]["count"]=1
        self.assertEqual(tuple(iterator),once*2)

    def test_sequential_groups_and_duplicate_actual_binary_hashes(self):
        images=[self.images[0],self.images[0]]
        groups=[dict(repeat=1,runs=[dict(image=1,count=2)]),
                dict(repeat=3,runs=[dict(image=0,count=1)])]
        schedule=make_launch_schedule(images,groups)
        self.assertEqual(expand_launch_schedule(schedule,image_sha256s=images),(images[0],)*5)

    def test_indices_hashes_placement_unused_images_and_unknown_fields_refuse(self):
        good=make_launch_schedule(self.images,self.groups)
        mutations=[]
        for location,key,value in (("root","program","new.bin"),("root","schema","other"),
                                   ("root","total_launches",263),("root","total_launches",True),
                                   ("group","program","new.bin"),("run","program","new.bin"),
                                   ("run","image",4),("run","image",-1),("run","image",True)):
            changed=deepcopy(good)
            target=changed if location=="root" else changed["groups"][0] if location=="group" else changed["groups"][0]["runs"][0]
            target[key]=value;mutations.append(changed)
        changed=deepcopy(good);changed["images"][0]="ff"*32;mutations.append(changed)
        changed=deepcopy(good);changed["groups"][0]["runs"].pop();changed["total_launches"]=136;mutations.append(changed)
        for schedule in mutations:
            with self.subTest(schedule=schedule),self.assertRaises(ValueError):
                self.validate(schedule)
        for placement in ("host","device","cpu",None):
            with self.subTest(placement=placement),self.assertRaises(ValueError):
                self.validate(good,placement=placement)
        with self.assertRaisesRegex(ValueError,"ordered native programs"):
            self.validate(good,images=list(reversed(self.images)))

    def test_nonpositive_noninteger_and_overflow_counts_refuse(self):
        for value in (0,-1,True,1.5,"2",None,MAX_LAUNCHES+1):
            for location in ("repeat","count"):
                groups=deepcopy(self.groups)
                if location=="repeat":
                    groups[0]["repeat"]=value
                else:
                    groups[0]["runs"][0]["count"]=value
                with self.subTest(value=value,location=location),self.assertRaises(ValueError):
                    make_launch_schedule(self.images,groups)
        with self.assertRaisesRegex(ValueError,"total exceeds"):
            make_launch_schedule(self.images,[dict(repeat=MAX_LAUNCHES,runs=self.groups[0]["runs"])])
        for groups in ([],[dict(repeat=1,runs=[])],{}):
            with self.assertRaises(ValueError):
                make_launch_schedule(self.images,groups)

    def test_large_valid_schedule_is_lazy_and_expansion_is_bounded(self):
        images=self.images[:1]
        schedule=make_launch_schedule(images,[dict(repeat=MAX_LAUNCHES,runs=[dict(image=0,count=1)])])
        self.assertEqual(tuple(islice(iter_launch_schedule(schedule,image_sha256s=images),3)),(images[0],)*3)
        with self.assertRaisesRegex(ValueError,"expansion limit"):
            expand_launch_schedule(schedule,image_sha256s=images,max_launches=10)
        with self.assertRaises(ValueError):
            expand_launch_schedule(schedule,image_sha256s=images,max_launches=True)


if __name__=="__main__":
    unittest.main()
