"""Fail-closed admission and functional transport for exact SiFive UART sites."""
import io
import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
from unittest.mock import Mock

TOOLS=Path(__file__).absolute().parents[1]/'tools'
sys.path[:0]=[str(TOOLS),str(TOOLS/'host_iss')]
from run import select_uart_bridge
from result_lines import verify
from segment_uart import Transport
from uart_device import MASK,SifiveUART,constants,sites,validate_profile
BASE=0x10020000


def store(base,reg,offset):
    return ((offset>>5)&127)<<25|reg<<20|base<<15|2<<12|(offset&31)<<7|0x23


def load(base,reg):return base<<15|2<<12|reg<<7|3


def image(divisor=False,init=True,clobber=None,compressed=False):
    callback=[(0x100,4,load(5,10)),(0x104,4,store(5,11,0))]
    setup=[(0x200,4,BASE|5<<7|0x37)]
    if clobber is not None:setup.append((0x204,4,clobber))
    if compressed:
        setup += [(0x208,2,0x8002|8<<7|5<<2),(0x20a,2,2<<13|9<<7|1<<2|1),
                  (0x20c,2,6<<13|1<<10|1<<2)]
    else:
        if divisor:setup.append((0x208,4,store(5,6,24)))
        if init:setup.append((0x20c,4,store(5,6,8)))
    data={'atlas_result_uart_putchar':callback,'atlas_workload_main':setup}
    return SimpleNamespace(symbols={name:{} for name in data},instructions=data.__getitem__)


def site(offset,store=True,role=None):
    return dict(base=5,reg=6,offset=offset,store=store,signed=not store,length=4,
                uart_register=BASE+offset,uart_role=role or ('tx-data' if offset==0 else 'initialize'))


class AdmissionTest(unittest.TestCase):
    def test_transport_opt_in_preserves_native_default(self):
        self.assertFalse(select_uart_bridge(None,False))
        native=dict(kind='ns16550',base=0x10000000)
        self.assertFalse(select_uart_bridge(native,False))
        self.assertTrue(select_uart_bridge(dict(kind='sifive',base=BASE),True))
        for profile,requested in ((None,True),(native,True),(dict(kind='sifive',base=BASE),False),
                (dict(kind='ns16550',base=0x10010000),False)):
            with self.subTest(profile=profile),self.assertRaises(ValueError):select_uart_bridge(profile,requested)

    def test_actual_callback_and_compressed_init_are_admitted(self):
        result=sites(image(),dict(kind='sifive',base=BASE,divisor=None))
        self.assertEqual({r['uart_register'] for r in result.values()},{BASE,BASE+8})
        result=sites(image(compressed=True),dict(kind='sifive',base=BASE))
        self.assertEqual(result[0x20c]['uart_register'],BASE+8)

    def test_profile_and_divisor_instruction_binding(self):
        for profile in [dict(kind='ns16550',base=BASE),dict(kind='sifive',base=True),
                dict(kind='sifive',base=BASE+1),dict(kind='sifive',base=MASK-3),
                *[dict(kind='sifive',base=BASE,divisor=value) for value in (0,-1,True,1.0,65536)]]:
            with self.subTest(profile=profile),self.assertRaises(ValueError):validate_profile(profile)
        with self.assertRaises(ValueError):sites(image(init=False),dict(kind='sifive',base=BASE))
        with self.assertRaises(ValueError):sites(image(),dict(kind='sifive',base=BASE,divisor=17))
        with self.assertRaises(ValueError):sites(image(divisor=True),dict(kind='sifive',base=BASE))
        self.assertTrue(sites(image(divisor=True),dict(kind='sifive',base=BASE,divisor=17)))

    def test_branch_or_unknown_register_write_cannot_reuse_stale_constant(self):
        for clobber in (0x63,0x6f,5<<7|1<<12|0x13,load(0,5)):
            with self.subTest(clobber=clobber),self.assertRaises(ValueError):
                sites(image(clobber=clobber),dict(kind='sifive',base=BASE))
        known={0:0,5:BASE};constants(known,0x100,2,0xa001);self.assertEqual(known,{0:0})


class TransportTest(unittest.TestCase):
    def fixture(self,divisor=None):
        profile=dict(kind='sifive',base=BASE,divisor=divisor)
        stream=io.BytesIO();broker=SifiveUART(profile,stream)
        remote=Mock();registers={5:BASE,6:0};remote.reg.side_effect=registers.__getitem__
        return broker,remote,registers,stream,profile

    def test_enable_divisor_poll_and_bytes_with_immutable_profile(self):
        broker,remote,registers,stream,profile=self.fixture(17)
        registers[6]=1
        with self.assertRaises(ValueError):broker.access(remote,site(8))
        registers[6]=18
        with self.assertRaises(ValueError):broker.access(remote,site(24))
        registers[6]=17;broker.access(remote,site(24))
        registers[6]=1;broker.access(remote,site(8));broker.access(remote,site(8))
        broker.access(remote,site(0,False));remote.set_reg.assert_called_once_with(6,0)
        profile['base']=123
        registers[6]=65;broker.access(remote,site(0));self.assertEqual(stream.getvalue(),b'A')
        summary=broker.summary();self.assertFalse(summary['hardware_execution']);self.assertFalse(summary['baud_timing'])
        self.assertEqual((summary['characters'],summary['polls'],summary['divisor'],summary['tx_control_writes']),(1,1,17,2))
        self.assertEqual(summary['profile']['base'],BASE)

    def test_wrong_address_enable_value_role_and_unsupported_register_fail(self):
        broker,remote,registers,stream,_=self.fixture()
        for access in (site(0),site(0,False),site(24),site(8,False),site(8,role='tx-data'),site(12)):
            with self.subTest(access=access),self.assertRaises(ValueError):broker.access(remote,access)
        registers[6]=1;broker.access(remote,site(8))
        registers[6]=256
        with self.assertRaises(ValueError):broker.access(remote,site(0))
        registers[5]=BASE+4;registers[6]=65
        with self.assertRaises(ValueError):broker.access(remote,site(0))
        self.assertEqual(stream.getvalue(),b'')


class ResultLinesTest(unittest.TestCase):
    def lines(self):
        return ("ATLAS_KERNEL_BEGIN name=tiny-full launches_completed=0 launches_expected=2\n"
                "ATLAS_KERNEL_PROGRESS name=tiny-full launches_completed=2 launches_expected=2 boundary=layer\n"
                "ATLAS_KERNEL_PASS name=tiny-full launches_completed=2 launches_expected=2\n").encode()

    def test_complete_uart_transcript(self):
        self.assertEqual(verify(self.lines(), "tiny-full", 2)["progress_records"], 1)
        self.assertEqual(verify(self.lines(), "tiny-full", 2, progress_launches=[2])["final"], "PASS")
        with self.assertRaisesRegex(ValueError, "coverage"):
            verify(self.lines(), "tiny-full", 2, progress_launches=[1, 2])

    def test_missing_failure_reordered_duplicate_and_wrong_workload_refused(self):
        raw = self.lines()
        for altered in (b"", raw.splitlines(keepends=True)[0], raw + raw.splitlines(keepends=True)[-1],
                        raw.replace(b"_PASS", b"_FAIL"), raw.replace(b"completed=2", b"completed=1"),
                        raw.replace(b"name=tiny-full", b"name=smol-full"),
                        b"".join(raw.splitlines(keepends=True)[::2]),
                        raw.replace(b"boundary=layer", b"boundary=layer name=tiny-full")):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                verify(altered, "tiny-full", 2)

class SegmentUARTTest(unittest.TestCase):
    def test_native_transport_preserves_existing_guest_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = Transport(dict(kind="ns16550", base=0x10000000), False, mock.Mock(), Path(directory))
            self.assertEqual(transport.prepare(), "uart.log")
            self.assertFalse(transport.handle(mock.Mock(), 12))
            transport.register(mock.Mock(), mock.Mock(), {1,2,3})
            self.assertEqual(transport.summary()["backend"], "qemu-native-ns16550")
            transport.close()

    def test_sifive_initialization_precedes_ready_without_a_device(self):
        base = 0x10020000
        access = dict(base=5, reg=6, offset=8, store=True, signed=False, length=4,
                      uart_register=base+8, uart_role="initialize")
        with tempfile.TemporaryDirectory() as directory, mock.patch("segment_uart.sites", return_value={100:access}):
            transport = Transport(dict(kind="sifive", base=base), True, mock.Mock(), Path(directory))
            self.assertEqual(transport.summary()["backend"], "modeled-sifive-uart-not-started")
            self.assertEqual(transport.prepare(), "qemu-native-uart.log")
            remote = mock.Mock()
            remote.reg.side_effect = {5:base, 6:1}.__getitem__
            image = mock.Mock()
            remote.read.return_value = image.bytes_at.return_value = b"abcd"
            transport.register(remote, image, {200})
            self.assertTrue(transport.handle(remote, 100))
            self.assertTrue(transport.summary()["tx_enabled"])
            remote.set_reg.assert_called_once_with(32, 104)
            with self.assertRaisesRegex(ValueError, "aliases"):
                transport.register(remote, image, {100})
            transport.close()


if __name__=='__main__':unittest.main()
