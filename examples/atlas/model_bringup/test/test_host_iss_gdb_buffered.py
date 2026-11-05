"""Buffered RSP framing controls over real fragmented socket streams."""
import random
import socket
import sys
import threading
import time
from pathlib import Path
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).absolute().parents[1]/'tools'
sys.path[:0] = [str(TOOLS),str(TOOLS/'host_iss')]
import gdb_remote as protocol


class LimitedReceive:
    def __init__(self, connection, sizes):
        self.connection,self.sizes = connection,list(sizes)
    def recv(self, count):
        limit = self.sizes.pop(0) if self.sizes else count
        return self.connection.recv(min(count,limit))
    def __getattr__(self, name):return getattr(self.connection,name)


class BufferedProtocolTest(unittest.TestCase):
    def endpoint(self, payload, fragments=None, deadline=None):
        client,server = socket.socketpair()
        remote = protocol.Remote.__new__(protocol.Remote)
        remote.socket,remote.timeout,remote.deadline = client,1,deadline
        remote._buffer,remote._offset,remote.trace = b'',0,None
        errors = []
        def write():
            try:
                if fragments:
                    cursor = 0
                    for length in fragments:
                        server.sendall(payload[cursor:cursor+length]);cursor += length
                    if cursor < len(payload):server.sendall(payload[cursor:])
                else:server.sendall(payload)
            except Exception as error:errors.append(error)
        thread = threading.Thread(target=write)
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.addCleanup(client.close);self.addCleanup(server.close)
        self.addCleanup(lambda:thread.join(timeout=2))
        return remote,server,errors

    def test_every_split_of_escapes_rle_and_checksum(self):
        encoded = b'x}\x030*#}\x0aend'
        packet = protocol.frame(encoded)
        for split in range(1,len(packet)):
            with self.subTest(split=split):
                remote,server,errors = self.endpoint(packet)
                remote.socket = LimitedReceive(remote.socket,[split])
                self.assertEqual(remote.receive(),b'x#'+b'0'*7+b'*end')
                self.assertEqual(server.recv(1),b'+');self.assertFalse(errors)

    def test_one_byte_fragments_and_multiple_queued_responses(self):
        payload = b'noise+'+protocol.frame(b'}\x04')+protocol.frame(b'OK')+protocol.frame(b'0*}')
        remote,server,_ = self.endpoint(payload)
        remote.socket = LimitedReceive(remote.socket,[1]*len(payload))
        self.assertEqual(remote.receive(),b'$')
        self.assertEqual(remote.receive(),b'OK')
        self.assertEqual(remote.receive(),b'0'*97)
        self.assertEqual(server.recv(3),b'+++')

    def test_checksum_failure_nacks_and_preserves_next_frame(self):
        remote,server,_ = self.endpoint(b'$OK#00'+protocol.frame(b'next'))
        with self.assertRaisesRegex(ValueError,'checksum'):remote.receive()
        self.assertEqual(server.recv(1),b'-')
        self.assertEqual(remote.receive(),b'next')
        self.assertEqual(server.recv(1),b'+')

    def test_encoded_and_decoded_limits(self):
        self.assertEqual(protocol.MAX_PACKET,16*1024*1024)
        with patch.object(protocol,'MAX_PACKET',64):
            remote,_,_ = self.endpoint(protocol.frame(b'x'*65))
            with self.assertRaisesRegex(ValueError,'bound'):remote.receive()
            remote,_,_ = self.endpoint(protocol.frame(b'a*}'))
            with self.assertRaisesRegex(ValueError,'bound'):remote.receive()
            self.assertEqual(protocol.decode_response(b'x'*64),b'x'*64)

    def test_partial_escape_checksum_and_frame_deadlines(self):
        for payload in (b'$x}',b'$x*',b'$OK#',b'$OK#9'):
            with self.subTest(payload=payload):
                remote,_,_ = self.endpoint(payload,deadline=time.monotonic()+0.03)
                with self.assertRaises(TimeoutError):remote.receive()
        remote,_,_ = self.endpoint(protocol.frame(b'OK'),deadline=time.monotonic()-1)
        with self.assertRaises(TimeoutError):remote.receive()
        with self.assertRaises(TimeoutError):remote.request(b'?')

    def test_seeded_valid_payloads_and_fragmentation(self):
        randomizer = random.Random(531)
        for case in range(80):
            encoded,expected = bytearray(),bytearray()
            for _ in range(20):
                kind = randomizer.randrange(3)
                if kind == 0:
                    value = randomizer.choice(b'abcdef0123456789')
                    encoded.append(value);expected.append(value)
                elif kind == 1:
                    value = randomizer.randrange(256)
                    encoded.extend((125,value ^ 32));expected.append(value)
                else:
                    value = randomizer.choice(b'abcdef')
                    count = randomizer.choice((6,96,97))
                    encoded.extend((value,42,count+29));expected.extend(bytes([value])*(count+1))
            remote,_,_ = self.endpoint(protocol.frame(encoded))
            remote.socket = LimitedReceive(remote.socket,[randomizer.randrange(1,8) for _ in range(len(encoded))])
            with self.subTest(case=case):self.assertEqual(remote.receive(),bytes(expected))

    def test_disconnect_with_partial_frame(self):
        client,server = socket.socketpair();server.sendall(b'$ab}');server.close()
        remote = protocol.Remote.__new__(protocol.Remote)
        remote.socket,remote.timeout,remote.deadline = client,1,None
        self.addCleanup(client.close)
        with self.assertRaises(EOFError):remote.receive()


if __name__ == '__main__':unittest.main()
