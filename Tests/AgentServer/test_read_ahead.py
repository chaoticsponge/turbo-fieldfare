import json
from pathlib import Path
import tempfile
import threading
import unittest

from expert_read_ahead import ExpertReadAhead, ReadAheadPool
from test_launcher import launcher


class ReadAheadTests(unittest.TestCase):
    def test_one_future_and_shared_byte_cap_with_decode_on_caller(self):
        pool=ReadAheadPool(100);a=ExpertReadAhead(pool);b=ExpertReadAhead(pool)
        started=threading.Event();finish=threading.Event();threads=[]
        def read():
            threads.append(threading.get_ident());started.set()
            if not finish.wait(3):raise TimeoutError('test did not release reader')
            return b'x'*40
        try:
            self.assertTrue(a.schedule('a',40,read));self.assertTrue(started.wait(3))
            self.assertFalse(a.schedule('another',1,lambda:b''))
            self.assertFalse(b.schedule('b',40,lambda:b'x'*40))
            self.assertEqual(pool.snapshot()['reserved_bytes'],80)
            finish.set()
            caller=threading.get_ident()
            def decode(data):
                self.assertEqual(threading.get_ident(),caller)
                return len(data)
            self.assertEqual(a.consume('a',lambda:self.fail('Unneeded synchronous read'),decode),40)
            self.assertNotEqual(threads[0],caller)
            self.assertEqual(pool.snapshot()['reserved_bytes'],0)
            self.assertTrue(b.schedule('b',40,lambda:b'x'*40))
            self.assertLessEqual(pool.snapshot()['peak_reserved_bytes'],100)
        finally:finish.set();a.close();b.close()
        self.assertEqual(pool.snapshot()['reserved_bytes'],0)

    def test_read_and_decode_errors_release_budget(self):
        for where in ('read','decode'):
            pool=ReadAheadPool(100);worker=ExpertReadAhead(pool)
            def broken():raise ValueError('broken bytes')
            try:
                worker.schedule('a',20,broken if where=='read' else lambda:b'x'*20)
                with self.assertRaisesRegex(ValueError,'broken bytes'):
                    worker.consume('a',lambda:b'',lambda data:broken() if where=='decode' else data)
                self.assertEqual(pool.snapshot()['reserved_bytes'],0)
                self.assertIsNone(worker.pending)
            finally:worker.close()

    def test_close_waits_for_reader_before_releasing_resources(self):
        pool=ReadAheadPool(100);worker=ExpertReadAhead(pool)
        started=threading.Event();finish=threading.Event();closed=threading.Event()
        def read():
            started.set()
            if not finish.wait(3):raise TimeoutError('test did not release reader')
            return b'x'*20
        worker.schedule('a',20,read);self.assertTrue(started.wait(3))
        def close():worker.close();closed.set()
        thread=threading.Thread(target=close);thread.start()
        try:
            self.assertFalse(closed.wait(.02))
            self.assertEqual(pool.snapshot()['reserved_bytes'],40)
        finally:finish.set();thread.join(3)
        self.assertFalse(thread.is_alive());self.assertTrue(closed.is_set())
        self.assertEqual(pool.snapshot()['reserved_bytes'],0)
        self.assertFalse(worker.schedule('a',20,lambda:b''))

    def test_stale_and_oversized_prefetch_fall_back_to_synchronous_read(self):
        pool=ReadAheadPool(30);worker=ExpertReadAhead(pool)
        try:
            self.assertFalse(worker.schedule('large',20,lambda:self.fail('Oversized task ran')))
            worker.schedule('stale',10,lambda:b'stale')
            self.assertEqual(worker.consume('wanted',lambda:b'wanted',lambda value:value),b'wanted')
            self.assertEqual(pool.snapshot()['reserved_bytes'],0)
        finally:worker.close()

    def test_admission_accounts_for_read_copy_headroom(self):
        with tempfile.TemporaryDirectory() as folder:
            state=Path(folder);config=launcher.settings(state,2,65536,48)
            profile={'planned_weight_bytes':1500000000,'cache_bytes':256*1024**2,'largest_expert_bytes':5000000}
            launcher.configure_fleet(state,['coder'],state,config,launcher.command(state,state,8080,2,48),
                2,65536,stream_profiles={'coder':profile},read_ahead_pool_bytes=16*1024**2)
            saved=json.loads((state/'routes.json').read_text())
            entry=saved['routes']['coder']
            self.assertTrue(entry['expert_streaming']['read_ahead'])
            self.assertEqual(entry['resident_weight_bytes'],1510000000)
            self.assertEqual(saved['expert_read_ahead']['pool_bytes'],16*1024**2)
