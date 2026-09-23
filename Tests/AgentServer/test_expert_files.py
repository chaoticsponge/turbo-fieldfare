import json
from pathlib import Path
import struct
import tempfile
import unittest

from expert_streaming import TensorFiles, ExpertCache, layout
from agent_models import weight_reservation
from test_launcher import launcher


def write_file(directory, header, payload):
    raw = json.dumps(header).encode()
    path = directory / 'model.safetensors'
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
    return path


class TensorFileTests(unittest.TestCase):
    def test_only_requested_expert_bytes_are_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            values=list(range(12))
            write_file(root, {'bank': {'dtype':'U32','shape':[3,2,2],'data_offsets':[0,48]}}, struct.pack('<12I',*values))
            with TensorFiles(root) as files:
                self.assertEqual(files.bytes_read,0)
                raw,dtype,shape=files.read('bank',1)
                self.assertEqual(struct.unpack('<4I',raw),(4,5,6,7))
                self.assertEqual(shape,(2,2))
                self.assertEqual(files.bytes_read,16)
                with self.assertRaises(ValueError): files.read('bank',3)
            self.assertFalse(files.descriptors)

    def test_bad_shape_overlap_and_unindexed_payload_are_refused(self):
        cases=[({'a':{'dtype':'U32','shape':[2],'data_offsets':[0,4]}},b'1234'),
               ({'a':{'dtype':'U32','shape':[1],'data_offsets':[0,4]},
                 'b':{'dtype':'U32','shape':[1],'data_offsets':[0,4]}},b'1234'),
               ({'a':{'dtype':'U32','shape':[1],'data_offsets':[0,4]}},b'12345678')]
        for header,data in cases:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);write_file(root,header,data)
                with self.assertRaises(ValueError): TensorFiles(root)

    def test_oversized_header_and_symlink_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/'model.safetensors'
            path.write_bytes(struct.pack('<Q',2**63))
            with self.assertRaises(ValueError): TensorFiles(root)
            path.unlink()
            real=root/'other.bin';real.write_bytes(b'12345678')
            path.symlink_to(real)
            with self.assertRaises(OSError): TensorFiles(root)

    def test_lru_never_exceeds_byte_cap_even_for_oversized_expert(self):
        cache=ExpertCache(10)
        for i in range(10):
            cache.get(i,6,lambda i=i: bytes([i])*6)
            self.assertLessEqual(cache.bytes,10)
        value=cache.get(9,6,lambda: self.fail('Cache miss'))
        self.assertEqual(value,bytes([9])*6)
        self.assertEqual(cache.hits,1)
        self.assertEqual(cache.get(11,20,lambda:b'x'*20),b'x'*20)
        self.assertEqual(cache.bytes,0)
        self.assertGreater(cache.evictions,0)

    def test_dense_and_unsupported_quantization_fail_closed(self):
        for config in [{'model_type':'qwen3_5'}, {'model_type':'qwen3_moe','quantization':{'bits':3}}]:
            with self.assertRaises(ValueError): layout(None,config,100)

    def test_streaming_profile_updates_admission_and_disables_weight_fusion(self):
        with tempfile.TemporaryDirectory() as folder:
            state=Path(folder).resolve()
            configured=launcher.settings(state,2,65536,48)
            profile={'planned_weight_bytes':1500000000,'cache_bytes':256*1024**2}
            launcher.configure_fleet(state,['coder','extract'],state,configured,
                launcher.command(state,state,8080,2,48),2,65536,
                stream_profiles={'coder':profile})
            routes=json.loads((state/'routes.json').read_text())['routes']
            self.assertEqual(weight_reservation(routes['coder']),1500000000*1.05)
            self.assertNotIn('expert_streaming',routes['extract'])
            preferences=json.loads((state/'model_settings.json').read_text())['models']
            self.assertFalse(preferences[routes['coder']['model']]['moe_gate_up_fusion_enabled'])
            self.assertEqual(preferences[routes['extract']['model']]['ttl_seconds'],180)

    def test_layer_lfu_retains_hits_across_repeated_expert_scans(self):
        from expert_streaming import LayerExpertCache
        cache=LayerExpertCache(24,['layer0','layer1'])
        for _ in range(3):
            for layer in ('layer0','layer1'):
                for expert in range(8):
                    cache.get((layer,expert),6,lambda: b'x'*6)
                    self.assertLessEqual(cache.bytes,24)
        self.assertGreater(cache.hits,0)
        self.assertEqual(cache.bytes,24)
        self.assertTrue(all(bucket.bytes==12 for bucket in cache.layers.values()))
