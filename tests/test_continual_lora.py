#!/usr/bin/env python3
"""Independent stdlib oracle for the raw-syscall experimental LoRA executable."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE_BYTES = 38464
WQ_OFFSET = 35392
PARAM_OFFSET = 64 + BASE_BYTES
CHECKSUM_OFFSET = PARAM_OFFSET + 256
CKPT_BYTES = CHECKSUM_OFFSET + 8
COMMANDS = []
METRICS = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(cmd, cwd=None):
    start = time.perf_counter()
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, capture_output=True, timeout=60)
    COMMANDS.append({"command": [str(x) for x in cmd], "cwd": str(cwd or ROOT),
                     "exit_code": p.returncode, "seconds": time.perf_counter()-start,
                     "stdout_bytes": len(p.stdout), "stdout_sha256": digest(p.stdout),
                     "stderr": p.stderr.decode("utf-8", "replace")[:400]})
    return p


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def floats(data):
    return list(struct.unpack("<" + "f" * (len(data)//4), data))


def pack(xs):
    return struct.pack("<" + "f" * len(xs), *xs)


def checksum(data):
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
    return h


def reseal(data):
    data = bytearray(data)
    struct.pack_into("<Q", data, CHECKSUM_OFFSET, checksum(data[:CHECKSUM_OFFSET]))
    return bytes(data)


def ref_forward(w, p, x):
    h = [sum(p[r*16+k]*x[k] for k in range(16)) for r in range(2)]
    return [sum(w[j*16+k]*x[k] for k in range(16)) +
            sum(p[32+j*2+r]*h[r] for r in range(2)) for j in range(16)]


def ref_gradient(w, p, x, target):
    h = [sum(p[r*16+k]*x[k] for k in range(16)) for r in range(2)]
    err = [y-t for y, t in zip(ref_forward(w, p, x), target)]
    dh = [sum(p[32+j*2+r]*err[j] for j in range(16)) for r in range(2)]
    return [dh[r]*x[k] for r in range(2) for k in range(16)] + [
        err[j]*h[r] for j in range(16) for r in range(2)]


def loss(w, p, records):
    return sum(sum((a-b)**2 for a, b in zip(ref_forward(w, p, x), target))/2
               for x, target in records) / len(records)


class ContinualLoRA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="niyah-lora-")
        cls.work = Path(cls.tmp.name)
        cls.bin = cls.work / "continual_lora"
        cls.testbin = cls.work / "continual_lora_test_step"
        flags = ["cc", "-nostdlib", "-static", "-no-pie", "-Wl,--build-id=none"]
        for exe, extra in [(cls.bin, []), (cls.testbin, ["-Wa,--defsym,LORA_TEST_STEP=1"])]:
            p = run(flags + extra + ["-o", exe, ROOT/"src/continual_lora.s"])
            if p.returncode:
                raise RuntimeError(p.stderr.decode())
        cls.source = Path(os.environ.get("NIYAH_LORA_BASELINE", ROOT/"src/attention_lm.s"))
        src = cls.source.read_bytes()
        cls.source_hash = digest(src)
        (cls.work/"base_source.s").write_bytes(src)
        (cls.work/"data").mkdir()
        for name in ["corpus.txt", "heldout.txt"]:
            shutil.copyfile(ROOT/"data"/name, cls.work/"data"/name)
        p = run(flags + ["-Wa,--defsym,EXPORT_WEIGHTS=1", "-Wa,--defsym,EPOCHS=240",
                         "-o", cls.work/"base_trainer", cls.work/"base_source.s"], cwd=cls.work)
        if p.returncode:
            raise RuntimeError(p.stderr.decode())
        p = run([cls.work/"base_trainer"], cwd=cls.work)
        if p.returncode or len(p.stdout) < BASE_BYTES:
            raise RuntimeError("private native base training failed")
        prefix = p.stdout[:-BASE_BYTES]
        if b"heldout_loss_x1000 " not in prefix or not prefix.endswith(b"\n"):
            raise RuntimeError("unexpected base trainer export protocol")
        cls.base_bytes = p.stdout[-BASE_BYTES:]
        cls.base_hash = digest(cls.base_bytes)
        cls.base = cls.work/"base.f32"
        cls.base.write_bytes(cls.base_bytes)
        cls.w = floats(cls.base_bytes[WQ_OFFSET:WQ_OFFSET+1024])
        cls.initial = cls.work/"initial.lora"
        p = run([cls.bin, "init", cls.base, cls.initial])
        if p.returncode:
            raise RuntimeError("adapter initialization failed: " + p.stderr.decode())
        cls.initial_bytes = cls.initial.read_bytes()
        cls.initial_params = floats(cls.initial_bytes[PARAM_OFFSET:CHECKSUM_OFFSET])
        cls.serial = 0
        METRICS.update({"base_source_sha256": cls.source_hash, "base_sha256": cls.base_hash,
                        "binary_sha256": digest(cls.bin.read_bytes()),
                        "test_binary_sha256": digest(cls.testbin.read_bytes()),
                        "checkpoint_bytes": CKPT_BYTES, "trainable_parameters": 64,
                        "trainable_parameter_bytes": 256,
                        "native_base_training_log": prefix.decode("ascii")})

    @classmethod
    def tearDownClass(cls):
        if digest(cls.base.read_bytes()) != cls.base_hash:
            raise AssertionError("native base file changed")
        if digest(cls.source.read_bytes()) != cls.source_hash:
            raise AssertionError("source snapshot changed")
        if cls.initial.read_bytes() != cls.initial_bytes:
            raise AssertionError("previous adapter changed")
        METRICS["base_and_source_and_previous_adapter_unchanged"] = True
        cls.tmp.cleanup()

    def name(self, suffix):
        type(self).serial += 1
        return self.work / ("case-%04d-%s" % (self.serial, suffix))

    def vector(self, x):
        path = self.name("x.f32")
        path.write_bytes(pack(x))
        return path

    def records(self, records):
        path = self.name("records.f32")
        path.write_bytes(b"".join(pack(x+t) for x, t in records))
        return path

    def predict(self, adapter, x, base=None):
        return run([self.bin, "predict", base or self.base, adapter, self.vector(x)])

    def update(self, adapter, new, replay, epochs=128, output=None):
        output = output or self.name("candidate.lora")
        p = run([self.bin, "update", self.base, adapter, self.records(new),
                 self.records(replay), output, str(epochs)])
        return p, output

    def with_params(self, p):
        raw = bytearray(self.initial_bytes)
        raw[PARAM_OFFSET:CHECKSUM_OFFSET] = pack(p)
        path = self.name("custom.lora")
        path.write_bytes(reseal(raw))
        return path

    def test_01_static_elf_and_frozen_projection(self):
        p = run(["readelf", "-l", "-d", self.bin])
        self.assertEqual(p.returncode, 0)
        self.assertNotIn(b"INTERP", p.stdout)
        self.assertNotIn(b"NEEDED", p.stdout)
        self.assertEqual(len(self.initial_bytes), CKPT_BYTES)
        self.assertEqual(self.initial_bytes[64:PARAM_OFFSET], self.base_bytes)
        self.assertEqual(self.initial_bytes[:8], b"NLORA001")
        # Use an actual embedding row from the private Assembly-trained model.
        x = floats(self.base_bytes[ord('n')*64:ord('n')*64+64])
        self.assertTrue(all(abs(z) <= 4 for z in x))
        p = self.predict(self.initial, x)
        self.assertEqual(p.returncode, 0)
        self.assertEqual(len(p.stdout), 64)
        expected = ref_forward(self.w, self.initial_params, x)
        self.assertLess(max(abs(a-b) for a, b in zip(floats(p.stdout), expected)), 3e-6)

    def test_02_independent_gradient_and_finite_difference(self):
        pars = [f32(((i*7)%17-8)/64) for i in range(64)]
        adapter = self.with_params(pars)
        x = [f32(((i*3)%11-5)/16) for i in range(16)]
        expected_y = ref_forward(self.w, pars, x)
        target = [f32(y+((i%3)-1)*0.07) for i, y in enumerate(expected_y)]
        record = self.records([(x, target)])
        p = run([self.testbin, "gradient", self.base, adapter, record])
        self.assertEqual(p.returncode, 0)
        self.assertEqual(len(p.stdout), 512)
        actual_g = floats(p.stdout[:256])
        actual_p = floats(p.stdout[256:])
        analytic = ref_gradient(self.w, pars, x, target)
        expected_g = [max(-1, min(1, g)) for g in analytic]
        self.assertLess(max(abs(a-b) for a, b in zip(actual_g, expected_g)), 3e-7)
        expected_p = [f32(a-f32(f32(g)*0.03125)) for a, g in zip(pars, actual_g)]
        self.assertEqual(pack(actual_p), pack(expected_p))
        fd_errors = []
        for coord in [0, 7, 16, 31, 32, 43, 63]:
            a, b = pars.copy(), pars.copy()
            a[coord] += 1e-5
            b[coord] -= 1e-5
            fd = (loss(self.w, a, [(x, target)])-loss(self.w, b, [(x, target)]))/2e-5
            fd_errors.append(abs(fd-analytic[coord]))
        self.assertLess(max(fd_errors), 1e-9)
        self.assertTrue(any(actual_p[i] != pars[i] for i in range(32)))
        self.assertTrue(any(actual_p[i] != pars[i] for i in range(32, 64)))
        METRICS["max_native_gradient_absolute_error"] = max(abs(a-b) for a,b in zip(actual_g,analytic))
        METRICS["max_finite_difference_absolute_error"] = max(fd_errors)

    def test_03_gradient_clipping(self):
        pars = [f32(0.2)]*64
        adapter = self.with_params(pars)
        x = [4.0]*16
        target = [-64.0]*16
        p = run([self.testbin,"gradient",self.base,adapter,self.records([(x,target)])])
        self.assertEqual(p.returncode,0)
        gs, ps = floats(p.stdout[:256]), floats(p.stdout[256:])
        self.assertTrue(all(math.isfinite(v) and abs(v)<=1 for v in gs))
        self.assertTrue(any(abs(v)==1 for v in gs))
        self.assertTrue(all(abs(a-b)<=0.03125001 for a,b in zip(ps,pars)))

    def test_04_sequential_tasks_and_persisted_reload(self):
        # Distinct feature directions; explicit prior examples are replayed.
        x1, x2 = [0.0]*16, [0.0]*16
        x1[0], x2[1] = 1.0, 1.0
        zero = [0.0]*16
        t1 = [f32(z) for z in ref_forward(self.w,self.initial_params,x1)]
        t2 = [f32(z) for z in ref_forward(self.w,self.initial_params,x2)]
        t1[0] = f32(t1[0]+0.30)
        t2[1] = f32(t2[1]-0.25)
        first = [(x1,t1)]
        second = [(x2,t2)]
        p1,a1 = self.update(self.initial, first, [(zero,zero)], epochs=256)
        self.assertEqual(p1.returncode,0,p1.stderr)
        old_bytes = a1.read_bytes()
        p2,a2 = self.update(a1, second, first, epochs=256)
        self.assertEqual(p2.returncode,0,p2.stderr)
        self.assertEqual(a1.read_bytes(),old_bytes)
        st1,st2 = floats(p1.stdout),floats(p2.stdout)
        for st in [st1,st2]:
            self.assertLess(st[1]+1e-7,st[0])
            self.assertLessEqual(st[3],st[2]+0.005001)
        p1par = floats(a1.read_bytes()[PARAM_OFFSET:CHECKSUM_OFFSET])
        p2par = floats(a2.read_bytes()[PARAM_OFFSET:CHECKSUM_OFFSET])
        for st,pbefore,pafter,new,rep in [(st1,self.initial_params,p1par,first,[(zero,zero)]),
                                        (st2,p1par,p2par,second,first)]:
            refstats = [loss(self.w,pbefore,new),loss(self.w,pafter,new),
                        loss(self.w,pbefore,rep),loss(self.w,pafter,rep)]
            self.assertLess(max(abs(a-b) for a,b in zip(st,refstats)),2e-6)
        copy = self.name("reload.lora")
        copy.write_bytes(a2.read_bytes())
        for x in [x1,x2,zero,[f32((i-8)/32) for i in range(16)]]:
            p,q = self.predict(a2,x),self.predict(copy,x)
            self.assertEqual((p.returncode,q.returncode),(0,0))
            self.assertEqual(p.stdout,q.stdout)
        self.assertEqual(struct.unpack_from("<QQ",a2.read_bytes(),32),(2,1024))
        METRICS["sequential_task_gate_metrics"] = {"first":st1,"second":st2}
        METRICS["persisted_prediction_byte_equality_cases"] = 4

    def test_05_conflicting_task_rejected_without_mutation(self):
        x = [0.0]*16
        x[0] = 1.0
        old = [f32(z) for z in ref_forward(self.w,self.initial_params,x)]
        new = old.copy()
        new[0] = f32(new[0]+1.0)
        before = self.initial.read_bytes()
        p,out = self.update(self.initial,[(x,new)],[(x,old)],epochs=256)
        self.assertEqual(p.returncode,10,p.stderr)
        self.assertFalse(out.exists())
        self.assertEqual(self.initial.read_bytes(),before)

    def test_06_no_improvement_rejected(self):
        z = [0.0]*16
        p,out = self.update(self.initial,[(z,z)],[(z,z)],epochs=1)
        self.assertEqual(p.returncode,10)
        self.assertFalse(out.exists())

    def test_07_checkpoint_format_and_corruption(self):
        bads = [self.initial_bytes[:-1], self.initial_bytes+b"x",b"",
                bytes([self.initial_bytes[0]^1])+self.initial_bytes[1:]]
        bad = bytearray(self.initial_bytes)
        bad[PARAM_OFFSET+19] ^= 1
        bads.append(bytes(bad))
        for offset,value in [(8,2),(12,17),(16,3),(20,BASE_BYTES-4),(24,65),(28,1),(48,1)]:
            bad = bytearray(self.initial_bytes)
            struct.pack_into("<I",bad,offset,value)
            bads.append(reseal(bad))
        for bad in bads:
            path = self.name("malformed.lora")
            path.write_bytes(bad)
            p = self.predict(path,[0.0]*16)
            self.assertEqual(p.returncode,2)
            self.assertEqual(p.stdout,b"")
        METRICS["checkpoint_negative_vectors"] = len(bads)

    def test_08_full_base_binding_not_only_projection(self):
        altered = bytearray(self.base_bytes)
        struct.pack_into("<f",altered,0,0.123)
        self.assertEqual(altered[WQ_OFFSET:WQ_OFFSET+1024],self.base_bytes[WQ_OFFSET:WQ_OFFSET+1024])
        pbase = self.name("other-base.f32")
        pbase.write_bytes(altered)
        p = self.predict(self.initial,[0.0]*16,base=pbase)
        self.assertEqual(p.returncode,2)
        self.assertEqual(p.stdout,b"")

    def test_09_finite_and_range_checks(self):
        for value in [float("nan"),float("inf"),-float("inf"),8.01]:
            pars = self.initial_params.copy()
            pars[33] = value
            p = self.predict(self.with_params(pars),[0.0]*16)
            self.assertEqual(p.returncode,2)
        for value in [float("nan"),float("inf"),-float("inf"),4.01]:
            x = [0.0]*16
            x[5] = value
            p = self.predict(self.initial,x)
            self.assertEqual(p.returncode,2)
        for value in [float("nan"),float("inf"),65.0]:
            x,t = [0.0]*16,[0.0]*16
            t[3]=value
            p,out = self.update(self.initial,[(x,t)],[(x,x)],epochs=1)
            self.assertEqual(p.returncode,2)
            self.assertFalse(out.exists())

    def test_10_dataset_lengths_and_epoch_bounds(self):
        replay = self.records([([0.0]*16,[0.0]*16)])
        for data in [b"",b"x",bytes(127),bytes(129),bytes(17*128)]:
            path = self.name("invalid-data.f32")
            path.write_bytes(data)
            out = self.name("absent.lora")
            p = run([self.bin,"update",self.base,self.initial,path,replay,out,"1"])
            self.assertEqual(p.returncode,2)
            self.assertFalse(out.exists())
        for epoch in ["", "0", "-1", "257", "1x", "99999999"]:
            out = self.name("absent.lora")
            p = run([self.bin,"update",self.base,self.initial,replay,replay,out,epoch])
            self.assertEqual(p.returncode,2)
            self.assertFalse(out.exists())

    def test_11_no_overwrite_or_symlink_follow(self):
        old = self.initial.read_bytes()
        p = run([self.bin,"init",self.base,self.initial])
        self.assertEqual(p.returncode,3)
        self.assertEqual(self.initial.read_bytes(),old)
        link = self.name("symlink.lora")
        link.symlink_to(self.initial)
        self.assertEqual(self.predict(link,[0.0]*16).returncode,3)
        self.assertEqual(self.initial.read_bytes(),old)

    def test_12_parameter_bound_rolls_back(self):
        pars = [0.0]*64
        pars[:32] = [f32(0.0625)]*32
        pars[32:] = [f32(7.999)]*32
        adapter = self.with_params(pars)
        before = adapter.read_bytes()
        x = [0.0]*16
        x[0] = 1.0
        y = [f32(z+1) for z in ref_forward(self.w,pars,x)]
        z = [0.0]*16
        p,out = self.update(adapter,[(x,y)],[(z,z)],epochs=1)
        self.assertEqual(p.returncode,10)
        self.assertFalse(out.exists())
        self.assertEqual(adapter.read_bytes(),before)

    def test_13_test_hook_absent_from_production(self):
        p = run([self.bin,"gradient",self.base,self.initial,
                 self.records([([0.0]*16,[0.0]*16)])])
        self.assertEqual(p.returncode,2)


if __name__ == "__main__":
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContinualLoRA)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {"schema":"niyah-native-continual-lora-v1", "passed":result.testsRun-len(result.failures)-len(result.errors),
               "failed":len(result.failures),"errors":len(result.errors),"tests_run":result.testsRun,
               "exit_code":0 if result.wasSuccessful() else 1,"elapsed_seconds":time.perf_counter()-started,
               "runtime":"static x86-64 ELF, raw Linux syscalls; no libc/PyTorch/CUDA",
               "scope":"rank-2 adaptation of frozen trained WQ, synthetic feature tasks, no language-quality claim",
               "metrics":METRICS,"commands":COMMANDS,
               "source_sha256":{p:digest((ROOT/p).read_bytes()) for p in
                                  ["src/continual_lora.s","tests/test_continual_lora.py"]}}
    if os.environ.get("NIYAH_LORA_RECEIPT"):
        Path(os.environ["NIYAH_LORA_RECEIPT"]).write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k not in ["commands","metrics"]}))
    raise SystemExit(receipt["exit_code"])
