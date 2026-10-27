#!/usr/bin/env python3
"""Fraction-oracle tests for the native arithmetic candidate auditor."""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = []
METRICS = {}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def command(argv, **kwargs):
    start = time.perf_counter()
    opts = {"stdout":subprocess.PIPE,"stderr":subprocess.PIPE,"timeout":10}
    opts.update(kwargs)
    p = subprocess.run([str(a) for a in argv],**opts)
    COMMANDS.append({"argv":[str(a) for a in argv],"exit_code":p.returncode,
                     "seconds":time.perf_counter()-start,
                     "stdout_sha256":sha(p.stdout) if p.stdout is not None else None,
                     "stderr":(p.stderr or b"").decode("utf-8","replace")[:200]})
    return p


class NativeAuditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="niyah-audit-")
        cls.work = Path(cls.tmp.name)
        cls.pair = cls.work/"directory with spaces"
        cls.pair.mkdir()
        cls.bin = cls.pair/"nexus_audit"
        cls.calc = cls.pair/"exactcalc"
        cls.calc_source_hash = sha((ROOT/"src/exactcalc.s").read_bytes())
        for exe,source in [(cls.bin,"nexus_audit.s"),(cls.calc,"exactcalc.s")]:
            p = command(["cc","-nostdlib","-static","-no-pie","-Wl,--build-id=none",
                         "-o",exe,ROOT/"src"/source])
            if p.returncode:
                raise RuntimeError(p.stderr.decode())
        METRICS.update({"binary_sha256":sha(cls.bin.read_bytes()),
                        "tested_exactcalc_sha256":sha(cls.calc.read_bytes()),
                        "exactcalc_source_sha256":cls.calc_source_hash})

    @classmethod
    def tearDownClass(cls):
        if sha((ROOT/"src/exactcalc.s").read_bytes())!=cls.calc_source_hash:
            raise AssertionError("exactcalc source modified")
        METRICS["exactcalc_source_unchanged"] = True
        cls.tmp.cleanup()

    def call(self,op,n1,d1,n2,d2,candidate,**kwargs):
        return command([self.bin,op,n1,d1,n2,d2,candidate],**kwargs)

    def test_01_fraction_oracle(self):
        rng = random.Random(20261027)
        cases = [(0,1,0,1),(-2147483648,2147483647,2147483647,2147483647),
                 (2147483647,1,-2147483648,1),(1,10,2,10),(5,7,-2,9)]
        cases += [(rng.randint(-10000,10000),rng.randint(1,10000),
                   rng.randint(-10000,10000),rng.randint(1,10000)) for _ in range(15)]
        count = 0
        for a,b,c,d in cases:
            x,y = Fraction(a,b),Fraction(c,d)
            for op in ["add","sub","mul","div"]:
                if op=="div" and not c:
                    continue
                expected = {"add":lambda:x+y,"sub":lambda:x-y,
                            "mul":lambda:x*y,"div":lambda:x/y}[op]()
                candidate = f"{expected.numerator}/{expected.denominator}"
                p = self.call(op,a,b,c,d,candidate)
                self.assertEqual(p.returncode,0,(op,a,b,c,d,p.stderr))
                self.assertEqual(p.stdout,b"VERIFIED "+candidate.encode()+b"\n")
                count += 1
        METRICS["fraction_oracle_cases"] = count

    def test_02_wrong_and_noncanonical_candidates(self):
        for candidate in ["4/10","0.3","6/20","The answer is 3/10.","3/10\n", "", "3/10 extra",
                          "3/10; echo compromised", "٠٫٣", "x"*4096]:
            p = self.call("add",1,10,2,10,candidate)
            self.assertEqual(p.returncode,1)
            self.assertEqual(p.stdout,b"MISMATCH\n")
        METRICS["mismatch_vectors"] = 10

    def test_03_calculator_errors_and_usage(self):
        invalids = [("div",1,2,0,1),("mul",1,0,2,3),("add",2147483648,1,0,1),
                    ("add",1,-2,2,1),("bad",1,1,1,1),("add","+1",1,2,3),
                    ("add","1;echo nope",1,2,3)]
        for args in invalids:
            p = command([self.bin,*args,"0/1"])
            self.assertEqual(p.returncode,3)
            self.assertNotIn(b"VERIFIED",p.stdout)
        p = command([self.bin,"add",1,2,3,4])
        self.assertEqual(p.returncode,2)

    def test_04_no_path_lookup_and_missing_sibling(self):
        p = self.call("add",1,10,2,10,"3/10",env={"PATH":"/does-not-exist"},cwd="/")
        self.assertEqual(p.returncode,0)
        missing = self.work/"missing-sibling"
        missing.mkdir()
        shutil.copyfile(self.bin,missing/"nexus_audit")
        (missing/"nexus_audit").chmod(0o700)
        p = command([missing/"nexus_audit","add",1,10,2,10,"3/10"],env={"PATH":str(self.pair)})
        self.assertEqual(p.returncode,3)
        self.assertEqual(p.stdout,b"")

    def test_05_malformed_child_protocol(self):
        for i,payload in enumerate([b"",b"not a fraction\n",b"3/10",b"01/1\n",b"-0/1\n",b"1/0\n",b"x"*150]):
            d = self.work/f"protocol-{i}"
            d.mkdir()
            shutil.copyfile(self.bin,d/"nexus_audit")
            (d/"nexus_audit").chmod(0o700)
            # Disposable native test child, never replacing the real calculator.
            data = ",".join(str(b) for b in payload) or "0"
            source = ".intel_syntax noprefix\n.section .rodata\ndata: .byte "+data+"\n.section .text\n.global _start\n_start:\nmov eax,1\nmov edi,1\nlea rsi,[rip+data]\nmov edx,"+str(len(payload))+"\nsyscall\nmov eax,60\nxor edi,edi\nsyscall\n.section .note.GNU-stack,\"\",@progbits\n"
            (d/"fake.s").write_text(source)
            p = command(["cc","-nostdlib","-static","-no-pie","-o",d/"exactcalc",d/"fake.s"])
            self.assertEqual(p.returncode,0)
            p = command([d/"nexus_audit","add",1,10,2,10,"3/10"])
            self.assertEqual(p.returncode,3)
            self.assertEqual(p.stdout,b"")

    def test_06_stdout_failures_are_exit4(self):
        with open("/dev/full","wb") as f:
            p = self.call("add",1,10,2,10,"3/10",stdout=f)
        self.assertEqual(p.returncode,4)
        r,w = os.pipe()
        os.close(r)
        try:
            p = self.call("add",1,10,2,10,"3/10",stdout=w)
        finally:
            os.close(w)
        self.assertEqual(p.returncode,4)

    def test_07_elf_dependency_contract(self):
        p = command(["readelf","-l","-d",self.bin])
        self.assertEqual(p.returncode,0)
        self.assertNotIn(b"INTERP",p.stdout)
        self.assertNotIn(b"NEEDED",p.stdout)


if __name__=="__main__":
    started=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeAuditor))
    receipt={"schema":"niyah-native-arithmetic-auditor-v1","tests_run":result.testsRun,
             "passed":result.testsRun-len(result.failures)-len(result.errors),
             "failed":len(result.failures),"errors":len(result.errors),
             "exit_code":0 if result.wasSuccessful() else 1,"elapsed_seconds":time.perf_counter()-started,
             "scope":"canonical bounded rational arithmetic candidates only; no general neural truth/retention claim",
             "metrics":METRICS,"commands":COMMANDS,
             "source_sha256":{n:sha((ROOT/n).read_bytes()) for n in ["src/nexus_audit.s","tests/test_nexus_audit.py"]}}
    if os.environ.get("NIYAH_AUDITOR_RECEIPT"):
        Path(os.environ["NIYAH_AUDITOR_RECEIPT"]).write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k not in ["metrics","commands"]}))
    raise SystemExit(receipt["exit_code"])
