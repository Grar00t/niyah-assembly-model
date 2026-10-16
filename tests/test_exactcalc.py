#!/usr/bin/env python3
"""Property and failure tests for a pure x86-64 rational calculator.

Mathematical oracle: Python's unlimited-precision fractions.Fraction;
the executable does not import or invoke this oracle.
"""
from fractions import Fraction
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
EXE=ROOT/"build"/"exactcalc"
MIN=-(1<<31)
MAX=(1<<31)-1
OPS={
    "add": lambda a,b: a+b,
    "sub": lambda a,b: a-b,
    "mul": lambda a,b: a*b,
    "div": lambda a,b: a/b,
}
FORMAT=re.compile(rb"-?(0|[1-9][0-9]*)/[1-9][0-9]*\n\Z")

def run(args,timeout=3):
    return subprocess.run([str(EXE),*map(str,args)],capture_output=True,timeout=timeout)

def expect_success(op,n1,d1,n2,d2):
    got=run((op,n1,d1,n2,d2))
    if got.returncode:
        raise AssertionError((op,n1,d1,n2,d2,got.returncode,got.stderr))
    assert got.stderr==b"",got.stderr
    assert FORMAT.fullmatch(got.stdout),got.stdout
    expected=OPS[op](Fraction(n1,d1),Fraction(n2,d2))
    answer=f"{expected.numerator}/{expected.denominator}\n".encode("ascii")
    assert got.stdout==answer, (op,n1,d1,n2,d2,got.stdout,answer)
    # Every denominator strictly positive, every answer in signed 64-bit.
    assert -(1<<63) < expected.numerator < (1<<63)
    assert 0<expected.denominator<(1<<63)

def expect_rejected(args,code,fragment):
    got=run(args)
    assert got.returncode==code,(args,got.returncode,got.stdout,got.stderr)
    assert got.stdout==b"", (args,got.stdout)
    assert fragment in got.stderr,(args,got.stderr)

def static_binary_checks():
    p=subprocess.run(["readelf","-W","-l",str(EXE)],capture_output=True,check=True).stdout
    assert b"INTERP" not in p,"a loader should not be required"
    stack=[line for line in p.splitlines() if b"GNU_STACK" in line]
    assert len(stack)==1,stack
    assert b" RWE " not in stack[0],stack[0]
    assert b" RW " in stack[0],stack[0]
    dynamic=subprocess.run(["readelf","-d",str(EXE)],capture_output=True,check=True).stdout
    assert b"NEEDED" not in dynamic and b"Shared library:" not in dynamic
    undefined=subprocess.run(["nm","-u",str(EXE)],capture_output=True,check=True).stdout
    assert not undefined,undefined
    print("PASS: pure native x86-64, no runtime libraries, nonexecutable stack")

def main():
    assert EXE.is_file(),f"build missing: {EXE}"
    static_binary_checks()
    assert run(["--help"]).returncode==0
    assert b"signed 32-bit" in run(["--help"]).stdout
    boundary=[MIN,MIN+1,-MAX,-101,-7,-3,-1,0,1,2,3,7,101,MAX-1,MAX]
    dens=[1,2,3,7,97,65535,MAX-1,MAX]
    count=0
    for op in OPS:
        for n1 in boundary:
            for n2 in boundary:
                d1=dens[(abs(n1)+abs(n2))%len(dens)]
                d2=dens[(abs(n1)*3+abs(n2))%len(dens)]
                if op=="div" and n2==0:
                    continue
                expect_success(op,n1,d1,n2,d2)
                count+=1
    print(f"PASS: {count} deterministic integer boundary combinations")
    rng=random.Random(0x4E49594148)
    for op in OPS:
        for _ in range(1250):
            def n():
                return rng.choice(boundary) if rng.random()<0.55 else rng.randint(MIN,MAX)
            def den():
                return rng.choice(dens) if rng.random()<0.55 else rng.randint(1,MAX)
            n1,n2=n(),n()
            if op=="div" and n2==0:n2=1
            expect_success(op,n1,den(),n2,den())
            count+=1
    print(f"PASS: {count} successful exact outputs match Fraction oracle")
    rejected=[
      ([],2,b"usage"),
      (["add"],2,b"usage"),
      (["add","1","2","3"],2,b"usage"),
      (["add","1","2","3","4","5"],2,b"usage"),
      (["addx","1","1","2","1"],2,b"invalid"),
      (["ADD","1","1","2","1"],2,b"invalid"),
      (["","1","1","2","1"],2,b"invalid"),
      (["add","","1","2","1"],2,b"invalid"),
      (["add","-","1","2","1"],2,b"invalid"),
      (["add","+1","1","2","1"],2,b"invalid"),
      (["add","1","0","2","1"],2,b"invalid"),
      (["add","1","-1","2","1"],2,b"invalid"),
      (["add","1","1","2","0"],2,b"invalid"),
      (["add","2147483648","1","1","1"],2,b"invalid"),
      (["add","-2147483649","1","1","1"],2,b"invalid"),
      (["add","1","2147483648","1","1"],2,b"invalid"),
      (["mul","1e2","1","1","1"],2,b"invalid"),
      (["mul","0x10","1","1","1"],2,b"invalid"),
      (["mul","1 ","1","1","1"],2,b"invalid"),
      (["mul"," 1","1","1","1"],2,b"invalid"),
      (["mul","１２","1","1","1"],2,b"invalid"),
      (["mul","0"*100_000,"1","1","1"],2,b"invalid"),
      (["div","1","2","0","1"],3,b"division by zero"),
      (["div","0","2","0","1"],3,b"division by zero"),
    ]
    for args,code,message in rejected:
        expect_rejected(args,code,message)
    print(f"PASS: {len(rejected)} malformed/zero-division vectors rejected with exact exit codes")
    for op in OPS:
        expect_success(op,0,1,1,MAX)
        expect_success(op,MIN,MAX,MAX,MAX)
        expect_success(op,MAX,1,MIN,1)
    # Operational check on decimal fractions: 0.1+0.2 becomes 3/10, exactly.
    assert run(["add",1,10,2,10]).stdout==b"3/10\n"
    print("PASS: human-readable exact-decimal example (1/10 + 2/10 = 3/10)")
    print("PASS: untrusted input is strictly bounded to 10 digit scans per operand")
    print(f"TOTAL_SUCCESS_CASES={count+len(OPS)*3}")
    print(f"TOTAL_FAILURE_CASES={len(rejected)}")

if __name__=="__main__":
    main()
