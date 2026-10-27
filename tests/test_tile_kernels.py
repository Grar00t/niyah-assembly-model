#!/usr/bin/env python3
"""Independent native GEMM/dispatch and tiled attention regression. Stdlib harness only."""
import hashlib, json, math, os, pathlib, random, struct, subprocess, tempfile, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
records=[]
def run(cmd, **kw):
    t=time.perf_counter(); q=subprocess.run(cmd,cwd=ROOT,capture_output=True,timeout=90,**kw)
    records.append({"command":[str(v) for v in cmd],"exit_code":q.returncode,"seconds":time.perf_counter()-t})
    return q
def check_elf(exe):
    a=run(["readelf","-l",str(exe)]); b=run(["readelf","-d",str(exe)]); c=run(["nm","-u",str(exe)])
    assert a.returncode==b.returncode==c.returncode==0
    assert b"INTERP" not in a.stdout and b"NEEDED" not in b.stdout and not c.stdout
def main():
    (ROOT/"build").mkdir(exist_ok=True)
    exe=ROOT/"build/tile_probe"
    q=run(["cc","-nostdlib","-static","-no-pie","-Wl,-z,noexecstack","-o",str(exe),"src/tile_probe.s","src/tile_kernels.s"])
    assert q.returncode==0,q.stderr.decode()
    check_elf(exe)
    f=run([str(exe),"features"]); assert f.returncode==0 and len(f.stdout)==64
    fields=struct.unpack("<8Q",f.stdout); assert f.stdout[:8]==b"NYCPUID1"
    caps=fields[1]
    features={"cpuid1_ecx":fields[2],"cpuid7_ebx":fields[3],"cpuid7_edx":fields[4],"xcr0":fields[5],"max_leaf":fields[6],
              "avx2_fma_os":bool(caps&1),"avx512f_fma_os":bool(caps&2),"amx_int8_xcr0":bool(caps&4)}
    paths=["scalar","auto"]+([ "avx2"] if caps&1 else [])+([ "avx512"] if caps&2 else [])
    rnd=random.Random(20261027)
    eye=[float(i==j) for i in range(16) for j in range(16)]
    fixtures=[([0.0]*256,eye),(eye,eye),([float(i%19-9) for i in range(256)],eye)]
    fixtures += [([float(rnd.randrange(-8,9)) for _ in range(256)],[float(rnd.randrange(-8,9)) for _ in range(256)]) for _ in range(24)]
    fixtures += [([rnd.uniform(-2,2) for _ in range(256)],[rnd.uniform(-2,2) for _ in range(256)]) for _ in range(8)]
    max_abs=0.0
    for n,(a,b) in enumerate(fixtures):
        payload=struct.pack("<512f",*(a+b)); quant=struct.unpack("<512f",payload); a=quant[:256]; b=quant[256:]
        expected=[sum(a[i*16+k]*b[k*16+j] for k in range(16)) for i in range(16) for j in range(16)]
        for mode in paths:
            q=run([str(exe),mode],input=payload); assert q.returncode==0 and len(q.stdout)==1024 and not q.stderr,(mode,q.returncode)
            got=struct.unpack("<256f",q.stdout)
            for actual,want in zip(got,expected):
                err=abs(actual-want); max_abs=max(max_abs,err)
                assert math.isfinite(actual) and err<=2e-5*(1+abs(want)),(n,mode,actual,want,err)
            if n<27: assert tuple(expected)==got,("integer oracle",n,mode)
    skips={}
    for mode,bit in [("avx2",1),("avx512",2),("amx-int8",4)]:
        if not caps&bit:
            q=run([str(exe),mode],input=b""); assert q.returncode==77 and not q.stdout,(mode,q.returncode)
            skips[mode]="observed native exit77 before unsupported instructions"
    amx={"hardware_executed":False}
    if caps&4:
        ai=[rnd.randrange(-128,128) for _ in range(1024)]; bi=[rnd.randrange(-128,128) for _ in range(1024)]
        q=run([str(exe),"amx-int8"],input=struct.pack("<2048b",*(ai+bi)))
        if q.returncode==77: skips["amx-int8"]="OS permission refused, exit77"
        else:
            assert q.returncode==0 and len(q.stdout)==1024
            expected=[sum(ai[i*64+k]*bi[k*16+j] for k in range(64)) for i in range(16) for j in range(16)]
            assert list(struct.unpack("<256i",q.stdout))==expected
            amx={"hardware_executed":True,"oracle":"signed INT8 exact integer"}
    payload=struct.pack("<512f",*(eye+eye))
    bad_cases=[([],b"",64),(["unknown"],b"",64),(["scalar","0"],b"",64),(["scalar","-1"],b"",64),
               (["scalar","10000001"],b"",64),(["scalar",""],b"",64),(["features","2"],b"",64),
               (["scalar"],payload[:-1],65),(["scalar"],payload+b"x",65),
               (["scalar"],struct.pack("<f",math.nan)+payload[4:],65),(["scalar"],struct.pack("<f",math.inf)+payload[4:],65)]
    for args,data,code in bad_cases:
        q=run([str(exe)]+args,input=data); assert q.returncode==code and not q.stdout,(args,q.returncode)
    q=run([str(exe),"auto","3"],input=payload); assert q.returncode==0 and q.stdout==struct.pack("<256f",*eye)
    with open("/dev/full","wb") as bad:
        q=subprocess.run([str(exe),"auto"],input=payload,stdout=bad,stderr=subprocess.PIPE,cwd=ROOT)
        assert q.returncode==74
        records.append({"command":[str(exe),"auto","stdout=/dev/full"],"exit_code":q.returncode})
    # Same floating-point reduction order must preserve all 9,616 trained weights.
    variants={}
    with tempfile.TemporaryDirectory(prefix="niyah-tiled-") as d:
        for name,defs in [("scalar",[]),("scalar_tiled",["TILED_PROJECTIONS"]),("auto",["AVX2_AUTO"]),
                          ("auto_tiled",["AVX2_AUTO","TILED_PROJECTIONS"]),("fallback_tiled",["AVX2_AUTO","FORCE_SCALAR","TILED_PROJECTIONS"])]:
            dst=pathlib.Path(d)/name
            cmd=["cc","-nostdlib","-static","-no-pie","-Wa,--defsym,EPOCHS=240","-Wa,--defsym,SAMPLE_LEN=0","-Wa,--defsym,EXPORT_WEIGHTS=1"]
            cmd += ["-Wa,--defsym,"+v+"=1" for v in defs]+["-o",str(dst),"src/attention_lm.s"]
            q=run(cmd); assert q.returncode==0,q.stderr.decode()
            q=run([str(dst)]); assert q.returncode==0 and not q.stderr
            variants[name]=q.stdout
        assert variants["scalar"]==variants["scalar_tiled"],"scalar tiling changes output"
        assert variants["auto"]==variants["auto_tiled"],"SIMD tiling changes output"
        # fallback contains one dispatch diagnostic line absent in scalar build.
        assert variants["fallback_tiled"].replace(b"kernel_avx2 0\n",b"",1)==variants["scalar"]
    report={"status":"PASS","features":features,"gemm_fixture_count":len(fixtures),"executed_fp32_paths":paths,
            "gemm_max_abs_error":max_abs,"expected_negative_vectors":len(bad_cases)+1,"unsupported":skips,"amx":amx,
            "tiled_attention":{"epochs":240,"byte_identical_scalar_vs_tiled":True,"byte_identical_auto_vs_tiled":True,"forced_scalar_verified":True},
            "native_binary_sha256":hashlib.sha256(exe.read_bytes()).hexdigest(),"native_binary_bytes":exe.stat().st_size,
            "static_elf_no_interp_no_needed_no_undefined":True,"commands":records}
    out=ROOT/"evidence/nexus-kernels.json";out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2)+"\n")
    compact={k:v for k,v in report.items() if k!="commands"};print(json.dumps(compact,indent=2))
if __name__=="__main__":main()
