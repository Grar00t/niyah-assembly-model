#!/usr/bin/env python3
import argparse,itertools,pathlib,subprocess,tempfile,struct,json,random,hashlib,time,os
P=pathlib.Path(__file__).resolve().parent
CAPS='''.intel_syntax noprefix
.bss
buf:.skip 48
.text
.global _start
_start:
 xor eax,eax
 xor edi,edi
 lea rsi,[rip+buf]
 mov edx,48
 syscall
 cmp eax,48
 jne fail
 mov eax,[rip+buf]
 mov [rip+nx_cpu+24],eax
 mov eax,[rip+buf+4]
 mov [rip+nx_cpu+32],eax
 mov eax,[rip+buf+8]
 mov [rip+nx_cpu+40],eax
 mov rax,[rip+buf+16]
 mov [rip+nx_cpu+16],rax
 mov rax,[rip+buf+24]
 mov [rip+nx_amx_limits],rax
 mov rax,[rip+buf+32]
 mov [rip+nx_amx_limits+8],rax
 mov rax,[rip+buf+40]
 mov [rip+nx_amx_limits+16],rax
 call nx_decode_caps
 mov [rip+buf],eax
 mov eax,1
 mov edi,1
 lea rsi,[rip+buf]
 mov edx,4
 syscall
 xor edi,edi
 jmp done
fail: mov edi,65
done: mov eax,60
 syscall
.section .note.GNU-stack,"",@progbits
'''
PACK='''.intel_syntax noprefix
.bss
input:.skip 1024
output:.skip 1024
.text
.global _start
_start:
 xor eax,eax
 xor edi,edi
 lea rsi,[rip+input]
 mov edx,1024
 syscall
 cmp eax,1024
 jne fail
 lea rdi,[rip+input]
 lea rsi,[rip+output]
 call nx_pack_b
 mov eax,1
 mov edi,1
 lea rsi,[rip+output]
 mov edx,1024
 syscall
 xor edi,edi
 jmp done
fail: mov edi,65
done: mov eax,60
 syscall
.section .note.GNU-stack,"",@progbits
'''
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dispatch',type=pathlib.Path,default=P/'tensor_dispatch.o');ap.add_argument('--runtime',type=pathlib.Path,default=P/'tensor_runtime.o');ap.add_argument('--program',type=pathlib.Path,default=P/'demo.o');ap.add_argument('--evidence',type=pathlib.Path,default=P/'dispatch_receipt.json');a=ap.parse_args();t0=time.perf_counter();observations=[]
 def run(cmd,inp=None,expected=0,stdin=None):
  q=subprocess.run([str(x) for x in cmd],input=inp,stdin=stdin,capture_output=True,timeout=10);assert q.returncode==expected,(cmd,expected,q.returncode,q.stderr);observations.append({'argv':[str(x) for x in cmd],'expected_exit':expected,'exit':q.returncode,'input_sha256':hashlib.sha256(inp).hexdigest() if inp is not None else None,'stdout_sha256':hashlib.sha256(q.stdout).hexdigest()});return q.stdout
 with tempfile.TemporaryDirectory(prefix='niyah-dispatch-') as d:
  p=pathlib.Path(d);(p/'caps.s').write_text(CAPS);(p/'pack.s').write_text(PACK)
  run(['as','--64','-o',p/'caps.o',p/'caps.s']);run(['ld','-static','--build-id=none','-z','noexecstack','-o',p/'caps',p/'caps.o',a.dispatch])
  n=0
  limits=[0x1e,1,(1024<<16)|8192,(8<<16)|64,16,(64<<8)|16]
  def snapshot(ecx,ebx,edx,xcr,geometry):
   return struct.pack('<4IQ6I',ecx,ebx,edx,0,xcr,*geometry)
  for xs,osx,avx,v2,v5,tile,i8 in itertools.product((0,1),repeat=7):
   for xcr in (0,2,4,6,7,0xe6,0x60000,0x60007,0x600e7):
    ecx=(xs<<26)|(osx<<27)|(avx<<28);ebx=(v2<<5)|(v5<<16);edx=(tile<<24)|(i8<<25)
    want=(1 if xs and osx and avx and v2 and xcr&6==6 else 0)|(2 if xs and osx and avx and v5 and xcr&0xe6==0xe6 else 0)|(4 if xs and osx and tile and i8 and xcr&0x60000==0x60000 else 0)
    got=struct.unpack('<I',run([p/'caps'],snapshot(ecx,ebx,edx,xcr,limits)))[0];assert got==want,(ecx,ebx,edx,xcr,got,want);n+=1
  # Exercise the actual pure decoder via capability selection. No mocked
  # snapshot can request permission or enter a vector/AMX kernel.
  geometry_cases=[(limits,True)]
  for field,value in [(0,0x1d),(1,0),(2,(1024<<16)|8191),(2,(1023<<16)|8192),(3,(8<<16)|63),(3,(7<<16)|64),(4,15),(5,(64<<8)|15),(5,(63<<8)|16)]:
   bad=limits.copy();bad[field]=value;geometry_cases.append((bad,False))
  geometry_cases.append(([0x30,2,(2048<<16)|16384,(16<<16)|128,32,(128<<8)|32],True))
  geometry_count=0
  for geometry,valid in geometry_cases:
   for vector in (False,True):
    ecx=(1<<26)|(1<<27)|((1<<28) if vector else 0)
    ebx=((1<<5)|(1<<16)) if vector else 0
    xcr=0x60000|(0xe6 if vector else 0)
    want=(3 if vector else 0)|(4 if valid else 0)
    got=struct.unpack('<I',run([p/'caps'],snapshot(ecx,ebx,0x03000000,xcr,geometry)))[0]
    assert got==want,(geometry,vector,got,want);geometry_count+=1
  # The probe renames only the test object's entry. Production entry/objects remain unchanged.
  run(['objcopy','--redefine-sym','_start=nx_unused_runtime_entry',a.runtime,p/'runtime_test.o'])
  run(['as','--64','-o',p/'pack.o',p/'pack.s']);run(['ld','-static','--build-id=none','-z','noexecstack','-o',p/'pack',p/'pack.o',p/'runtime_test.o',a.dispatch,a.program])
  rnd=random.Random(270127);cases=[bytes([0]*1024),bytes([255]*1024),bytes(range(256))*4]+[rnd.randbytes(1024) for _ in range(61)]
  for raw in cases:
   want=bytes(raw[(4*g+l)*16+j] for g in range(16) for j in range(16) for l in range(4));got=run([p/'pack'],raw);assert got==want
  # A directory fd reaches the runtime's actual raw read error path.
  run(['ld','-static','--build-id=none','-z','noexecstack','-o',p/'program',a.program,a.runtime,a.dispatch])
  fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
  try:assert run([p/'program','scalar'],expected=66,stdin=fd)==b''
  finally:os.close(fd)
  result={'status':'PASS','exit':0,'cpuid_xstate_combinations':n,'palette_limit_combinations':geometry_count,'native_B_pack_cases':len(cases),'raw_read_error_cases':1,'all_expected_exits_matched':True,'seconds':time.perf_counter()-t0,'amx_instructions_executed':False,'avx512_instructions_executed':False,'scope':'snapshot decoding and native B relayout; not AMX/AVX512 hardware validation','commands':observations}
  a.evidence.parent.mkdir(exist_ok=True,parents=True);a.evidence.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='commands'},indent=2))
if __name__=='__main__':main()
