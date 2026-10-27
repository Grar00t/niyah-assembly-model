#!/usr/bin/env python3
"""Independent native bridge regression; Python is an oracle, never a runtime dependency."""
import argparse,hashlib,json,os,pathlib,random,re,struct,subprocess,tempfile,time
P=pathlib.Path(__file__).resolve().parent
records=[];counts={};sha=lambda b:hashlib.sha256(b).hexdigest()
def run(args,inp=None,expected=0,group='commands',cwd=None):
 t=time.perf_counter();q=subprocess.run([str(a) for a in args],input=inp,capture_output=True,cwd=cwd or P,timeout=20)
 rec={'argv':[str(a) for a in args],'expected_exit':expected,'exit':q.returncode,'seconds':time.perf_counter()-t,'stdout_bytes':len(q.stdout),'stdout_sha256':sha(q.stdout),'stderr':q.stderr.decode(errors='replace')[:500]};records.append(rec)
 assert q.returncode==expected,rec
 if expected:assert not q.stdout,rec
 counts[group]=counts.get(group,0)+1
 return q.stdout

def elf(path):
 a=run(['readelf','-lW',path]);b=run(['readelf','-dW',path]);c=run(['nm','-u',path])
 assert b'INTERP' not in a and b'NEEDED' not in b and not c
 assert not any(' RWE ' in x or ' RWX ' in x for x in a.decode().splitlines())
 return {'bytes':path.stat().st_size,'sha256':sha(path.read_bytes()),'static_no_interp_needed_undefined':True}

def unpack_ir(b):
 assert b[:8]==b'NIRG0001';nv,no,ni,ne,mask,sl=struct.unpack_from('<6I',b,8)
 values=[]
 for i in range(nv):
  at=32+1024*i;kind,start,end,n,line,col,na,nr=struct.unpack_from('<8I',b,at)
  name=b[at+32:at+32+n].decode('utf8');roots=[struct.unpack_from('<3I',b,at+160+12*j) for j in range(nr)]
  values.append({'id':i,'type':kind,'start':start,'end':end,'name':name,'line':line,'col':col,'morph_candidate_count':na,'roots':roots})
 ops=[struct.unpack_from('<4I',b,32+1024*nv+16*i) for i in range(no)]
 source=b[32+1024*nv+16*no:];assert len(source)==sl
 assert len(b)==32+1024*nv+16*no+sl
 # Independent def/use and type oracle.
 ends={};definitions={};live=0;outputs=0;inputs=0
 for i,(op,d,a,bb) in enumerate(ops):
  if op in (1,2,3):
   assert d==len(definitions) and values[d]['type']==op;definitions[d]=i;ends[d]=i+1
  if op in (1,2):inputs+=1
  if op==3:
   assert a<d and bb<d and values[a]['type']==1 and values[bb]['type']==2
   ends[a]=ends[bb]=i+1
  if op==4:
   assert d in ends and values[d]['type']==3;ends[d]=i+1;outputs+=1
 for v in values:assert (v['start'],v['end'])==(definitions[v['id']],ends[v['id']])
 assert (inputs,outputs)==(ni,ne)
 return {'values':values,'ops':ops,'mask':mask,'source':source,'ni':ni,'ne':ne}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--compiler',type=pathlib.Path,default=P/'niyahc-tile');ap.add_argument('--allocator',type=pathlib.Path,default=pathlib.Path('/home/a/niyah-assembly-model/build/tile_allocator'));ap.add_argument('--runtime',type=pathlib.Path,default=P/'tensor_runtime.o');ap.add_argument('--dispatch',type=pathlib.Path,default=P/'tensor_dispatch.o');ap.add_argument('--lang',type=pathlib.Path,default=pathlib.Path('/home/a/niyah-lang-integrated-20261027'));ap.add_argument('--evidence',type=pathlib.Path,default=P/'test_receipt.json');a=ap.parse_args();start=time.perf_counter()
 assert sha(a.allocator.read_bytes())=='59447d2ededb37fb66985be7cbb2bb3c4e7b8326443141b61880ece2158e0866'
 compiler_before=sha(a.compiler.read_bytes());root_files={n:sha((a.lang/'src'/n).read_bytes()) for n in ('lexer.c','lexer.h','morph.c','morph.h')}
 compiler_elf=elf(a.compiler);rng=random.Random(270106);trace=None;outputs_checked=0
 with tempfile.TemporaryDirectory(prefix='niyah-ir-') as tmp:
  t=pathlib.Path(tmp);src=t/'source.niyah';ir=t/'plan.nir';alloc=t/'allocation';asm=t/'plan.s';obj=t/'plan.o';exe=t/'program'
  def compile_source(text,mask=None):
   if isinstance(text,str):text=text.encode()
   if mask is not None:text=('reserve %d;\n'%mask).encode()+text
   src.write_bytes(text);data=run([a.compiler,'--ir',src],group='frontend');ir.write_bytes(data);plan=unpack_ir(data)
   iv=run([a.compiler,'--intervals',ir],group='liveness')
   assert len(iv)==16+8*len(plan['values']) and iv[:8]==b'NTILE001'
   assert struct.unpack_from('<'+str(2*len(plan['values']))+'I',iv,16)==tuple(x for v in plan['values'] for x in (v['start'],v['end']))
   peak=max(sum(v['start']<=i<v['end'] for v in plan['values']) for i in range(len(plan['ops'])))
   if peak>8-plan['mask'].bit_count():run([a.allocator],iv,77,'pressure');return None
   colors=run([a.allocator],iv,group='allocation');assert len(colors)==8+len(plan['values']);alloc.write_bytes(colors)
   emitted=run([a.compiler,'--asm',ir,alloc],group='codegen');asm.write_bytes(emitted)
   run(['as','--64','-o',obj,asm]);run(['ld','-static','--build-id=none','-z','noexecstack','-o',exe,obj,a.runtime,a.dispatch]);return data,plan,colors,emitted
  demo='input_a والكتاب;\ninput_b سيكتب;\ngemm نتيجة = والكتاب * سيكتب;\nemit نتيجة;\n'
  original=compile_source(demo,1);data,plan,colors,emitted=original
  assert colors[8:]==bytes([1,2,3]);assert [(v['start'],v['end']) for v in plan['values']]==[(0,3),(1,3),(2,4)]
  assert b'tdpbssd tmm3,tmm1,tmm2' in emitted
  program_elf=elf(exe)
  feature=run([exe,'features'],group='features');fields=struct.unpack('<8Q',feature);assert feature[:8]==b'NXISA001';caps=fields[1]
  trace={'source_utf8':plan['source'].decode(),'ir_bytes':len(data),'interval_bytes':40,'assignment_hex':colors.hex(),'values':plan['values'],'ops':plan['ops'],'generated_asm':emitted.decode(),'features':dict(zip(['magic','caps','xcr0','cpuid1_ecx','cpuid7_ebx','cpuid7_edx','selected','amx_permission'],fields))}
  # Compare recorded root alternatives with the actual lexer --morph, not a guessed root dictionary.
  if (a.lang/'niyahlex').is_file():
   for v in plan['values']:
    word=t/'word';word.write_text(v['name']);out=run([a.lang/'niyahlex','--morph',word],group='morph_provenance').decode()
    roots=re.findall(r'\troot=([^\n]+)',out)
    assert [tuple(map(ord,s)) for s in roots]==[tuple(vv) for vv in v['roots']],(v,out)
    assert int(re.search(r'candidates=(\d+)',out).group(1))==v['morph_candidate_count']
  paths=['scalar','auto']+(['avx2'] if caps&1 else [])+(['avx512'] if caps&2 else [])+(['amx'] if fields[7] else [])
  # Independent signed-integer GEMM, including -128 where saturating shortcuts break.
  cases=[([0]*1024,[127]*1024),([-128]*1024,[-128]*1024),([127]*1024,[-128]*1024),([-128 if i%2 else 127 for i in range(1024)],[127 if i%3 else -128 for i in range(1024)])]
  cases += [([rng.randrange(-128,128) for _ in range(1024)],[rng.randrange(-128,128) for _ in range(1024)]) for _ in range(28)]
  for A,B in cases:
   payload=struct.pack('<2048b',*(A+B));expected=struct.pack('<256i',*[sum(A[i*64+k]*B[k*16+j] for k in range(64)) for i in range(16) for j in range(16)])
   for mode in paths:
    got=run([exe,mode],payload,group='gemm_oracle');assert got==expected,(mode,sha(got),sha(expected));outputs_checked+=256
  for mode,bit in [('avx2',1),('avx512',2),('amx',4)]:
   if not caps&bit or (mode=='amx' and not fields[7]):run([exe,mode],b'',77,'unsupported')
  for payload in [b'',b'\0'*2047,b'\0'*2049]:run([exe],payload,65,'runtime_invalid')
  run([exe,'bad'],b'',64,'runtime_invalid');run([exe,'auto','bad'],b'',64,'runtime_invalid')
  # I/O failure: closed pipe and /dev/full, no SIGPIPE termination.
  for prog,args,inp in [(exe,[],b'\0'*2048),(a.compiler,['--ir',src],None)]:
   rd,wr=os.pipe();os.close(rd)
   try:q=subprocess.run([str(prog)]+[str(x) for x in args],input=inp,stdout=wr,stderr=subprocess.PIPE,timeout=10)
   finally:os.close(wr)
   assert q.returncode==74;records.append({'argv':[str(prog),'closed_pipe'],'exit':q.returncode,'expected_exit':74});counts['io']=counts.get('io',0)+1
   with open('/dev/full','wb') as f:q=subprocess.run([str(prog)]+[str(x) for x in args],input=inp,stdout=f,stderr=subprocess.PIPE,timeout=10)
   assert q.returncode==74;counts['io']+=1;records.append({'argv':[str(prog),'/dev/full'],'exit':q.returncode,'expected_exit':74})
  saved_data=data;saved_colors=colors;sourceoff=len(data)-len(plan['source'])
  mutations=[data[:i] for i in [0,7,8,31,32,len(data)-1]]+[data+b'x']
  for _ in range(180):
   b=bytearray(data);at=rng.randrange(sourceoff);b[at]^=1<<rng.randrange(8);mutations.append(bytes(b))
  for b in mutations:
   ir.write_bytes(b);run([a.compiler,'--intervals',ir],expected=65,group='corrupt_ir');run([a.compiler,'--asm',ir,alloc],expected=65,group='corrupt_ir')
  ir.write_bytes(saved_data)
  badalloc=[b'',saved_colors[:-1],saved_colors+b'x',b'BAD!'+saved_colors[4:],saved_colors[:4]+b'\4'+saved_colors[5:],saved_colors[:5]+b'\1'+saved_colors[6:],saved_colors[:8]+bytes([1,1,3]),saved_colors[:8]+bytes([0,2,3]),saved_colors[:8]+bytes([1,2,8])]
  for b in badalloc:alloc.write_bytes(b);run([a.compiler,'--asm',ir,alloc],expected=65,group='corrupt_allocation')
  alloc.write_bytes(saved_colors)
  # Alternative valid coloring must change generated physical registers.
  alloc.write_bytes(saved_colors[:8]+bytes([7,6,5]));other=run([a.compiler,'--asm',ir,alloc],group='alternate_color');assert b'tdpbssd tmm5,tmm7,tmm6' in other
  alloc.write_bytes(saved_colors)
  # Front-end rejection, including no normalization-based symbol aliasing.
  malformed=['','input_a a;','input_a a; emit a;','input_a a; input_a a;','gemm c=a*b; emit c;','input_a a; input_b b; gemm c=b*a; emit c;',demo+'garbage;', 'reserve 256;\n'+demo,'reserve -1;\n'+demo,'reserve 0; reserve 0;\n'+demo,demo+'reserve 0;',demo.replace('سيكتب;', 'سيكتب',1),demo.replace('نتيجة =','نتيجة =='),demo.replace('والكتاب *','كتاب *'), 'input_a '+('a'*129)+';\n'+demo,'input_a a; input_b b; gemm c=a*b; gemm d=c*b; emit d;']
  for text in malformed:
   src.write_text(text);run([a.compiler,'--ir',src],expected=65,group='invalid_source')
  for b in [b'\xff',b'\xc0\xaf',demo.encode()+b'\0',demo.encode()+b'\xed\xa0\x80',b' '*1048577]:
   src.write_bytes(b);run([a.compiler,'--ir',src],expected=65,group='invalid_source')
  run([a.compiler,'--ir',t/'missing'],expected=66,group='file_errors')
  run([a.compiler],expected=64,group='cli');run([a.compiler,'--bad',src],expected=64,group='cli')
  link=t/'source-link';link.symlink_to(src);run([a.compiler,'--ir',link],expected=66,group='file_errors')
  # Every reservation mask is exercised with known peak=3. No AMX hardware needed.
  for mask in range(256):
   src.write_text('reserve %d;\n'%mask+demo);b=run([a.compiler,'--ir',src],group='mask_frontend');ir.write_bytes(b);iv=run([a.compiler,'--intervals',ir],group='mask_liveness')
   code=0 if 8-mask.bit_count()>=3 else 77
   col=run([a.allocator],iv,code,'mask_allocation')
   if code==0:
    alloc.write_bytes(col);em=run([a.compiler,'--asm',ir,alloc],group='mask_codegen');assert len(col)==11
    assert all(not(mask>>i&1) for i in col[8:])
  # Chained scheduling, shared A/B, SSA values with same root remain distinct.
  multi='input_a كاتب; input_a مكتوب; input_b كتاب; gemm أول = كاتب * كتاب; emit أول; gemm ثاني = مكتوب * كتاب; emit ثاني; gemm ثالث = كاتب * كتاب; emit ثالث;'
  compiled=compile_source(multi,3);assert compiled is not None
  b,pl,col,em=compiled;assert len(pl['values'])==6
  A=[rng.randrange(-128,128) for _ in range(1024)];B=[rng.randrange(-128,128) for _ in range(1024)];C=[rng.randrange(-128,128) for _ in range(1024)]
  def mm(x,y):return struct.pack('<256i',*[sum(x[i*64+k]*y[k*16+j] for k in range(64)) for i in range(16) for j in range(16)])
  expected=mm(A,C)+mm(B,C)+mm(A,C);payload=struct.pack('<3072b',*(A+B+C))
  for mode in paths:assert run([exe,mode],payload,group='ssa_reuse')==expected
  pressure=' '.join('input_a a%d;'%i for i in range(9))+' input_b b; '+' '.join('gemm c%d=a%d*b; emit c%d;'%(i,i,i) for i in range(9))
  assert compile_source(pressure) is None
  # Definition bound 64, operation bound 128; preserve no-partial-output semantics.
  over=' '.join('input_a a%d;'%i for i in range(65))+' emit a0;'
  src.write_text(over);run([a.compiler,'--ir',src],expected=65,group='bounds')
  many='input_a a;input_b b;gemm c=a*b;'+('emit c;'*125)
  assert compile_source(many,0) is not None
  src.write_text(many+'emit c;');run([a.compiler,'--ir',src],expected=65,group='bounds')
  # Separate pure policy probe, testing hypothetical capabilities without executing them.
  policy=t/'policy.s';policy.write_text('.intel_syntax noprefix\n.bss\nbuf:.skip 8\n.text\n.global _start\n_start:\n xor eax,eax\n xor edi,edi\n lea rsi,[rip+buf]\n mov edx,8\n syscall\n cmp eax,8\n jne fail\n mov edi,[rip+buf]\n mov esi,[rip+buf+4]\n call nx_policy\n mov [rip+buf],eax\n mov eax,1\n mov edi,1\n lea rsi,[rip+buf]\n mov edx,4\n syscall\n xor edi,edi\n jmp done\nfail: mov edi,65\ndone: mov eax,60\n syscall\n.section .note.GNU-stack,"",@progbits\n')
  run(['as','--64','-o',t/'policy.o',policy]);run(['ld','-static','--build-id=none','-z','noexecstack','-o',t/'policy',t/'policy.o',a.dispatch])
  for caps2 in range(8):
   for perm in (0,1,2):
    expected=3 if caps2&4 and perm==1 else 2 if caps2&2 else 1 if caps2&1 else 0
    got=run([t/'policy'],struct.pack('<2I',caps2,perm),group='dispatch_policy');assert got==struct.pack('<I',expected)
  # Build-to-build byte identity: exact same source, a different object filename.
  compile_source(demo,1);before=exe.read_bytes();run(['as','--64','-o',t/'other.o',asm]);run(['ld','-static','--build-id=none','-z','noexecstack','-o',t/'other',t/'other.o',a.runtime,a.dispatch]);assert (t/'other').read_bytes()==before
  program_elf=elf(exe)
  a.evidence.parent.mkdir(parents=True,exist_ok=True)
  (a.evidence.parent/'trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2)+'\n')
  (a.evidence.parent/'verified_demo.s').write_bytes(emitted)
 assert compiler_before==sha(a.compiler.read_bytes())
 assert root_files=={n:sha((a.lang/'src'/n).read_bytes()) for n in root_files}
 report={'schema':'niyah-native-morph-ir-bridge-v1','status':'PASS','exit':0,'counts':counts,'observations':len(records),'seconds':time.perf_counter()-start,'compiler':compiler_elf,'demo_elf':program_elf,'integer_outputs_checked':outputs_checked,'paths_executed':paths,'same_arithmetic_contract':'signed i8 [16,64] * signed i8 [64,16] -> i32 [16,16]','upstream_frontend_hashes':root_files,'allocator_binary_unchanged':True,'rebuild_byte_identical':True,'amx_hardware_executed':'amx' in paths,'avx512_hardware_executed':'avx512' in paths,'root_is_provenance_not_embedding':True,'trace':'trace.json','commands':records}
 a.evidence.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='commands'},indent=2))
if __name__=='__main__':main()
