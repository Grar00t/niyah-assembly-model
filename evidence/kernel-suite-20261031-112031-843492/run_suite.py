#!/usr/bin/env python3
"""Build against the explicit local repository; no torch imports or network."""
import argparse, ast, hashlib, json, os, platform, shlex, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--cc',default='gcc')
p.add_argument('--opt',choices=['O0','O2','O3'],default='O2')
p.add_argument('--full',action='store_true')
p.add_argument('--verify-only',action='store_true')
a=p.parse_args(); repo=a.repo.resolve(); dest=a.out.resolve()
if dest.exists(): raise SystemExit('STOP: output directory exists; choose a new one')
gemm=repo/'src/niyah_i8_gemm.s'
wrapper=repo/'tools/qwen/qwen_native_layer0.py'
harness=Path(__file__).resolve().with_name('niyah_kernel_suite.c')
for f in (gemm,wrapper,harness):
 if not f.is_file(): raise SystemExit(f'STOP: missing {f}')
tree=ast.parse(wrapper.read_text()); nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ASM' for t in n.targets)]
if len(nodes)!=1: raise SystemExit('STOP: expected one literal ASM assignment')
asm=ast.literal_eval(nodes[0].value)
if 'niyah_dot128:' not in asm: raise SystemExit('STOP: dot128 missing')
dest.mkdir(parents=True)
(dest/'wrapper_dots.s').write_text(asm)
flags=['-std=c11','-'+a.opt,'-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize','-Wl,-z,noexecstack']
command=[a.cc,*flags,str(harness),str(gemm),str(dest/'wrapper_dots.s'),'-lm','-o',str(dest/'suite')]
meta={'repository':str(repo),'wrapper':str(wrapper),'platform':platform.platform(),'compiler':subprocess.check_output([a.cc,'--version'],text=True).splitlines()[0],'command':command,'hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (gemm,wrapper,harness)},'scope':'Serial kernel verification; no Qwen weights, no g32 validation; timings are host-specific'}
try:
 print(shlex.join(command),flush=True)
 subprocess.run(command,check=True)
 cmd=[str(dest/'suite')]
 if a.full: cmd+=['--full']
 if a.verify_only: cmd+=['--verify-only']
 with (dest/'benchmark.csv').open('w') as stdout,(dest/'verification.log').open('w') as stderr:
  result=subprocess.run(cmd,stdout=stdout,stderr=stderr)
 meta['returncode']=result.returncode
 print((dest/'verification.log').read_text(),end='')
 print((dest/'benchmark.csv').read_text(),end='')
except subprocess.CalledProcessError as e:
 meta['returncode']=e.returncode
finally:
 (dest/'metadata.json').write_text(json.dumps(meta,indent=2))
sys.exit(meta.get('returncode',1))
