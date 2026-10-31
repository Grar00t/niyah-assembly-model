# Inventory D:/Niyah.Engine via native Win32 scandir, read-only on target.
import os,csv,gzip,json,time,datetime,sys
from pathlib import Path
ROOT='D:/Niyah.Engine'
OUT='//wsl.localhost/kali-linux/home/a/qwen-layer0-quant-verify-v1'
SCRIPT=os.path.join(OUT,'inventory_engine_windows_v2.py')
TABLE=os.path.join(OUT,'inventory_engine_windows_v1.tsv.gz')
SUMMARY=os.path.join(OUT,'inventory_engine_windows_summary_v1.json')
BRANCH='feature/oracle-engine-three-lobe-20261002'
GIT_ROOT='D:/Niyah.Engine'
LAST='2026-10-12T02:16:31+03:00'
DIRTY='DIRTY'
HEAD='2a0ec79faf28'
FIELDS=['path','kind','size_bytes','git_root','git_branch','repo_dirty','last_repo_commit_date','purpose','archive_candidate_reason']
counts={'file':0,'directory':0,'symlink':0,'errors':0}
errors=[]
top=[]
start=time.monotonic()
archive_dirs=[]
def describe(p,k):
 n=os.path.basename(p).lower()
 parts=p.replace('\\','/').lower().split('/')
 if k=='directory':
  if n=='.git':return 'Git history and index; preserve'
  if n.startswith('build') or n in ('.hadeed-build','.release'):
   return 'Generated build/release directory; dependency status unverified'
  if n in ('corpus','datasets'):return 'Training/corpus data; not the Layer-0 validation target'
  if n in ('src','include','tests','docs','scripts','artifacts','runs','reports','eval','tools'):return n+' directory within Windows native runtime'
  return 'Directory within Windows Niyah.Engine checkout'
 if n.endswith(('.safetensors','.gguf','.pt','.pth','.bin')):return 'Model weights / binary payload; preserve'
 if '.git' in parts:return 'Git metadata'
 if 'tests' in parts:return 'Test source or test fixture'
 if 'corpus' in parts or 'datasets' in parts:return 'Corpus/data asset'
 if 'reports' in parts or 'artifacts' in parts:return 'Report or generated artifact'
 if n.endswith(('.c','.h','.s','.cpp','.cu','.py','.ps1','.cmake')):return 'Native source or build script'
 if n.endswith(('.exe','.o','.obj','.so','.dll','.a','.lib','.pdb')):return 'Build output binary'
 if n.endswith(('.md','.rst')):return 'Documentation'
 if n.endswith(('.json','.txt','.log','.csv','.tsv')):return 'Data, configuration or log'
 return 'Other runtime/repository asset'
def candidate(p,k):
 if k!='directory':return ''
 n=os.path.basename(p).lower()
 if n.startswith('build-') or n in ('.hadeed-build','build_repair','build_repair2'):
  return 'ARCHIVE-CANDIDATE: additional generated build tree; confirm outputs/consumers before any move'
 return ''
if not os.path.isdir(ROOT):raise SystemExit('TARGET_MISSING')
if os.path.exists(TABLE) or os.path.exists(SUMMARY):
 raise SystemExit('STOP: one or more destination files already exist')
# This source is saved by the launcher before running; never overwrite any source file.
with gzip.open(TABLE,'xt',encoding='utf-8',newline='',compresslevel=3) as f:
 writer=csv.writer(f,delimiter='\t',lineterminator='\n')
 writer.writerow(FIELDS)
 def emit(p,k,sz):
  why=candidate(p,k)
  writer.writerow([p.replace('\\','/'),k,sz,GIT_ROOT,BRANCH,DIRTY,LAST,describe(p,k),why])
  counts[k]+=1
  if why:archive_dirs.append(p.replace('\\','/'))
  n=counts['file']+counts['directory']+counts['symlink']
  if n and n%100000==0:
   print('PROGRESS',n, 'files',counts['file'],'elapsed_s',round(time.monotonic()-start,1), flush=True)
 def start_frame(path):
  return {'path':path,'iter':os.scandir(path),'bytes':0}
 stack=[start_frame(ROOT)]
 while stack:
  frame=stack[-1]
  try:
   ent=next(frame['iter'])
  except StopIteration:
   frame['iter'].close()
   sz=frame['bytes']
   emit(frame['path'],'directory',sz)
   stack.pop()
   if stack:stack[-1]['bytes']+=sz
   else:total_bytes=sz
   if len(stack)==1:
    top.append({'path':frame['path'].replace('\\','/'),'size_bytes':sz,'archive_candidate_reason':candidate(frame['path'],'directory')})
   continue
  except OSError as exc:
   counts['errors']+=1
   if len(errors)<30:errors.append(str(exc))
   frame['iter'].close()
   emit(frame['path'],'directory',frame['bytes'])
   stack.pop()
   if stack:stack[-1]['bytes']+=frame['bytes']
   continue
  try:
   if ent.is_symlink():
    size=ent.stat(follow_symlinks=False).st_size
    emit(ent.path,'symlink',size)
   elif ent.is_dir(follow_symlinks=False):
    stack.append(start_frame(ent.path))
   else:
    size=ent.stat(follow_symlinks=False).st_size
    emit(ent.path,'file',size)
    frame['bytes']+=size
  except OSError as exc:
   counts['errors']+=1
   if len(errors)<30:errors.append(f'{ent.path}: {exc}')
print('FINISHED',counts,'total_bytes',total_bytes,'elapsed_s',round(time.monotonic()-start,1),flush=True)
with open(SUMMARY,'x',encoding='utf-8') as f:
 json.dump({'status':'COMPLETE' if not counts['errors'] else 'PARTIAL_ERRORS',
            'scope':ROOT,'table':TABLE,'counts':counts,'total_file_bytes':total_bytes,
            'git_branch':BRANCH,'git_dirty':DIRTY,'git_HEAD_snapshot':HEAD,
            'last_commit_date':LAST,'top_level_directories':top,'archive_candidates':archive_dirs,
            'errors':errors,'elapsed_seconds':round(time.monotonic()-start,1),
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f,indent=2,ensure_ascii=False)
