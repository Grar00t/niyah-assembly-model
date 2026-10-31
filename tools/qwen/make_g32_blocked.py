#!/usr/bin/env python3
# make_g32_blocked.py — creates tools/qwen/qwen_native_layer0_g32_blocked.py from qwen_native_layer0_g32.py.
# Original is not touched. Refuses to overwrite. Every substitution must match exactly once.
# Change: down_proj g32 now runs niyah_i8_gemm per K=32 block (weights stored group-major) instead of
# 25M separate niyah_i8_dot calls. Accumulation order per output is unchanged, so results must be bit-identical.
import ast, sys
from pathlib import Path
src = Path('tools/qwen/qwen_native_layer0_g32.py'); dst = Path('tools/qwen/qwen_native_layer0_g32_blocked.py')
if not src.is_file(): raise SystemExit('STOP: missing ' + str(src))
if dst.exists(): raise SystemExit('STOP: exists ' + str(dst))
s = src.read_text()
def sub(old, new):
    global s
    if s.count(old) != 1: raise SystemExit('STOP: expected exactly one match for: ' + old[:70])
    s = s.replace(old, new)

BLOCKED = r"""/* down_proj g32, blocked: weights are group-major [g][rows][32]; activations packed [g][t][32].
   Same per-(i,j) accumulation order as linear_g32 (g ascending, double), so results are bit-identical. */
static int linear_g32_blocked(const float *x,float *y,const Matrix *m,int t,int8_t *aq,int32_t *acc){
 const int group=32;
 if(!x||!y||!m||!aq||!acc||!m->w||!m->s||t<1||t>16||m->rows!=4096||m->cols!=12288)return 64;
 const int groups=m->cols/group,rows=m->rows;
 double scales[16][384];
 for(int g=0;g<groups;++g)for(int i=0;i<t;++i){
  int rc=quant(x+(int64_t)i*m->cols+(int64_t)g*group,aq+((int64_t)g*t+i)*group,group,&scales[i][g]);
  if(rc)return rc;}
 double *tot=calloc((size_t)t*rows,sizeof(double));
 if(!tot)return 70;
 for(int g=0;g<groups;++g){
  if(niyah_i8_gemm(aq+(int64_t)g*t*group,m->w+(int64_t)g*rows*group,acc,t,rows,group)){free(tot);return 64;}
  for(int i=0;i<t;++i){const double sc=scales[i][g];double *o=tot+(int64_t)i*rows;const int32_t *a=acc+(int64_t)i*rows;
   for(int j=0;j<rows;++j)o[j]+=(double)a[j]*sc;}}
 for(int i=0;i<t;++i)for(int j=0;j<rows;++j){float v=(float)(tot[(int64_t)i*rows+j]*(double)m->s[j]);
  if(!isfinite(v)){free(tot);return 65;}y[(int64_t)i*rows+j]=v;}
 free(tot);return 0;
}
"""

a = s.index('/* Verification path: existing assembly INT8 dot per 32-element group.')
b = s.index('static int norm(')
s = s[:a] + BLOCKED + '\n' + s[b:]
sub('CHECK(linear_g32(z,p,m+6,t,aq));', 'CHECK(linear_g32_blocked(z,p,m+6,t,aq,acc));')
sub('        packed=[]\n', '        packed=[];shapes=[]\n')
sub("            packed.append((q,s))\n",
    "            shapes.append(q.shape)\n"
    "            if name=='mlp.down_proj.weight':q=np.ascontiguousarray(q.reshape(q.shape[0],-1,32).transpose(1,0,2))\n"
    "            packed.append((q,s))\n")
sub("print('QUANTIZED:',name,q.shape,flush=True)",
    "print('QUANTIZED:',name,shapes[-1],'group-major g32' if name=='mlp.down_proj.weight' else '',flush=True)")
sub('Matrix(q.ctypes.data,s.ctypes.data,*q.shape) for q,s in packed',
    'Matrix(q.ctypes.data,s.ctypes.data,*sh) for (q,s),sh in zip(packed,shapes)')
sub('down_proj uses assembly INT8 dot per g32 with C FP64 scaled accumulation',
    'down_proj runs niyah_i8_gemm per K=32 block (group-major weights) with C FP64 scaled accumulation')
sub('down_proj uses unblocked assembly dot calls per g32', 'down_proj g32 blocks are K=32 GEMM calls, not a fused kernel')
ast.parse(s)
dst.write_text(s)
print('Created:', dst)
