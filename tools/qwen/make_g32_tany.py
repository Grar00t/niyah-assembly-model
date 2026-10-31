#!/usr/bin/env python3
# make_g32_tany.py — creates tools/qwen/qwen_native_layer0_g32_tany.py from qwen_native_layer0_g32_blocked.py.
# Original is not touched. Refuses to overwrite. Every substitution must match exactly once.
# Change: removes the t<=16 limit (stack arrays scale[16] and scales[16][384] become heap buffers).
# Per-(i,j) arithmetic is unchanged, so t=1/7/16 outputs must be bit-identical to the blocked version.
# Adds a t=45 case (whole prompt) for rel L2 only.
import ast
from pathlib import Path
src = Path('tools/qwen/qwen_native_layer0_g32_blocked.py'); dst = Path('tools/qwen/qwen_native_layer0_g32_tany.py')
if not src.is_file(): raise SystemExit('STOP: missing ' + str(src))
if dst.exists(): raise SystemExit('STOP: exists ' + str(dst))
s = src.read_text()
def sub(old, new):
    global s
    if s.count(old) != 1: raise SystemExit('STOP: expected exactly one match for: ' + old[:70])
    s = s.replace(old, new)

sub(" double scale[16];\n for(int i=0;i<t;++i){int rc=quant(x+(int64_t)i*m->cols,aq+(int64_t)i*m->cols,m->cols,&scale[i]);if(rc)return rc;}\n if(niyah_i8_gemm(aq,m->w,acc,t,m->rows,m->cols))return 64;\n",
    " double *scale=malloc((size_t)t*sizeof(double));if(!scale)return 70;\n for(int i=0;i<t;++i){int rc=quant(x+(int64_t)i*m->cols,aq+(int64_t)i*m->cols,m->cols,&scale[i]);if(rc){free(scale);return rc;}}\n if(niyah_i8_gemm(aq,m->w,acc,t,m->rows,m->cols)){free(scale);return 64;}\n")
sub("if(!isfinite(f))return 65;y[(int64_t)i*m->rows+j]=f;}\n return 0;\n",
    "if(!isfinite(f)){free(scale);return 65;}y[(int64_t)i*m->rows+j]=f;}\n free(scale);return 0;\n")
sub("t<1||t>16||m->rows!=4096", "t<1||t>4096||m->rows!=4096")
sub(" double scales[16][384];\n", " double *scales=malloc((size_t)t*384*sizeof(double));if(!scales)return 70;\n")
sub("&scales[i][g]);\n  if(rc)return rc;}", "&scales[(int64_t)i*384+g]);\n  if(rc){free(scales);return rc;}}")
sub(" if(!tot)return 70;\n", " if(!tot){free(scales);return 70;}\n")
sub("{free(tot);return 64;}", "{free(tot);free(scales);return 64;}")
sub("const double sc=scales[i][g];", "const double sc=scales[(int64_t)i*384+g];")
sub("  if(!isfinite(v)){free(tot);return 65;}", "  if(!isfinite(v)){free(tot);free(scales);return 65;}")
sub(" free(tot);return 0;\n}", " free(tot);free(scales);return 0;\n}")
sub("!sine||t<1||t>16||", "!sine||t<1||t>4096||")
sub("['input_ids'][:16]\n        if len(ids)!=16:", "['input_ids'][:45]\n        if len(ids)!=45:")
sub("for t in (1,7,16):", "for t in (1,7,16,45):")
sub("'scope':'Full decoder layer 0 only; no model generation'", "'scope':'Full decoder layer 0 only; no model generation; t up to 4096 (tested 1,7,16,45)'")
ast.parse(s)
dst.write_text(s)
print('Created:', dst)
