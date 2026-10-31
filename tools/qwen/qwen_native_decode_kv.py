#!/usr/bin/env python3
# qwen_native_decode_kv.py — native KV-cache greedy decoding for Qwen3-8B-Base (g32 activations on all seven linears),
# compared token for token with greedy decoding by transformers FP32 decoder layers (layer-streamed, full recompute).
# Native weights: INT8 per-row + g32 activations, cached once under build/qwen_q8g32/ (regenerable) and mmapped.
# lm_head: FP32 math on BF16 weights mmapped straight from safetensors (bf16->fp32 is exact).
# Native C/asm: qwen_native_layer0_g32_tany.py + qwen_native_full_gate.py EXTRA_C, unmodified, plus DECODE_C below.
import ctypes, gc, hashlib, importlib.util, json, mmap, os, resource, shutil, struct, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SPEC = importlib.util.spec_from_file_location('gate', HERE/'qwen_native_full_gate.py')
GATE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(GATE)
TANY = GATE.TANY
T0, NEW = GATE.T, 64
CAP = T0 + NEW
NTHREADS = os.cpu_count() or 1

DECODE_C = r'''
#include <omp.h>
void niyah_set_threads(int n){omp_set_num_threads(n);}
static int attention_kv(const float *q,const float *kc,const float *vc,float *y,int t,int pos0){
 const double scale=1.0/sqrt(128.0);
 for(int i=0;i<t;++i)for(int h=0;h<32;++h){double acc[128]={0},maximum=-INFINITY,total=0;
  const float *query=q+((int64_t)i*32+h)*128;
  for(int j=0;j<=pos0+i;++j){const float *key=kc+((int64_t)j*8+h/4)*128;const float *value=vc+((int64_t)j*8+h/4)*128;
   double score=niyah_dot128(query,key)*scale,next=fmax(maximum,score),a=exp(maximum-next),b=exp(score-next);
   total=total*a+b;for(int d=0;d<128;++d)acc[d]=acc[d]*a+b*value[d];maximum=next;}
  if(!(total>0)||!isfinite(total))return 65;
  for(int d=0;d<128;++d)y[((int64_t)i*32+h)*128+d]=(float)(acc[d]/total);}
 return 0;
}
/* Same operation order as niyah_layer_g32all; K/V for positions pos0..pos0+t-1 are written into kc/vc [cap][8][128]. */
int niyah_layer_kv(const float *x,float *y,const Matrix *m,
 const float *inorm,const float *pnorm,const float *qn,const float *kn,
 const float *cosine,const float *sine,int pos0,int t,int cap,float *kc,float *vc,float eps){
 if(!available())return 77;
 if(!x||!y||!m||!inorm||!pnorm||!qn||!kn||!cosine||!sine||!kc||!vc||t<1||pos0<0||cap>4096||pos0+t>cap||!(eps>0)||!isfinite(eps))return 64;
 if(fegetround()!=FE_TONEAREST)return 65;
 const int rows[7]={4096,1024,1024,4096,12288,12288,4096};
 const int cols[7]={4096,4096,4096,4096,4096,4096,12288};
 for(int i=0;i<7;++i)if(m[i].rows!=rows[i]||m[i].cols!=cols[i]||!m[i].w||!m[i].s)return 64;
 size_t n=(size_t)t*4096,nm=(size_t)t*12288;
 float *memory=calloc(5*n+3*nm,sizeof(float));int8_t *aq=malloc((size_t)t*12288);int32_t *acc=malloc((size_t)t*12288*sizeof(int32_t));
 if(!memory||!aq||!acc){free(memory);free(aq);free(acc);return 70;}
 float *h=memory,*q=h+n,*a=q+n,*p=a+n,*res=p+n,*g=res+n,*u=g+nm,*z=u+nm;
 float *k=kc+(int64_t)pos0*1024,*v=vc+(int64_t)pos0*1024;
 int rc=0;
 #undef CHECK
 #define CHECK(expr) do{rc=(expr);if(rc)goto cleanup;}while(0)
 CHECK(norm(x,inorm,h,t,4096,eps));
 CHECK(linear_g32_any(h,q,m+0,t,aq,acc));CHECK(linear_g32_any(h,k,m+1,t,aq,acc));CHECK(linear_g32_any(h,v,m+2,t,aq,acc));
 CHECK(norm(q,qn,q,t*32,128,eps));CHECK(norm(k,kn,k,t*8,128,eps));
 rope(q,cosine+(int64_t)pos0*128,sine+(int64_t)pos0*128,t,32);rope(k,cosine+(int64_t)pos0*128,sine+(int64_t)pos0*128,t,8);
 CHECK(attention_kv(q,kc,vc,a,t,pos0));CHECK(linear_g32_any(a,p,m+3,t,aq,acc));
 for(size_t i=0;i<n;++i)res[i]=x[i]+p[i];
 CHECK(norm(res,pnorm,h,t,4096,eps));
 CHECK(linear_g32_any(h,g,m+4,t,aq,acc));CHECK(linear_g32_any(h,u,m+5,t,aq,acc));
 for(size_t i=0;i<nm;++i){double d=g[i],sig;if(d>=0)sig=1.0/(1.0+exp(-d));else{double e=exp(d);sig=e/(1.0+e);}z[i]=(float)(d*sig*(double)u[i]);}
 CHECK(linear_g32_any(z,p,m+6,t,aq,acc));
 for(size_t i=0;i<n;++i){y[i]=res[i]+p[i];if(!isfinite(y[i])){rc=65;goto cleanup;}}
cleanup:
 free(memory);free(aq);free(acc);return rc;
}
__attribute__((target("avx2,fma"))) static inline __m256 bf16x8(const uint16_t *p){
 return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)),16));}
/* Same accumulator layout and order as dot_f32, on exact bf16->fp32 weights. */
__attribute__((target("avx2,fma"))) static float dot_bf16(const float *a,const uint16_t *b,int n){
 __m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps(),s2=_mm256_setzero_ps(),s3=_mm256_setzero_ps();
 for(int k=0;k<n;k+=32){
  s0=_mm256_fmadd_ps(_mm256_loadu_ps(a+k),bf16x8(b+k),s0);
  s1=_mm256_fmadd_ps(_mm256_loadu_ps(a+k+8),bf16x8(b+k+8),s1);
  s2=_mm256_fmadd_ps(_mm256_loadu_ps(a+k+16),bf16x8(b+k+16),s2);
  s3=_mm256_fmadd_ps(_mm256_loadu_ps(a+k+24),bf16x8(b+k+24),s3);}
 s0=_mm256_add_ps(_mm256_add_ps(s0,s1),_mm256_add_ps(s2,s3));
 float lane[8];_mm256_storeu_ps(lane,s0);double r=0;for(int i=0;i<8;++i)r+=lane[i];return (float)r;
}
int niyah_bf16_matmul(const float *x,const uint16_t *w,float *y,int t,int rows,int cols){
 if(!available())return 77;
 if(!x||!w||!y||t<1||rows<1||cols<32||cols%32)return 64;
 #pragma omp parallel for schedule(static)
 for(int j=0;j<rows;++j){const uint16_t *wr=w+(int64_t)j*cols;
  for(int i=0;i<t;++i)y[(int64_t)i*rows+j]=dot_bf16(x+(int64_t)i*cols,wr,cols);}
 return 0;
}
'''

def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def willneed(arrays):
    # Async kernel readahead of whole mappings; fault-driven mmap reads measured ~120 MB/s vs ~740 MB/s sequential.
    for a in arrays:
        m=getattr(a,'_mmap',None)
        if m is not None: m.madvise(mmap.MADV_WILLNEED)

class ST:
    def __init__(self, model):
        self.model=model; self.map=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']; self.hdr={}
    def mm(self, name):
        path=self.model/self.map[name]
        if path not in self.hdr:
            with open(path,'rb') as f:
                n=struct.unpack('<Q',f.read(8))[0]; self.hdr[path]=(n,json.loads(f.read(n)))
        n,h=self.hdr[path]; info=h[name]
        if info['dtype']!='BF16': raise SystemExit('STOP: expected BF16 '+name)
        b,e=info['data_offsets']
        return np.memmap(path,dtype=np.uint16,mode='r',offset=8+n+b,shape=tuple(info['shape']))
    def f32(self, name):
        return (np.asarray(self.mm(name)).astype(np.uint32)<<16).view(np.float32)

def build_cache(st, cfg, cache):
    done=cache/'COMPLETE'
    if done.is_file(): return json.loads(done.read_text())
    cache.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    for l in range(cfg.num_hidden_layers):
        for i,name in enumerate(GATE.NAMES):
            q,s=GATE.quant_rows(st.f32('model.layers.%d.%s'%(l,name)))
            q=np.ascontiguousarray(q.reshape(q.shape[0],-1,32).transpose(1,0,2))
            q.tofile(cache/('L%02d_%d.q'%(l,i))); s.tofile(cache/('L%02d_%d.s'%(l,i)))
        for i,k in enumerate(('input_layernorm.weight','post_attention_layernorm.weight','self_attn.q_norm.weight','self_attn.k_norm.weight')):
            np.ascontiguousarray(st.f32('model.layers.%d.%s'%(l,k))).tofile(cache/('L%02d_n%d.f32'%(l,i)))
        print(json.dumps({'cache_layer':l,'s':round(time.perf_counter()-t0,1)}),flush=True)
    meta={'gate_script_sha256':hashlib.sha256((HERE/'qwen_native_full_gate.py').read_bytes()).hexdigest(),
          'tany_sha256':hashlib.sha256((HERE/'qwen_native_layer0_g32_tany.py').read_bytes()).hexdigest(),
          'model_dir':str(st.model),'seconds':round(time.perf_counter()-t0,1)}
    done.write_text(json.dumps(meta)); return meta

def load_layers(cfg, cache):
    shapes=[(4096,4096),(1024,4096),(1024,4096),(4096,4096),(12288,4096),(12288,4096),(4096,12288)]
    # First layers up to NIYAH_PIN_GB are read into anonymous memory (never evicted); the rest stay mmapped.
    layers=[]; budget=float(os.environ.get('NIYAH_PIN_GB','3.5'))*2**30; used=0
    for l in range(cfg.num_hidden_layers):
        pin=used+sum((cache/('L%02d_%d.q'%(l,i))).stat().st_size for i in range(7))<=budget
        load=(lambda p,dt: np.fromfile(p,dtype=dt)) if pin else (lambda p,dt: np.memmap(p,dtype=dt,mode='r'))
        qs=[load(cache/('L%02d_%d.q'%(l,i)),np.int8) for i in range(7)]
        ss=[load(cache/('L%02d_%d.s'%(l,i)),np.float64) for i in range(7)]
        if pin: used+=sum(q.nbytes for q in qs)
        ns=[np.fromfile(cache/('L%02d_n%d.f32'%(l,i)),dtype=np.float32) for i in range(4)]
        mats=(GATE.Matrix*7)(*[GATE.Matrix(q.ctypes.data,s.ctypes.data,*sh) for q,s,sh in zip(qs,ss,shapes)])
        layers.append((mats,ns,qs,ss))
    print(json.dumps({'pinned_layers':sum(isinstance(x[2][0],np.memmap)==False for x in layers),'pinned_gb':round(used/2**30,2)}),flush=True)
    return layers

def native_generate(libs, layers, st, cfg, ids_list, cos_np, sin_np, nthreads):
    for lib in libs: lib.niyah_set_threads(nthreads)
    P=len(ids_list); L=cfg.num_hidden_layers
    emb=st.mm('model.embed_tokens.weight'); head=st.mm('lm_head.weight'); fw=st.f32('model.norm.weight')
    kc=[[np.zeros((CAP,8,128),np.float32) for _ in range(L)] for _ in range(P)]
    vc=[[np.zeros((CAP,8,128),np.float32) for _ in range(L)] for _ in range(P)]
    out=[[] for _ in range(P)]; logits_last=[None]*P
    workers=min(nthreads,P); pool=ThreadPoolExecutor(max_workers=workers) if workers>1 else None
    def embed(tokens): return (np.asarray(emb[tokens]).astype(np.uint32)<<16).view(np.float32)
    x=[np.ascontiguousarray(embed(ids)) for ids in ids_list]; pos=0
    t_prefill=None; start=time.perf_counter()
    for step in range(NEW):
        t=x[0].shape[0]
        for l in range(L):
            mats,ns,_,_=layers[l]
            willneed(layers[l+1][2]+layers[l+1][3] if l+1<L else [head])
            def run(p,lib):
                y=np.empty_like(x[p])
                rc=lib.niyah_layer_kv(x[p].ctypes.data,y.ctypes.data,ctypes.addressof(mats),*[a.ctypes.data for a in ns],
                    cos_np.ctypes.data,sin_np.ctypes.data,pos,t,CAP,kc[p][l].ctypes.data,vc[p][l].ctypes.data,cfg.rms_norm_eps)
                if rc: raise RuntimeError('native kv layer %d prompt %d rc=%d'%(l,p,rc))
                x[p]=y
            if pool:
                list(pool.map(lambda k:[run(p,libs[k]) for p in range(k,P,workers)],range(workers)))
            else:
                for p in range(P): run(p,libs[0])
        last=np.ascontiguousarray(np.stack([x[p][-1] for p in range(P)]))
        hn=np.empty_like(last)
        if libs[0].niyah_rmsnorm(last.ctypes.data,fw.ctypes.data,hn.ctypes.data,P,4096,cfg.rms_norm_eps): raise RuntimeError('norm')
        lg=np.empty((P,cfg.vocab_size),np.float32)
        if libs[0].niyah_bf16_matmul(hn.ctypes.data,head.ctypes.data,lg.ctypes.data,P,cfg.vocab_size,4096): raise RuntimeError('lm_head')
        willneed(layers[0][2]+layers[0][3])
        nxt=lg.argmax(-1)
        for p in range(P): out[p].append(int(nxt[p])); logits_last[p]=lg[p].copy()
        pos+=t
        if step==0: t_prefill=time.perf_counter()-start; t_dec=time.perf_counter()
        x=[np.ascontiguousarray(embed([int(nxt[p])])) for p in range(P)]
    t_decode=time.perf_counter()-t_dec
    if pool: pool.shutdown()
    return out, logits_last, {'prefill_s':round(t_prefill,2),'decode_s':round(t_decode,2),'decode_steps':NEW-1,
        'streams':P,'tokens_per_s_decode':round((NEW-1)*P/t_decode,4)}

def reference_teacher_forced(st, cfg, ids_list, native_out, rotary):
    # One layer-streamed pass over prompt + native tokens. The reference greedy sequence equals the native one
    # exactly up to the first position where the reference argmax differs; that argmax is the reference's token there.
    P=len(ids_list); seqs=[list(a)+list(b[:-1]) for a,b in zip(ids_list,native_out)]
    fw=torch.from_numpy(st.f32('model.norm.weight')); V=cfg.vocab_size; start=time.perf_counter()
    for step in range(1):
        t=len(seqs[0])
        emb=st.mm('model.embed_tokens.weight')
        h=torch.from_numpy(np.stack([(np.asarray(emb[s]).astype(np.uint32)<<16).view(np.float32) for s in seqs]))
        with torch.inference_mode(): co,si=rotary(h[:1],torch.arange(t).unsqueeze(0))
        mask=torch.full((1,1,t,t),float('-inf')).triu(1)
        for l in range(cfg.num_hidden_layers):
            with torch.device('meta'): layer=Qwen3DecoderLayer(cfg,layer_idx=l)
            if l+1<cfg.num_hidden_layers: willneed([st.mm('model.layers.%d.%s'%(l+1,k)) for k in layer.state_dict()])
            state={k:torch.from_numpy(st.f32('model.layers.%d.%s'%(l,k))) for k in layer.state_dict()}
            layer.load_state_dict(state,strict=True,assign=True); layer.eval()
            with torch.inference_mode():
                v=layer(h,attention_mask=mask,position_ids=torch.arange(t).unsqueeze(0),position_embeddings=(co,si),use_cache=False)
                h=(v[0] if isinstance(v,tuple) else v).contiguous()
            del layer,state
            print(json.dumps({'ref_layer':l,'s':round(time.perf_counter()-start,1)}),flush=True)
        with torch.inference_mode():
            r=h[:,T0-1:].reshape(-1,4096); rn=(r*torch.rsqrt(r.pow(2).mean(-1,keepdim=True)+cfg.rms_norm_eps))*fw
            lg=torch.empty((r.shape[0],V))
            for b in range(0,V,16384):
                e=min(b+16384,V); lg[:,b:e]=rn@torch.from_numpy((np.asarray(st.mm('lm_head.weight')[b:e]).astype(np.uint32)<<16).view(np.float32)).T
        ref_next=lg.argmax(-1).reshape(P,NEW).tolist()
    return ref_next, round(time.perf_counter()-start,1)

def main():
    import faulthandler; faulthandler.enable()
    torch.set_num_threads(NTHREADS)
    model=Path(os.environ.get('NIYAH_MODEL_DIR',str(Path.home()/'models/Qwen3-8B-Base'))).expanduser().resolve()
    cfg=AutoConfig.from_pretrained(str(model),local_files_only=True,trust_remote_code=False); cfg._attn_implementation='eager'
    st=ST(model); tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    ids=[tok(text,add_special_tokens=False)['input_ids'][:T0] for _,text in GATE.PROMPTS]
    report={'scope':'Greedy decoding, %d prompts x %d prompt tokens + %d new tokens'%(len(ids),T0,NEW),
            'native':'INT8 weights per row, g32 activations on all seven linears, KV cache; lm_head FP32 math on BF16 weights',
            'reference':'transformers Qwen3DecoderLayer FP32 eager, layer-streamed, one teacher-forced pass over prompt+native tokens; '
                        'equals reference greedy decoding up to and including the first divergence; reference text after divergence not computed; not model.generate',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'torch_version':torch.__version__,'cpu_threads':NTHREADS}
    report['cache']=build_cache(st,cfg,REPO/'build'/'qwen_q8g32')
    rotary=Qwen3RotaryEmbedding(config=cfg)
    with torch.inference_mode(): co,si=rotary(torch.zeros(1,CAP,4096),torch.arange(CAP).unsqueeze(0))
    cos_np=np.ascontiguousarray(co[0].numpy()); sin_np=np.ascontiguousarray(si[0].numpy())
    with tempfile.TemporaryDirectory(prefix='qwen-native-decode-') as tmp:
        work=Path(tmp)
        gemm_src=(REPO/'src/niyah_i8_gemm.s').read_text()
        (work/'kernels.s').write_text(TANY.ASM+gemm_src.replace('.section .note.GNU-stack, "", @progbits\n',''))
        csrc=TANY.C_SOURCE+GATE.EXTRA_C+'\n'+GATE.g32all_layer_source(TANY.C_SOURCE)+DECODE_C
        (work/'layer.c').write_text(csrc)
        report['native_source_sha256']=hashlib.sha256((TANY.ASM+gemm_src+csrc).encode()).hexdigest()
        subprocess.run(['gcc','-std=c11','-O2','-Wall','-Wextra','-Werror','-fPIC','-shared','-fopenmp',
            '-fno-fast-math','-ffp-contract=off','-Wl,-z,defs,-z,noexecstack',
            '-o',str(work/'layer.so'),str(work/'layer.c'),str(work/'kernels.s'),'-lm'],check=True,timeout=120)
        libs=[]
        for k in range(NTHREADS):   # separate copies: niyah_i8_gemm keeps per-library .bss scratch
            path=work/('layer_%d.so'%k); shutil.copyfile(work/'layer.so',path); lib=ctypes.CDLL(str(path))
            lib.niyah_layer_kv.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int]*3+[ctypes.c_void_p]*2+[ctypes.c_float]; lib.niyah_layer_kv.restype=ctypes.c_int
            lib.niyah_layer_g32all.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int,ctypes.c_float]; lib.niyah_layer_g32all.restype=ctypes.c_int
            lib.niyah_rmsnorm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_float]; lib.niyah_rmsnorm.restype=ctypes.c_int
            lib.niyah_bf16_matmul.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3; lib.niyah_bf16_matmul.restype=ctypes.c_int
            lib.niyah_f32_matmul.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4; lib.niyah_f32_matmul.restype=ctypes.c_int
            lib.niyah_set_threads.argtypes=[ctypes.c_int]; lib.niyah_set_threads.restype=None
            libs.append(lib)
        if not libs[0].available(): raise SystemExit('STOP: AVX2 unavailable')
        layers=load_layers(cfg,REPO/'build'/'qwen_q8g32')
        prior=os.environ.get('NIYAH_PERF1_FROM_LOG')
        if prior:   # reuse a logged 1-thread measurement (same script logic; that run died later in a separate step)
            perf1=next(json.loads(s)['native_1thread'] for s in Path(prior).read_text().splitlines() if s.startswith('{"native_1thread"'))
            perf1['source_log']=prior; out1=None
        else:       # 1 thread, 1 stream (prompt 0)
            out1,last1,perf1=native_generate(libs[:1],layers,st,cfg,ids[:1],cos_np,sin_np,1)
        print(json.dumps({'native_1thread':perf1}),flush=True)
        # all threads, 10 streams
        outN,lastN,perfN=native_generate(libs,layers,st,cfg,ids,cos_np,sin_np,NTHREADS)
        print(json.dumps({'native_all_threads':perfN}),flush=True)
        state=REPO/'build'/('decode_state_%d.json'%os.getpid())
        state.write_text(json.dumps({'outN':outN,'perfN':perfN,'lastN0_sha256':hashlib.sha256(lastN[0].tobytes()).hexdigest()}))
        print(json.dumps({'checkpoint':str(state)}),flush=True)
        same_single_vs_batch=None if out1 is None else (out1[0]==outN[0] and np.array_equal(last1[0],lastN[0]))
        # KV cache vs no-cache recompute (prompt 0): final-position logits must be bit-identical
        seq=ids[0]+outN[0][:-1]; emb=st.mm('model.embed_tokens.weight')
        h=np.ascontiguousarray((np.asarray(emb[seq]).astype(np.uint32)<<16).view(np.float32)); t=len(seq)
        for l in range(cfg.num_hidden_layers):
            mats,ns,_,_=layers[l]; y=np.empty_like(h)
            rc=libs[0].niyah_layer_g32all(h.ctypes.data,y.ctypes.data,ctypes.addressof(mats),*[a.ctypes.data for a in ns],cos_np.ctypes.data,sin_np.ctypes.data,t,cfg.rms_norm_eps)
            if rc: raise RuntimeError('recompute rc=%d'%rc)
            h=y
        fw=st.f32('model.norm.weight'); last=np.ascontiguousarray(h[-1:]); hn=np.empty_like(last); lg=np.empty((1,cfg.vocab_size),np.float32)
        libs[0].niyah_rmsnorm(last.ctypes.data,fw.ctypes.data,hn.ctypes.data,1,4096,cfg.rms_norm_eps)
        head=st.mm('lm_head.weight')   # keep the mapping alive across the native call
        if libs[0].niyah_bf16_matmul(hn.ctypes.data,head.ctypes.data,lg.ctypes.data,1,cfg.vocab_size,4096): raise RuntimeError('lm_head')
        del head,layers; gc.collect()
        kv_vs_recompute='BIT_IDENTICAL' if np.array_equal(lg[0],lastN[0]) else 'DIFFERENT max_abs=%g'%float(np.max(np.abs(lg[0]-lastN[0])))
        print(json.dumps({'kv_vs_recompute':kv_vs_recompute,'single_vs_batch_identical':(None if same_single_vs_batch is None else bool(same_single_vs_batch))}),flush=True)
    ref_next,ref_s=reference_teacher_forced(st,cfg,ids,outN,rotary)
    rows=[]
    for p,(kind,_) in enumerate(GATE.PROMPTS):
        a,b=outN[p],ref_next[p]; div=next((i for i in range(NEW) if a[i]!=b[i]),None)
        refseq=a if div is None else a[:div]+[b[div]]
        rows.append({'kind':kind,'exact_64':div is None,'first_divergence':div,'native_ids':a,'reference_ids_to_divergence':refseq,
                     'native_text':tok.decode(a),'reference_text_to_divergence':tok.decode(refseq),
                     'teacher_forced_top1_agreement_64':sum(x==y for x,y in zip(a,b))/NEW})
    report.update({'native_1thread_1stream':perf1,'native_all_threads_10streams':perfN,'kv_vs_recompute_prompt0':kv_vs_recompute,
        'single_vs_batch_identical':(None if same_single_vs_batch is None else bool(same_single_vs_batch)),'reference_seconds':ref_s,
        'exact_match_prompts':sum(r['exact_64'] for r in rows),'per_prompt':rows,'peak_rss_mb':round(rss_mb(),1),
        'limitations':['Native weights (~7 GB INT8) exceed free RAM; decode speed includes page-cache misses from build/qwen_q8g32',
                       'All-threads figure is 10 concurrent streams (one per thread, lm_head via OpenMP), not one faster stream',
                       'Reference continuation after first divergence not computed (disk ~110 MB/s makes 64 streamed reference steps ~2.5 h)']})
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_prompt',)},indent=1),flush=True)
    for r in rows: print(json.dumps({k:r[k] for k in ('kind','exact_64','first_divergence')}),flush=True)
    if os.environ.get('NIYAH_REPORT'):
        with Path(os.environ['NIYAH_REPORT']).open('x',encoding='utf-8') as f: json.dump(report,f,indent=2,ensure_ascii=False)

if __name__=='__main__': main()
