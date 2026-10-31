#!/usr/bin/env python3
# qwen_native_full_gate.py — all 36 Qwen3 decoder layers + final RMSNorm + lm_head, native vs transformers FP32.
# Layer-streamed: for each layer, weights are read from safetensors (mmap), converted to FP32, run by the
# transformers reference and quantized for the native engine, then freed. All 36 FP32 layers are never held at once.
# Native C/asm comes unmodified from qwen_native_layer0_g32_tany.py; extra functions are appended below.
# NIYAH_G32_ALL=1: all seven linears use K=32 activation groups (generated from niyah_layer0 by exact substitution).
# Gate (fixed before measuring): 5 Arabic + 5 English prompts, 32 teacher-forced positions each.
#   PASS = next-token top-1 agreement >= 0.95 AND mean KL(ref||native) <= 0.03 nats.
#   top-5 overlap and per-layer rel L2 are information only.
import ctypes, gc, hashlib, importlib.util, json, os, resource, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('tany', HERE/'qwen_native_layer0_g32_tany.py')
TANY = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(TANY)

T = 32
TOP1_MIN, KL_MAX = 0.95, 0.03
PROMPTS = [
 ('ar_msa', 'تعد اللغة العربية من أقدم اللغات السامية الحية، وقد انتشرت مع توسع الدولة الإسلامية حتى أصبحت لغة العلم والأدب في مناطق واسعة من آسيا وأفريقيا لقرون طويلة.'),
 ('ar_msa', 'أعلنت وزارة الصحة اليوم عن إطلاق حملة وطنية للتطعيم ضد الإنفلونزا الموسمية، وتستهدف الحملة كبار السن والأطفال والعاملين في القطاع الصحي في جميع المناطق.'),
 ('ar_najdi', 'يا خوي وش رايك نطلع البر الليلة؟ الجو زين والقمرا طالعة، بس لا تنسى تجيب الحطب والدلة، وقل لأبوك يجي معنا إذا فاضي، طال عمره.'),
 ('ar_najdi', 'أمس رحت السوق أدور لي ثوب جديد للعيد، لقيت واحد عاجبني بس سعره غالي حيل، قعدت أفاصل البياع ساعة كاملة لين نزل لي السعر شوي.'),
 ('ar_msa', 'يعتمد التعلم الآلي على بناء نماذج رياضية تتعلم من البيانات، ثم تستخدم هذه النماذج للتنبؤ بنتائج جديدة دون أن تبرمج بشكل صريح لكل حالة على حدة.'),
 ('en', 'The old lighthouse keeper climbed the spiral staircase every evening at dusk, carrying a small brass lantern and a notebook in which he recorded the weather, the passing ships, and the moods of the sea.'),
 ('en', 'In 1905, Albert Einstein published four papers that changed physics: one on the photoelectric effect, one on Brownian motion, one introducing special relativity, and a short note deriving the equivalence of mass and energy.'),
 ('en', 'To make a simple tomato sauce, warm olive oil in a heavy pan, add sliced garlic and cook it gently until fragrant, then pour in crushed tomatoes, season with salt, and let it simmer for twenty minutes.'),
 ('en', 'The committee reviewed the budget proposal on Tuesday and asked the finance team to explain why maintenance costs had risen by nearly thirty percent compared with the previous fiscal year, and to prepare a revised plan before the next board meeting in March.'),
 ('en', 'def binary_search(items, target):\n    low, high = 0, len(items) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if items[mid] == target:\n            return mid\n'),
]

EXTRA_C = r'''
#include <immintrin.h>
static int linear_g32_any(const float *x,float *y,const Matrix *m,int t,int8_t *aq,int32_t *acc){
 const int group=32;
 if(!x||!y||!m||!aq||!acc||!m->w||!m->s||t<1||t>4096||m->cols%group)return 64;
 const int groups=m->cols/group,rows=m->rows;
 double *scales=malloc((size_t)t*groups*sizeof(double));if(!scales)return 70;
 for(int g=0;g<groups;++g)for(int i=0;i<t;++i){
  int rc=quant(x+(int64_t)i*m->cols+(int64_t)g*group,aq+((int64_t)g*t+i)*group,group,&scales[(int64_t)i*groups+g]);
  if(rc){free(scales);return rc;}}
 double *tot=calloc((size_t)t*rows,sizeof(double));
 if(!tot){free(scales);return 70;}
 for(int g=0;g<groups;++g){
  if(niyah_i8_gemm(aq+(int64_t)g*t*group,m->w+(int64_t)g*rows*group,acc,t,rows,group)){free(tot);free(scales);return 64;}
  for(int i=0;i<t;++i){const double sc=scales[(int64_t)i*groups+g];double *o=tot+(int64_t)i*rows;const int32_t *a=acc+(int64_t)i*rows;
   for(int j=0;j<rows;++j)o[j]+=(double)a[j]*sc;}}
 for(int i=0;i<t;++i)for(int j=0;j<rows;++j){float v=(float)(tot[(int64_t)i*rows+j]*(double)m->s[j]);
  if(!isfinite(v)){free(tot);free(scales);return 65;}y[(int64_t)i*rows+j]=v;}
 free(tot);free(scales);return 0;
}
int niyah_rmsnorm(const float *x,const float *w,float *y,int rows,int cols,float eps){
 if(!x||!w||!y||rows<1||cols<1||!(eps>0))return 64;
 return norm(x,w,y,rows,cols,eps);}
__attribute__((target("avx2,fma"))) static float dot_f32(const float *a,const float *b,int n){
 __m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps(),s2=_mm256_setzero_ps(),s3=_mm256_setzero_ps();
 for(int k=0;k<n;k+=32){
  s0=_mm256_fmadd_ps(_mm256_loadu_ps(a+k),_mm256_loadu_ps(b+k),s0);
  s1=_mm256_fmadd_ps(_mm256_loadu_ps(a+k+8),_mm256_loadu_ps(b+k+8),s1);
  s2=_mm256_fmadd_ps(_mm256_loadu_ps(a+k+16),_mm256_loadu_ps(b+k+16),s2);
  s3=_mm256_fmadd_ps(_mm256_loadu_ps(a+k+24),_mm256_loadu_ps(b+k+24),s3);}
 s0=_mm256_add_ps(_mm256_add_ps(s0,s1),_mm256_add_ps(s2,s3));
 float lane[8];_mm256_storeu_ps(lane,s0);double r=0;for(int i=0;i<8;++i)r+=lane[i];return (float)r;
}
/* FP32 lm_head chunk: y[i*ystride+j] = x[i,:] . w[j,:]; rows split across OpenMP threads (per-output order fixed). */
int niyah_f32_matmul(const float *x,const float *w,float *y,int t,int rows,int cols,int ystride){
 if(!available())return 77;
 if(!x||!w||!y||t<1||rows<1||cols<32||cols%32||ystride<rows)return 64;
 #pragma omp parallel for schedule(static)
 for(int j=0;j<rows;++j){const float *wr=w+(int64_t)j*cols;
  for(int i=0;i<t;++i)y[(int64_t)i*ystride+j]=dot_f32(x+(int64_t)i*cols,wr,cols);}
 return 0;
}
'''

def g32all_layer_source(c):
    body = c[c.index('int niyah_layer0('):]
    if body.count('CHECK(linear(') != 6 or body.count('CHECK(linear_g32_blocked(') != 1:
        raise SystemExit('STOP: unexpected niyah_layer0 body')
    body = body.replace('int niyah_layer0(', 'int niyah_layer_g32all(', 1)
    body = body.replace('CHECK(linear(', 'CHECK(linear_g32_any(').replace('CHECK(linear_g32_blocked(', 'CHECK(linear_g32_any(')
    return body.replace('#define CHECK', '#undef CHECK\n #define CHECK', 1)

class Matrix(ctypes.Structure):
    _fields_=[('w',ctypes.c_void_p),('s',ctypes.c_void_p),('rows',ctypes.c_int),('cols',ctypes.c_int)]

NAMES = ['self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
         'self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']

def quant_rows(w):
    q=np.empty(w.shape,np.int8); s=np.empty((len(w),1),np.float64)
    for b in range(0,len(w),1024):
        e=min(b+1024,len(w)); q[b:e],s[b:e]=TANY.quant(w[b:e])
    return q,s

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def main():
    g32all = os.environ.get('NIYAH_G32_ALL') == '1'
    torch.set_num_threads(os.cpu_count() or 1)
    model=Path(os.environ.get('NIYAH_MODEL_DIR',str(Path.home()/'models/Qwen3-8B-Base'))).expanduser().resolve()
    cfg=AutoConfig.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    if (cfg.hidden_size,cfg.intermediate_size,cfg.num_attention_heads,cfg.num_key_value_heads,cfg.head_dim)!=(4096,12288,32,8,128):
        raise SystemExit('STOP: unsupported configuration')
    if getattr(cfg,'use_sliding_window',False) or (getattr(cfg,'layer_types',None) and set(cfg.layer_types)!={'full_attention'}):
        raise SystemExit('STOP: non-full attention layers')
    cfg._attn_implementation='eager'
    L=cfg.num_hidden_layers
    mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    def read(name,sel=slice(None)):
        with safe_open(str(model/mapping[name]),framework='pt',device='cpu') as f:
            return f.get_slice(name)[sel].float()
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    ids=[]
    for kind,text in PROMPTS:
        x=tok(text,add_special_tokens=False)['input_ids']
        if len(x)<T+1: raise SystemExit('STOP: prompt too short: '+kind+' '+str(len(x)))
        ids.append(x[:T])
    P=len(ids)
    report={'scope':'Qwen3-8B-Base all %d layers + final RMSNorm + lm_head, teacher-forced, %d prompts x %d positions'%(L,P,T),
            'mode':'g32 all seven linears' if g32all else 'per-token activations; down_proj g32',
            'gate':{'top1_min':TOP1_MIN,'mean_kl_max_nats':KL_MAX,'info_only':['top5_overlap','per_layer_rel_l2']},
            'reference':'transformers Qwen3DecoderLayer FP32 eager, layer-streamed; not model.forward (FP32 model does not fit in RAM)',
            'prompts':[{'kind':k,'text':t,'ids':i} for (k,t),i in zip(PROMPTS,ids)],
            'model_dir':str(model),'config_sha256':hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'tany_sha256':hashlib.sha256((HERE/'qwen_native_layer0_g32_tany.py').read_bytes()).hexdigest(),
            'torch_version':torch.__version__,'threads_native_layers':1,'threads_lm_head':os.cpu_count()}
    with tempfile.TemporaryDirectory(prefix='qwen-native-full-') as tmp:
        work=Path(tmp)
        gemm_src=(Path.home()/'niyah-assembly-model/src/niyah_i8_gemm.s').read_text()
        (work/'kernels.s').write_text(TANY.ASM+gemm_src.replace('.section .note.GNU-stack, "", @progbits\n',''))
        csrc=TANY.C_SOURCE+EXTRA_C+'\n'+g32all_layer_source(TANY.C_SOURCE)
        (work/'layer.c').write_text(csrc)
        report['native_source_sha256']=hashlib.sha256((TANY.ASM+gemm_src+csrc).encode()).hexdigest()
        subprocess.run(['gcc','-std=c11','-O2','-Wall','-Wextra','-Werror','-fPIC','-shared','-fopenmp',
            '-fno-fast-math','-ffp-contract=off','-Wl,-z,defs,-z,noexecstack',
            '-o',str(work/'layer.so'),str(work/'layer.c'),str(work/'kernels.s'),'-lm'],check=True,timeout=120)
        lib=ctypes.CDLL(str(work/'layer.so'))
        if not lib.available(): raise SystemExit('STOP: AVX2 unavailable')
        layer_fn=lib.niyah_layer_g32all if g32all else lib.niyah_layer0
        layer_fn.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int,ctypes.c_float]; layer_fn.restype=ctypes.c_int
        lib.niyah_rmsnorm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_float]; lib.niyah_rmsnorm.restype=ctypes.c_int
        lib.niyah_f32_matmul.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4; lib.niyah_f32_matmul.restype=ctypes.c_int

        emb=[torch.cat([read('model.embed_tokens.weight',slice(i,i+1)) for i in x]).numpy() for x in ids]
        nat=[np.ascontiguousarray(e) for e in emb]
        ref=torch.from_numpy(np.stack(emb))
        rotary=Qwen3RotaryEmbedding(config=cfg)
        with torch.inference_mode():
            co,si=rotary(ref[:1],torch.arange(T).unsqueeze(0))
        cos_np=np.ascontiguousarray(co[0].numpy()); sin_np=np.ascontiguousarray(si[0].numpy())
        mask=torch.full((1,1,T,T),float('-inf')).triu(1)
        curve=[]; t_native=0.0; t_ref=0.0; t_load=0.0
        for l in range(L):
            t0=time.perf_counter()
            with torch.device('meta'):
                layer=Qwen3DecoderLayer(cfg,layer_idx=l)
            state={k:read('model.layers.%d.%s'%(l,k)) for k in layer.state_dict()}
            layer.load_state_dict(state,strict=True,assign=True); layer.eval()
            packed=[];shapes=[]
            for name in NAMES:
                q,s=quant_rows(state[name].numpy()); shapes.append(q.shape)
                if g32all or name=='mlp.down_proj.weight':
                    q=np.ascontiguousarray(q.reshape(q.shape[0],-1,32).transpose(1,0,2))
                packed.append((q,s))
            mats=(Matrix*7)(*[Matrix(q.ctypes.data,s.ctypes.data,*sh) for (q,s),sh in zip(packed,shapes)])
            norms=[np.ascontiguousarray(state[k].numpy()) for k in ('input_layernorm.weight','post_attention_layernorm.weight','self_attn.q_norm.weight','self_attn.k_norm.weight')]
            t1=time.perf_counter(); t_load+=t1-t0
            for p in range(P):
                out=np.empty_like(nat[p])
                rc=layer_fn(nat[p].ctypes.data,out.ctypes.data,ctypes.addressof(mats),*[a.ctypes.data for a in norms],
                            cos_np.ctypes.data,sin_np.ctypes.data,T,cfg.rms_norm_eps)
                if rc: raise RuntimeError('native layer %d prompt %d rc=%d'%(l,p,rc))
                if not np.isfinite(out).all(): raise RuntimeError('nonfinite native layer %d'%l)
                nat[p]=out
            t2=time.perf_counter(); t_native+=t2-t1
            with torch.inference_mode():
                v=layer(ref,attention_mask=mask,position_ids=torch.arange(T).unsqueeze(0),
                        position_embeddings=(co,si),use_cache=False)
                ref=(v[0] if isinstance(v,tuple) else v).contiguous()
            t_ref+=time.perf_counter()-t2
            r=ref.numpy().astype(np.float64); n=np.stack(nat).astype(np.float64)
            curve.append(float(np.linalg.norm(n-r)/np.linalg.norm(r)))
            print(json.dumps({'layer':l,'rel_l2':curve[-1],'rss_peak_mb':round(rss_mb(),1),'elapsed_s':round(t_load+t_native+t_ref,1)}),flush=True)
            del layer,state,packed,mats,norms; gc.collect()
        fw=np.ascontiguousarray(read('model.norm.weight').numpy())
        hn=np.empty((P*T,4096),np.float32); hs=np.ascontiguousarray(np.concatenate(nat))
        if lib.niyah_rmsnorm(hs.ctypes.data,fw.ctypes.data,hn.ctypes.data,P*T,4096,cfg.rms_norm_eps): raise RuntimeError('native norm')
        with torch.inference_mode():
            rv=ref.reshape(P*T,4096); rn=(rv*torch.rsqrt(rv.pow(2).mean(-1,keepdim=True)+cfg.rms_norm_eps))*torch.from_numpy(fw)
        V=cfg.vocab_size; ln=np.empty((P*T,V),np.float32); lr=torch.empty((P*T,V),dtype=torch.float32)
        t3=time.perf_counter(); t_head_native=0.0
        for b in range(0,V,16384):
            e=min(b+16384,V); W=np.ascontiguousarray(read('lm_head.weight',slice(b,e)).numpy())
            s0=time.perf_counter()
            if lib.niyah_f32_matmul(hn.ctypes.data,W.ctypes.data,ln[:,b:].ctypes.data,P*T,e-b,4096,V): raise RuntimeError('native lm_head')
            t_head_native+=time.perf_counter()-s0
            with torch.inference_mode(): lr[:,b:e]=rn@torch.from_numpy(W).T
            del W
        if not np.isfinite(ln).all(): raise RuntimeError('nonfinite native logits')
        lr=lr.double(); lnt=torch.from_numpy(ln).double()
        lpr=torch.log_softmax(lr,-1); lpn=torch.log_softmax(lnt,-1)
        kl=(lpr.exp()*(lpr-lpn)).sum(-1)
        top1=(lr.argmax(-1)==lnt.argmax(-1)).double()
        t5r=lr.topk(5,-1).indices; t5n=lnt.topk(5,-1).indices
        ov=torch.tensor([len(set(a.tolist())&set(b.tolist()))/5 for a,b in zip(t5r,t5n)],dtype=torch.float64)
        res={'top1_agreement':float(top1.mean()),'mean_kl_nats':float(kl.mean()),'max_kl_nats':float(kl.max()),
             'top5_overlap_mean':float(ov.mean()),'final_norm_rel_l2':float(np.linalg.norm(hn.astype(np.float64)-rn.numpy())/np.linalg.norm(rn.numpy())),
             'per_prompt':[{'kind':PROMPTS[p][0],'top1':float(top1[p*T:(p+1)*T].mean()),'mean_kl':float(kl[p*T:(p+1)*T].mean())} for p in range(P)]}
        res['gate']='PASS' if res['top1_agreement']>=TOP1_MIN and res['mean_kl_nats']<=KL_MAX else 'FAIL'
        report.update({'result':res,'per_layer_rel_l2':curve,'peak_rss_mb':round(rss_mb(),1),
            'seconds':{'load_and_quantize':round(t_load,1),'native_layers_1thread':round(t_native,1),'reference_layers':round(t_ref,1),
                       'native_lm_head_all_threads':round(t_head_native,1),'lm_head_total':round(time.perf_counter()-t3,1)},
            'limitations':['Reference is layer-streamed HF decoder layers, not one model.forward call','Teacher-forced only; no KV cache, no generation',
                           'Weights per output row INT8; activations INT8 (per token, g32 where stated); lm_head FP32',
                           'Official Hugging Face SHA-256 of the weights not verified']})
        print(json.dumps(report['result'],indent=1),flush=True)
        print(json.dumps({'peak_rss_mb':report['peak_rss_mb'],'seconds':report['seconds']}),flush=True)
        if os.environ.get('NIYAH_REPORT'):
            with Path(os.environ['NIYAH_REPORT']).open('x',encoding='utf-8') as f: json.dump(report,f,indent=2)

if __name__=='__main__': main()
