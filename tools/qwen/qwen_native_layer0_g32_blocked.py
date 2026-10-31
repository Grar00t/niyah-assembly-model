#!/usr/bin/env python3
import ctypes, hashlib, json, os, statistics, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

ASM = '.intel_syntax noprefix\n.text\n.global niyah_i8_dot\n.type niyah_i8_dot,@function\nniyah_i8_dot:\n vpxor ymm2,ymm2,ymm2\n xor ecx,ecx\n.Li8:\n vpmovsxbw ymm0,XMMWORD PTR [rdi+rcx]\n vpmovsxbw ymm1,XMMWORD PTR [rsi+rcx]\n vpmaddwd ymm0,ymm0,ymm1\n vpaddd ymm2,ymm2,ymm0\n add ecx,16\n cmp ecx,edx\n jb .Li8\n vextracti128 xmm1,ymm2,1\n vpaddd xmm0,xmm2,xmm1\n vpshufd xmm1,xmm0,0x4e\n vpaddd xmm0,xmm0,xmm1\n vpshufd xmm1,xmm0,0xb1\n vpaddd xmm0,xmm0,xmm1\n vmovd eax,xmm0\n vzeroupper\n ret\n.size niyah_i8_dot,.-niyah_i8_dot\n.global niyah_dot128\n.type niyah_dot128,@function\nniyah_dot128:\n pxor xmm0,xmm0\n xor eax,eax\n.Lfp:\n movss xmm1,DWORD PTR [rdi+rax*4]\n movss xmm2,DWORD PTR [rsi+rax*4]\n cvtss2sd xmm1,xmm1\n cvtss2sd xmm2,xmm2\n mulsd xmm1,xmm2\n addsd xmm0,xmm1\n inc eax\n cmp eax,128\n jb .Lfp\n ret\n.size niyah_dot128,.-niyah_dot128\n.section .note.GNU-stack,"",@progbits\n'

C_SOURCE = r'''#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <fenv.h>
#include <string.h>
typedef struct {const int8_t *w;const double *s;int rows,cols;} Matrix;
extern int32_t niyah_i8_dot(const int8_t *,const int8_t *,int);
extern double niyah_dot128(const float *,const float *);
extern int niyah_i8_gemm(const int8_t *,const int8_t *,int32_t *,int,int,int);

int available(void){__builtin_cpu_init();return !!__builtin_cpu_supports("avx2");}

static int quant(const float *x,int8_t *q,int n,double *s){
 double m=0;
 for(int i=0;i<n;++i){if(!isfinite(x[i]))return 65;double a=fabs((double)x[i]);if(a>m)m=a;}
 *s=m?(m/127.0):1.0;
 if(!(*s>0)||!isfinite(*s))return 65;
 for(int i=0;i<n;++i){double v=nearbyint((double)x[i]/(double)*s);q[i]=(int8_t)fmax(-127.0,fmin(127.0,v));}
 return 0;
}

static int linear(const float *x,float *y,const Matrix *m,int t,int8_t *aq,int32_t *acc){
 double scale[16];
 for(int i=0;i<t;++i){int rc=quant(x+(int64_t)i*m->cols,aq+(int64_t)i*m->cols,m->cols,&scale[i]);if(rc)return rc;}
 if(niyah_i8_gemm(aq,m->w,acc,t,m->rows,m->cols))return 64;
 for(int i=0;i<t;++i)for(int j=0;j<m->rows;++j){float f=(float)((double)acc[(int64_t)i*m->rows+j]*(double)scale[i]*(double)m->s[j]);if(!isfinite(f))return 65;y[(int64_t)i*m->rows+j]=f;}
 return 0;
}

/* down_proj g32, blocked: weights are group-major [g][rows][32]; activations packed [g][t][32].
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

static int norm(const float *x,const float *w,float *y,int rows,int cols,float eps){
 for(int r=0;r<rows;++r){const float *p=x+(int64_t)r*cols;float *d=y+(int64_t)r*cols;double sum=0;
  for(int j=0;j<cols;++j){if(!isfinite(p[j]))return 65;sum+=(double)p[j]*(double)p[j];}
  double inv=1.0/sqrt(sum/cols+(double)eps);
  for(int j=0;j<cols;++j)d[j]=(float)((double)p[j]*inv*(double)w[j]);}
 return 0;
}

static void rope(float *x,const float *c,const float *s,int t,int heads){
 for(int i=0;i<t;++i)for(int h=0;h<heads;++h){float *p=x+((int64_t)i*heads+h)*128;
  for(int j=0;j<64;++j){float a=p[j],b=p[j+64],co=c[i*128+j],si=s[i*128+j];
   p[j]=a*co-b*si;p[j+64]=b*co+a*si;}}
}

static int attention(const float *q,const float *k,const float *v,float *y,int t){
 const double scale=1.0/sqrt(128.0);
 for(int i=0;i<t;++i)for(int h=0;h<32;++h){double acc[128]={0},maximum=-INFINITY,total=0;
  const float *query=q+((int64_t)i*32+h)*128;
  for(int j=0;j<=i;++j){const float *key=k+((int64_t)j*8+h/4)*128;const float *value=v+((int64_t)j*8+h/4)*128;
   double score=niyah_dot128(query,key)*scale,next=fmax(maximum,score),a=exp(maximum-next),b=exp(score-next);
   total=total*a+b;for(int d=0;d<128;++d)acc[d]=acc[d]*a+b*value[d];maximum=next;}
  if(!(total>0)||!isfinite(total))return 65;
  for(int d=0;d<128;++d)y[((int64_t)i*32+h)*128+d]=(float)(acc[d]/total);}
 return 0;
}

/* Matrix order: q,k,v,o,gate,up,down. Normalization weights remain FP32. Linear int8 products use niyah_i8_gemm. */
int niyah_layer0(const float *x,float *y,const Matrix *m,
 const float *inorm,const float *pnorm,const float *qn,const float *kn,
 const float *cosine,const float *sine,int t,float eps){
 if(!available())return 77;
 if(!x||!y||!m||!inorm||!pnorm||!qn||!kn||!cosine||!sine||t<1||t>16||!(eps>0)||!isfinite(eps))return 64;
 if(fegetround()!=FE_TONEAREST)return 65;
 const int rows[7]={4096,1024,1024,4096,12288,12288,4096};
 const int cols[7]={4096,4096,4096,4096,4096,4096,12288};
 for(int i=0;i<7;++i){if(m[i].rows!=rows[i]||m[i].cols!=cols[i]||!m[i].w||!m[i].s)return 64;
  for(int j=0;j<rows[i];++j)if(!(m[i].s[j]>0)||!isfinite(m[i].s[j]))return 65;}
 size_t n=(size_t)t*4096,nk=(size_t)t*1024,nm=(size_t)t*12288;
 float *memory=calloc(6*n+2*nk+3*nm,sizeof(float));int8_t *aq=malloc((size_t)t*12288);int32_t *acc=malloc((size_t)t*12288*sizeof(int32_t));
 if(!memory||!aq||!acc){free(memory);free(aq);free(acc);return 70;}
 float *h=memory,*q=h+n,*k=q+n,*v=k+nk,*a=v+nk,*p=a+n,*res=p+n,*g=res+n,*u=g+nm,*z=u+nm;
 int rc=0;
 #define CHECK(expr) do{rc=(expr);if(rc)goto cleanup;}while(0)
 CHECK(norm(x,inorm,h,t,4096,eps));
 CHECK(linear(h,q,m+0,t,aq,acc));CHECK(linear(h,k,m+1,t,aq,acc));CHECK(linear(h,v,m+2,t,aq,acc));
 CHECK(norm(q,qn,q,t*32,128,eps));CHECK(norm(k,kn,k,t*8,128,eps));
 rope(q,cosine,sine,t,32);rope(k,cosine,sine,t,8);
 CHECK(attention(q,k,v,a,t));CHECK(linear(a,p,m+3,t,aq,acc));
 for(size_t i=0;i<n;++i)res[i]=x[i]+p[i];
 CHECK(norm(res,pnorm,h,t,4096,eps));
 CHECK(linear(h,g,m+4,t,aq,acc));CHECK(linear(h,u,m+5,t,aq,acc));
 for(size_t i=0;i<nm;++i){double d=g[i],sig;if(d>=0)sig=1.0/(1.0+exp(-d));else{double e=exp(d);sig=e/(1.0+e);}z[i]=(float)(d*sig*(double)u[i]);}
 CHECK(linear_g32_blocked(z,p,m+6,t,aq,acc));
 for(size_t i=0;i<n;++i){y[i]=res[i]+p[i];if(!isfinite(y[i])){rc=65;goto cleanup;}}
cleanup:
 free(memory);free(aq);free(acc);return rc;
}
'''

def digest(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()

class Matrix(ctypes.Structure):
    _fields_=[('w',ctypes.c_void_p),('s',ctypes.c_void_p),('rows',ctypes.c_int),('cols',ctypes.c_int)]

def quant(a):
    a=np.ascontiguousarray(a,dtype=np.float32)
    m=np.max(np.abs(a.astype(np.float64)),axis=-1,keepdims=True)
    s=np.where(m==0,1.0,m/127.0).astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(s).all() or not (s>0).all():
        raise RuntimeError('Invalid quantization input')
    q=np.clip(np.rint(a.astype(np.float64)/s.astype(np.float64)),-127,127).astype(np.int8)
    return np.ascontiguousarray(q),np.ascontiguousarray(s)

def main():
    torch.set_num_threads(1)
    model=Path(os.environ.get(
        'NIYAH_MODEL_DIR',
        str(Path.home()/'models/Qwen3-8B-Base')
    )).expanduser().resolve()
    cfg=AutoConfig.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    expected=(4096,12288,32,8,128)
    actual=(cfg.hidden_size,cfg.intermediate_size,cfg.num_attention_heads,cfg.num_key_value_heads,cfg.head_dim)
    if actual!=expected or cfg.hidden_act!='silu' or getattr(cfg,'attention_bias',False):
        raise SystemExit('STOP: unsupported configuration')
    if getattr(cfg,'layer_types',None) and cfg.layer_types[0]!='full_attention':
        raise SystemExit('STOP: layer 0 is not full causal attention')
    cfg._attn_implementation='eager'
    index=model/'model.safetensors.index.json'
    mapping=json.loads(index.read_text())['weight_map'] if index.is_file() else {}
    def read(name,selection=slice(None)):
        paths=[model/mapping[name]] if name in mapping else sorted(model.glob('*.safetensors'))
        for path in paths:
            with safe_open(str(path),framework='pt',device='cpu') as f:
                if name in f.keys():return f.get_slice(name)[selection].float()
        raise RuntimeError('Missing tensor: '+name)
    with tempfile.TemporaryDirectory(prefix='qwen-native-layer0-') as tmp:
        work=Path(tmp)
        gemm_src=(Path.home()/'niyah-assembly-model/src/niyah_i8_gemm.s').read_text()
        if 'niyah_i8_gemm:' not in gemm_src: raise SystemExit('STOP: blocked GEMM source missing')
        gemm_link=gemm_src.replace('.section .note.GNU-stack, "", @progbits\n','')
        (work/'kernels.s').write_text(ASM+gemm_link)
        (work/'layer.c').write_text(C_SOURCE)
        subprocess.run(['gcc','-std=c11','-O2','-Wall','-Wextra','-Werror','-fPIC','-shared',
            '-fno-fast-math','-ffp-contract=off','-Wl,-z,defs,-z,noexecstack',
            '-o',str(work/'layer.so'),str(work/'layer.c'),str(work/'kernels.s'),'-lm'],check=True,timeout=90)
        lib=ctypes.CDLL(str(work/'layer.so'))
        lib.available.argtypes=[];lib.available.restype=ctypes.c_int
        if not lib.available():raise SystemExit('STOP: AVX2 unavailable')
        dot=lib.niyah_i8_dot
        dot.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int];dot.restype=ctypes.c_int32
        for n in (16,4096,12288):
            for a,b in ((-128,-128),(-128,127),(127,127)):
                x=np.full(n,a,np.int8);y=np.full(n,b,np.int8)
                if dot(x.ctypes.data,y.ctypes.data,n)!=n*a*b:raise RuntimeError('Integer dot failed')
        print('AVX2_INTEGER_BOUNDARIES: PASS',flush=True)
        gemm=lib.niyah_i8_gemm
        gemm.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        gemm.restype=ctypes.c_int
        rng=np.random.default_rng(20261029)
        for M,N,K in ((1,1,16),(3,5,32),(7,6,48),(8,4,64),(9,6,80),(16,17,128)):
            A=np.ascontiguousarray(rng.integers(-128,128,size=(M,K),dtype=np.int8))
            B=np.ascontiguousarray(rng.integers(-128,128,size=(N,K),dtype=np.int8))
            C=np.empty((M,N),np.int32)
            if gemm(A.ctypes.data,B.ctypes.data,C.ctypes.data,M,N,K)!=0: raise RuntimeError('gemm rejected '+str((M,N,K)))
            for i in range(M):
                for j in range(N):
                    if int(C[i,j])!=dot(A[i].ctypes.data,B[j].ctypes.data,K):
                        raise RuntimeError('gemm mismatch '+str((M,N,K,i,j)))
        A=np.full((4,32),-128,np.int8); B=np.full((3,32),127,np.int8); C=np.empty((4,3),np.int32)
        if gemm(A.ctypes.data,B.ctypes.data,C.ctypes.data,4,3,32)!=0: raise RuntimeError('boundary gemm rejected')
        if int(C[0,0])!=32*(-128)*127: raise RuntimeError('boundary gemm value')
        if gemm(A.ctypes.data,B.ctypes.data,C.ctypes.data,0,3,32)!=64: raise RuntimeError('gemm geometry')
        print('AVX2_GEMM_MATCHES_DOT: PASS',flush=True)
        with torch.device('meta'):
            layer=Qwen3DecoderLayer(cfg,layer_idx=0)
        state={}
        for key in layer.state_dict():
            print('READING:',key,flush=True)
            state[key]=read('model.layers.0.'+key)
        layer.load_state_dict(state,strict=True,assign=True)
        layer.eval()
        if any(p.is_meta for p in layer.parameters()):raise RuntimeError('Unloaded weights')
        names=['self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
               'self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']
        packed=[];shapes=[]
        for name in names:
            w=state[name].numpy();q=np.empty(w.shape,np.int8);s=np.empty((len(w),1),np.float64)
            for begin in range(0,len(w),128):
                end=min(begin+128,len(w));q[begin:end],s[begin:end]=quant(w[begin:end])
            shapes.append(q.shape)
            if name=='mlp.down_proj.weight':q=np.ascontiguousarray(q.reshape(q.shape[0],-1,32).transpose(1,0,2))
            packed.append((q,s))
            print('QUANTIZED:',name,shapes[-1],'group-major g32' if name=='mlp.down_proj.weight' else '',flush=True)
        matrices=(Matrix*7)(*[Matrix(q.ctypes.data,s.ctypes.data,*sh) for (q,s),sh in zip(packed,shapes)])
        norms=[np.ascontiguousarray(state[k].numpy()) for k in (
            'input_layernorm.weight','post_attention_layernorm.weight','self_attn.q_norm.weight','self_attn.k_norm.weight')]
        original_hashes=[digest(a) for pair in packed for a in pair]+[digest(a) for a in norms]
        fn=lib.niyah_layer0
        fn.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int,ctypes.c_float];fn.restype=ctypes.c_int
        tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
        text='نريد تنفيذ حساب حقيقي من نموذج كوين باستخدام الأوزان الأصلية ومقارنة نتيجة الاسمبلي بمرجع مستقل قبل دمجها في التشغيل الكامل.'
        ids=tok(text,add_special_tokens=False)['input_ids'][:16]
        if len(ids)!=16:raise RuntimeError('Insufficient tokens')
        embeddings=torch.cat([read('model.embed_tokens.weight',slice(i,i+1)) for i in ids])
        rotary=Qwen3RotaryEmbedding(config=cfg)
        report={'scope':'Full decoder layer 0 only; no model generation',
                'native':'Blocked AVX2 INT8 GEMM for six linears; down_proj runs niyah_i8_gemm per K=32 block (group-major weights) with C FP64 scaled accumulation; FP32 buffers',
                'quantization':'Per output-row weight scale; per-token activations except down_proj g32',
                'input_token_ids':ids,'tests':[],'quality_gate':'NOT_ESTABLISHED',
                'native_source_sha256':hashlib.sha256((ASM+gemm_src+C_SOURCE).encode()).hexdigest()}
        report.update({
            'model_dir':str(model),
            'script_sha256':hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            'config_sha256':hashlib.sha256(
                (model/'config.json').read_bytes()
            ).hexdigest(),
            'torch_version':torch.__version__,
        })
        for t in (1,7,16):
            x=np.ascontiguousarray(embeddings[:t].numpy());before=x.copy()
            with torch.inference_mode():
                co,si=rotary(torch.from_numpy(x).unsqueeze(0),torch.arange(t).unsqueeze(0))
            co=np.ascontiguousarray(co[0].numpy());si=np.ascontiguousarray(si[0].numpy())
            if co.shape!=(t,128) or not np.array_equal(co[:,:64],co[:,64:]) or not np.array_equal(si[:,:64],si[:,64:]):
                raise RuntimeError('Unsupported RoPE layout')
            guard=np.full(x.size+32,np.nan,np.float32);out=guard[16:-16].reshape(x.shape)
            def native():
                args=[x.ctypes.data,out.ctypes.data,ctypes.addressof(matrices)]+[a.ctypes.data for a in norms]+[co.ctypes.data,si.ctypes.data]
                rc=fn(*args,t,cfg.rms_norm_eps)
                if rc:raise RuntimeError('Native layer error: '+str(rc))
            mask=torch.full((1,1,t,t),float('-inf')).triu(1)
            def reference():
                value=layer(torch.from_numpy(x).unsqueeze(0),attention_mask=mask,
                    position_ids=torch.arange(t).unsqueeze(0),position_embeddings=(torch.from_numpy(co).unsqueeze(0),torch.from_numpy(si).unsqueeze(0)),use_cache=False)
                if isinstance(value,tuple):value=value[0]
                return value[0].contiguous().numpy()
            native()
            with torch.inference_mode():ref=reference()
            if not np.isfinite(out).all():raise RuntimeError('Nonfinite output')
            if not np.isnan(guard[:16]).all() or not np.isnan(guard[-16:]).all():raise RuntimeError('Output guard overwritten')
            np.testing.assert_array_equal(x,before)
            difference=out.astype(np.float64)-ref.astype(np.float64)
            denom=np.linalg.norm(ref.astype(np.float64))
            row={'tokens':t,'relative_l2_vs_transformers_fp32':float(np.linalg.norm(difference)/denom) if denom else None,
                 'max_abs_vs_transformers_fp32':float(np.max(np.abs(difference))),'finite_and_output_guards':'PASS'}
            if t==16:
                baseline=out.copy();x[-1]+=np.float32(0.05);native()
                np.testing.assert_array_equal(out[:-1],baseline[:-1]);x[:]=before;native()
                row['future_token_isolation']='PASS'
                saved=out.copy();native();np.testing.assert_array_equal(out,saved)
                def bench(call):
                    call();values=[]
                    for _ in range(3):
                        start=time.perf_counter_ns();call();values.append((time.perf_counter_ns()-start)/1e6)
                    return statistics.median(values)
                with torch.inference_mode():
                    row['native_median_ms']=bench(native);row['torch_fp32_eager_cpu_median_ms']=bench(reference)
                row['torch_over_native_ratio']=row['torch_fp32_eager_cpu_median_ms']/row['native_median_ms']
            print(json.dumps(row),flush=True);report['tests'].append(row)
        current_hashes=[digest(a) for pair in packed for a in pair]+[digest(a) for a in norms]
        if current_hashes!=original_hashes:raise RuntimeError('Weights or scales modified')
        report['execution_status']='COMPLETED'
        report['weight_immutability']='PASS'
        report['limitations']=['Not full 36-layer inference','No KV cache','No VNNI or FlashAttention',
            'Quantized layer accuracy is measured, not certified','Native RMSNorm/RoPE use C, not earlier assembly functions','Activation quant and FP scale operations stay in C; down_proj g32 blocks are K=32 GEMM calls, not a fused kernel',
            'INT8 dots accumulate INT32; floating reductions use FP64','Benchmarks exclude loading and weight quantization',
            'Canaries cover tested output buffers, not proof of universal memory safety']
        print(json.dumps(report,indent=2),flush=True)
        if os.environ.get('NIYAH_REPORT'):
            dest=Path(os.environ['NIYAH_REPORT'])
            with dest.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
        print('Temporary native build removed on exit; model and project unchanged.',flush=True)

if __name__=='__main__':main()
