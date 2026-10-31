#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <fenv.h>
#include <limits.h>
extern int niyah_i8_gemm(const int8_t *,const int8_t *,int32_t *,int,int,int);
extern double niyah_dot128(const float *,const float *);
int available(void){
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
 __builtin_cpu_init(); return !!__builtin_cpu_supports("avx2");
#else
 return 0;
#endif
}
static uint32_t state=0x12345678u;
static uint32_t rnd(void){state^=state<<13;state^=state>>17;state^=state<<5;return state;}
static void die(const char *s){fprintf(stderr,"FAIL: %s\n",s);exit(1);}
static void *alloc(size_t n){void *p=malloc(n);if(!p)die("allocation");return p;}
static double now(void){struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))die("clock");return t.tv_sec+t.tv_nsec*1e-9;}
static volatile double sink;
static unsigned cases;
static int cmp(const void *a,const void *b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static void stats(const char *name,int m,int n,int k,double *v,int samples,int batch,double ops){
 qsort(v,(size_t)samples,sizeof(*v),cmp);
 double med=v[samples/2]/batch,lo=v[0]/batch,hi=v[samples-1]/batch;
 printf("%s,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g\n",name,m,n,k,samples,batch,lo*1e6,med*1e6,hi*1e6,ops/med/1e9);
}
static void gemm_case(int m,int n,int k,int mode){
 size_t na=(size_t)m*k,nb=(size_t)n*k,nc=(size_t)m*n;
 unsigned char *ab=alloc(na+64),*bb=alloc(nb+64),*cb=alloc(nc*4+128);
 int8_t *a=(int8_t*)ab+17,*b=(int8_t*)bb+19;
 int32_t *v=(int32_t*)(void*)(cb+64);
 int8_t *ac=alloc(na),*bc=alloc(nb);
 memset(cb,0xa5,nc*4+128);
 for(size_t i=0;i<na;i++)a[i]=mode==0?(int8_t)((int)(rnd()&255)-128):mode==1?-128:mode==2?127:0;
 for(size_t i=0;i<nb;i++)b[i]=mode==0?(int8_t)((int)(rnd()&255)-128):mode==1?-128:mode==2?-128:0;
 memcpy(ac,a,na);memcpy(bc,b,nb);
 if(niyah_i8_gemm(a,b,v,m,n,k))die("valid geometry rejected");
 for(int i=0;i<m;i++)for(int j=0;j<n;j++){
  int64_t ref=0;for(int x=0;x<k;x++)ref+=(int64_t)a[(size_t)i*k+x]*b[(size_t)j*k+x];
  if(ref!=v[(size_t)i*n+j]){fprintf(stderr,"M=%d N=%d K=%d i=%d j=%d\n",m,n,k,i,j);die("INT64 oracle mismatch");}
 }
 for(int i=0;i<64;i++)if(cb[i]!=0xa5||cb[64+nc*4+(size_t)i]!=0xa5)die("output canary");
 if(memcmp(a,ac,na)||memcmp(b,bc,nb))die("input mutated");
 int32_t *saved=alloc(nc*4);memcpy(saved,v,nc*4);
 if(niyah_i8_gemm(a,b,v,m,n,k)||memcmp(saved,v,nc*4))die("repeatability");
 free(saved);free(ac);free(bc);free(ab);free(bb);free(cb);
  cases++;
}
static void invalid_tests(void){
 const int shapes[][3]={{0,1,16},{-1,1,16},{1,0,16},{1,-1,16},{1,1,0},{1,1,-16},{1,1,15},{1,1,17},{1,1,16400},{100000,100000,16},{INT_MAX,1,16},{1,INT_MAX,16}};
 int8_t a[16]={0},b[16]={0};int32_t v;
 for(size_t i=0;i<sizeof(shapes)/sizeof(shapes[0]);i++){
  v=123456789;
  if(niyah_i8_gemm(a,b,&v,shapes[i][0],shapes[i][1],shapes[i][2])!=64||v!=123456789)die("invalid geometry contract");
  cases++;
 }
}
static void dot_tests(void){
 float a[130],b[130];
 for(int mode=0;mode<8;mode++)for(int trial=0;trial<64;trial++){
  long double ref=0,mag=0;
  for(int i=0;i<128;i++){
   float x=(float)((int)(rnd()%2049)-1024)/1024.0f;
   float y=(float)((int)(rnd()%2049)-1024)/1024.0f;
   if(mode==1)x=y=0;
   if(mode==2)x=1,y=(i&1)?-1:1;
   if(mode==3)x=ldexpf(x,-120),y=ldexpf(y,-120);
   if(mode==4)x=ldexpf(x,100),y=ldexpf(y,100);
   if(mode==5)x=(i==127)?1:0;
   if(mode==6)x=FLT_MIN,y=-FLT_MIN;
   if(mode==7)x=FLT_MAX,y=(i&1)?-FLT_MAX:FLT_MAX;
   a[i+1]=x;b[i+1]=y;long double p=(long double)x*y;ref+=p;mag+=fabsl(p);
  }
  float saved_a[128],saved_b[128];memcpy(saved_a,a+1,sizeof saved_a);memcpy(saved_b,b+1,sizeof saved_b);
  double got=niyah_dot128(a+1,b+1);
  long double tol=256.0L*DBL_EPSILON*mag+LDBL_MIN;
  if(!isfinite(got)||fabsl((long double)got-ref)>tol)die("dot128 long-double oracle");
  if(memcmp(saved_a,a+1,sizeof saved_a)||memcmp(saved_b,b+1,sizeof saved_b))die("dot inputs mutated");
  if(niyah_dot128(a+1,b+1)!=got)die("dot repeatability");
  cases++;
 }
 for(int i=0;i<128;i++)a[i]=b[i]=1;
 a[3]=NAN;if(!isnan(niyah_dot128(a,b)))die("NaN propagation");
  cases++;
 a[3]=INFINITY;if(!isinf(niyah_dot128(a,b)))die("Inf propagation");
  cases++;
}
static void bench_gemm(int m,int n,int k){
 size_t na=(size_t)m*k,nb=(size_t)n*k,nc=(size_t)m*n;
 int8_t *a=alloc(na),*b=alloc(nb);int32_t *v=alloc(nc*4);
 for(size_t i=0;i<na;i++)a[i]=(int8_t)((int)(rnd()&255)-128);
 for(size_t i=0;i<nb;i++)b[i]=(int8_t)((int)(rnd()&255)-128);
 if(niyah_i8_gemm(a,b,v,m,n,k))die("benchmark geometry");
 double t=now();if(niyah_i8_gemm(a,b,v,m,n,k))die("benchmark call");double elapsed=now()-t;
 int batch=(int)fmin(1000.0,fmax(1.0,0.01/fmax(elapsed,1e-9)));double samples[9];
 for(int s=0;s<9;s++){t=now();for(int r=0;r<batch;r++)if(niyah_i8_gemm(a,b,v,m,n,k))die("benchmark call");samples[s]=now()-t;sink+=v[0];}
 stats("gemm",m,n,k,samples,9,batch,2.0*m*n*k);free(a);free(b);free(v);
}
static void bench_dot(void){
 float a[128],b[128];for(int i=0;i<128;i++){a[i]=(float)i/128;b[i]=(float)(127-i)/128;}
 double values[9];int batch=10000;sink+=niyah_dot128(a,b);
 for(int s=0;s<9;s++){double t=now(),sum=0;for(int i=0;i<batch;i++)sum+=niyah_dot128(a,b);values[s]=now()-t;sink+=sum;}
 stats("dot128",1,1,128,values,9,batch,256);
}
int main(int argc,char **argv){
 int full=0,bench=1;
 for(int i=1;i<argc;i++){if(!strcmp(argv[i],"--full"))full=1;else if(!strcmp(argv[i],"--verify-only"))bench=0;else{fprintf(stderr,"usage: %s [--full] [--verify-only]\n",argv[0]);return 2;}}
 if(!available()){fprintf(stderr,"SKIP: AVX2 unavailable\n");return 77;}
 if(fesetround(FE_TONEAREST))die("rounding mode");
 invalid_tests();
 const int ms[]={1,2,3,4,5,7,8,9,12,15,16,17};
 const int ks[]={16,32,48,128,4096,12288,16384};
 for(size_t i=0;i<sizeof(ms)/sizeof(ms[0]);i++)for(size_t j=0;j<sizeof(ks)/sizeof(ks[0]);j++)for(int mode=0;mode<4;mode++)gemm_case(ms[i],5,ks[j],mode);
 for(int trial=0;trial<128;trial++)gemm_case(1+(int)(rnd()%20),1+(int)(rnd()%25),16*(1+(int)(rnd()%64)),0);
 dot_tests();
 if(full){const int dims[][2]={{4096,4096},{1024,4096},{12288,4096},{4096,12288}};for(int i=0;i<4;i++)for(int t=1;t<=16;t=(t==1?7:16)){gemm_case(t,dims[i][0],dims[i][1],0);if(t==16)break;}}
 fprintf(stderr,"PASS: %u verification cases; seed=0x12345678; serial only\n",cases);
 if(bench){puts("kernel,M,N,K,samples,batch,min_us,median_us,max_us,Gops_per_s");bench_dot();bench_gemm(1,256,4096);bench_gemm(7,256,4096);bench_gemm(16,256,12288);if(full){bench_gemm(16,4096,4096);bench_gemm(16,1024,4096);bench_gemm(16,12288,4096);bench_gemm(16,4096,12288);}}
 return 0;
}
