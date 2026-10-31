.intel_syntax noprefix
.text
.global niyah_i8_dot
.type niyah_i8_dot,@function
niyah_i8_dot:
 vpxor ymm2,ymm2,ymm2
 xor ecx,ecx
.Li8:
 vpmovsxbw ymm0,XMMWORD PTR [rdi+rcx]
 vpmovsxbw ymm1,XMMWORD PTR [rsi+rcx]
 vpmaddwd ymm0,ymm0,ymm1
 vpaddd ymm2,ymm2,ymm0
 add ecx,16
 cmp ecx,edx
 jb .Li8
 vextracti128 xmm1,ymm2,1
 vpaddd xmm0,xmm2,xmm1
 vpshufd xmm1,xmm0,0x4e
 vpaddd xmm0,xmm0,xmm1
 vpshufd xmm1,xmm0,0xb1
 vpaddd xmm0,xmm0,xmm1
 vmovd eax,xmm0
 vzeroupper
 ret
.size niyah_i8_dot,.-niyah_i8_dot
.global niyah_dot128
.type niyah_dot128,@function
niyah_dot128:
 pxor xmm0,xmm0
 xor eax,eax
.Lfp:
 movss xmm1,DWORD PTR [rdi+rax*4]
 movss xmm2,DWORD PTR [rsi+rax*4]
 cvtss2sd xmm1,xmm1
 cvtss2sd xmm2,xmm2
 mulsd xmm1,xmm2
 addsd xmm0,xmm1
 inc eax
 cmp eax,128
 jb .Lfp
 ret
.size niyah_dot128,.-niyah_dot128
.section .note.GNU-stack,"",@progbits
