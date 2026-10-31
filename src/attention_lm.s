# Niyah: learnable one-head causal attention + residual ReLU FFN LM
# x86-64 GNU as, Linux kernel syscalls, no libc, no PyTorch, no runtime deps.
# Educational neural language model, NOT a large language model.
.intel_syntax noprefix
.equ D, 16
.equ CTX, 32
.equ V, 256
# AVX2_AUTO performs OSXSAVE+AVX+FMA+AVX2 capability dispatch at startup.
# AVX2_FAST is explicitly unsafe on unsupported CPUs; prefer AVX2_AUTO.
.ifdef AVX2_AUTO
.equ COMPILE_SIMD,1
.else
.ifdef AVX2_FAST
.equ COMPILE_SIMD,1
.else
.equ COMPILE_SIMD,0
.endif
.endif
.ifndef EPOCHS
.equ EPOCHS, 240
.endif
.ifndef SAMPLE_LEN
.equ SAMPLE_LEN, 120
.endif
.ifndef SEED
.equ SEED, 2463534242
.endif
.if EPOCHS < 0
.error "negative epoch count"
.endif
.if SAMPLE_LEN < 0
.error "negative sample count"
.endif
.if SEED == 0
.error "seed must be nonzero"
.endif
.section .rodata
train_bytes: .incbin "data/corpus.txt"
.equ N, . - train_bytes
.if N <= CTX
.error "training corpus too short"
.endif
valid_bytes: .incbin "data/heldout.txt"
.equ NV, . - valid_bytes
.if NV <= CTX
.error "validation corpus too short"
.endif
lr:      .float 0.025
scale:   .float 0.06
randdiv: .float 0.000030517578125
quarter: .float 0.25
one:     .float 1.0
small:   .float 1.0e-30
neg:     .float -3.0e+38
mil:     .float 1000.0
inv24:   .float 5.9604644775390625e-8
s_init:  .ascii "initial_train_loss_x1000 "
.equ l_init, . - s_init
s_final: .ascii "final_train_loss_x1000 "
.equ l_final, . - s_final
s_valid: .ascii "heldout_loss_x1000 "
.equ l_valid, . - s_valid
s_dump: .ascii "VALID_TRUE_PROBS_F32\n"
.equ l_dump, . - s_dump
s_epoch: .ascii "epoch "
.equ l_epoch, . - s_epoch
s_loss: .ascii " loss_x1000 "
.equ l_loss, . - s_loss
s_gen: .ascii "--- attention sample ---\n"
.equ l_gen, . - s_gen
s_kernel: .ascii "kernel_avx2 "
.equ l_kernel, . - s_kernel
.ifdef INFERENCE
.section .rodata
.balign 16
params_start:
    .incbin "models/attention_weights.f32"
params_end:
.if (params_end-params_start) != 38464
.error "checkpoint length mismatch"
.endif
.equ E, params_start
.equ POS, params_start+16384
.equ F, params_start+16896
.equ FB, params_start+17920
.equ O, params_start+17984
.equ OB, params_start+34368
.equ WQ, params_start+35392
.equ WK, params_start+36416
.equ WV, params_start+37440
.else
.section .bss
.balign 16
params_start:
E:      .skip 16384
POS:    .skip 512
F:      .skip 1024
FB:     .skip 256
O:      .skip 16384
OB:     .skip 1024
WQ:     .skip 1024
WK:     .skip 1024
WV:     .skip 1024
params_end:
.endif
.section .bss
.balign 16
X:      .skip 2048
Q:      .skip 256
K:      .skip 2048
VCTX:   .skip 2048
GQ:     .skip 64
GSC:     .skip 32
A:      .skip 128
S:      .skip 128
H:      .skip 256
Z:      .skip 256
Y:      .skip 256
L:      .skip 4096
PROB:   .skip 4096
GY:     .skip 256
GH:     .skip 256
GA:     .skip 128
GX:     .skip 2048
GL:     .skip 4096
TOK:    .skip 8
MASK:   .skip 256
seqbase:.skip 8
seqlen: .skip 4
target: .skip 4
simd_capable: .skip 4
number: .skip 32
ftmp:   .skip 16
genbuf: .skip SAMPLE_LEN+CTX+1
.section .text
.global _start
# xmm0=exp(xmm0), x87 with stable shifted softmax inputs.
exp_f:
    movss [rip+ftmp],xmm0
    fld dword ptr [rip+ftmp]
    fldl2e
    fmulp st(1),st
    fld st(0)
    frndint
    fxch st(1)
    fsub st(0),st(1)
    f2xm1
    fld1
    faddp st(1),st
    fscale
    fstp st(1)
    fstp dword ptr [rip+ftmp]
    movss xmm0,[rip+ftmp]
    ret
# xmm0=ln(xmm0)
ln_f:
    movss [rip+ftmp],xmm0
    fldln2
    fld dword ptr [rip+ftmp]
    fyl2x
    fstp dword ptr [rip+ftmp]
    movss xmm0,[rip+ftmp]
    ret
fatal:
    mov eax,60
    mov edi,3
    syscall
# rsi=bytes, rdx=len, EINTR and partial-write safe.
write_all:
    test rdx,rdx
    jz .Lwdone
.Lwloop:
    mov eax,1
    mov edi,1
    syscall
    cmp rax,-4
    je .Lwloop
    test rax,rax
    jle fatal
    add rsi,rax
    sub rdx,rax
    jnz .Lwloop
.Lwdone:
    ret
# eax=unsigned, r9b=trailing character
print_u:
    lea r8,[rip+number+31]
    mov byte ptr [r8],r9b
    mov ecx,10
.Ldigits:
    xor edx,edx
    div ecx
    add dl,'0'
    dec r8
    mov byte ptr [r8],dl
    test eax,eax
    jnz .Ldigits
    lea rdx,[rip+number+32]
    sub rdx,r8
    mov rsi,r8
    call write_all
    ret
.macro SAY name, len
    lea rsi,[rip+\name]
    mov edx,\len
    call write_all
.endm
# Deterministic nonzero xorshift32, eax = next uint32, r14=state
rng:
    mov eax,r14d
    mov edx,eax
    shl edx,13
    xor eax,edx
    mov edx,eax
    shr edx,17
    xor eax,edx
    mov edx,eax
    shl edx,5
    xor eax,edx
    mov r14d,eax
    ret
# CPU feature detection: CPUID(1).OSXSAVE+AVX+FMA, XGETBV(0) XMM/YMM,
# and CPUID(7,0).AVX2. xgetbv MUST NOT execute if OSXSAVE unavailable.
# Only called on the AVX2_AUTO build, before model initialization.
.ifdef AVX2_AUTO
detect_simd:
    push rbx
    mov eax, 0
    cpuid
    cmp eax, 7
    jb .Lno_simd
    mov eax, 1
    cpuid
    bt ecx, 27
    jnc .Lno_simd
    bt ecx, 28
    jnc .Lno_simd
    bt ecx, 12
    jnc .Lno_simd
    xor ecx,ecx
    xgetbv
    and eax,6
    cmp eax,6
    jne .Lno_simd
    mov eax,7
    xor ecx,ecx
    cpuid
    bt ebx,5
    jnc .Lno_simd
    .ifndef FORCE_SCALAR
    mov dword ptr [rip+simd_capable],1
    .endif
.Lno_simd:
    pop rbx
    ret
.endif

# Fixed width GEMV inner product, 16 x float32.
# r10=A, r11=B; output xmm0. Preserves r8/r9, xmm7..xmm15.
# AVX2 with FMA3/256-bit registers inspired by NN_S (MIT) and Intel manual,
# but the code here is ORIGINAL. Scalar SSE path remains portable.
dot16:
    .ifdef AVX2_AUTO
    cmp dword ptr [rip+simd_capable],0
    je .Ldot16_scalar
    .endif
    .if COMPILE_SIMD
    vmovups ymm0, YMMWORD PTR [r10]
    vmovups ymm1, YMMWORD PTR [r11]
    vmulps ymm0, ymm0, ymm1
    vmovups ymm2, YMMWORD PTR [r10+32]
    vmovups ymm3, YMMWORD PTR [r11+32]
    vfmadd231ps ymm0, ymm2, ymm3
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vzeroupper
    ret
    .endif
.Ldot16_scalar:
    xorps xmm0, xmm0
    xor ecx, ecx
.Ldot16_scalar_loop:
    movss xmm1, [r10+rcx*4]
    mulss xmm1, [r11+rcx*4]
    addss xmm0, xmm1
    inc ecx
    cmp ecx, D
    jb .Ldot16_scalar_loop
    ret


# Four-row tiled projection. ABI: RDI=W16x16, RSI=x16, RDX=y16.
# Preserves RBX/R8/R9/R12-R15; clobbers RAX/RCX/R10/R11 and XMM0-XMM6.
# Floating reduction order per row exactly matches the existing dot16 SIMD.
.ifdef TILED_PROJECTIONS
matvec16_tiled:
 .ifdef AVX2_AUTO
 cmp dword ptr [rip+simd_capable],0
 je .Lmv_scalar
 .endif
 .if COMPILE_SIMD
 vmovups ymm4,[rsi]
 vmovups ymm5,[rsi+32]
 xor eax,eax
 xor ecx,ecx
.Lmv_block:
 vmovups ymm0,[rdi+rax+0]
 vmulps ymm0,ymm0,ymm4
 vmovups ymm6,[rdi+rax+32]
 vfmadd231ps ymm0,ymm6,ymm5
 vmovups ymm1,[rdi+rax+64]
 vmulps ymm1,ymm1,ymm4
 vmovups ymm6,[rdi+rax+96]
 vfmadd231ps ymm1,ymm6,ymm5
 vmovups ymm2,[rdi+rax+128]
 vmulps ymm2,ymm2,ymm4
 vmovups ymm6,[rdi+rax+160]
 vfmadd231ps ymm2,ymm6,ymm5
 vmovups ymm3,[rdi+rax+192]
 vmulps ymm3,ymm3,ymm4
 vmovups ymm6,[rdi+rax+224]
 vfmadd231ps ymm3,ymm6,ymm5
 vextractf128 xmm6,ymm0,1
 vaddps xmm0,xmm0,xmm6
 vhaddps xmm0,xmm0,xmm0
 vhaddps xmm0,xmm0,xmm0
 vmovss [rdx+rcx+0],xmm0
 vextractf128 xmm6,ymm1,1
 vaddps xmm1,xmm1,xmm6
 vhaddps xmm1,xmm1,xmm1
 vhaddps xmm1,xmm1,xmm1
 vmovss [rdx+rcx+4],xmm1
 vextractf128 xmm6,ymm2,1
 vaddps xmm2,xmm2,xmm6
 vhaddps xmm2,xmm2,xmm2
 vhaddps xmm2,xmm2,xmm2
 vmovss [rdx+rcx+8],xmm2
 vextractf128 xmm6,ymm3,1
 vaddps xmm3,xmm3,xmm6
 vhaddps xmm3,xmm3,xmm3
 vhaddps xmm3,xmm3,xmm3
 vmovss [rdx+rcx+12],xmm3
 add eax,256
 add ecx,16
 cmp eax,1024
 jb .Lmv_block
 vzeroupper
 ret
 .endif
.Lmv_scalar:
 xor eax,eax
.Lmv_scalar_row:
 lea r10,[rdi+rax]
 mov r11,rsi
 call dot16
 mov ecx,eax
 shr ecx,4
 movss [rdx+rcx],xmm0
 add eax,64
 cmp eax,1024
 jb .Lmv_scalar_row
 ret
.endif

# Forward: r13 = position >= CTX, seqbase = ptr, predicts token at r13.
# Q=WQ*X[last], K[t]=WK*X[t], V[t]=WV*X[t].
# attention = softmax(Q*K/sqrt(16)); H=X[last]+attention*V;
# Y=H+ReLU(F*H+FB); logits=O*Y+OB; PROB=masked-softmax(logits).
forward:
    mov rbx,[rip+seqbase]
    xor r8d,r8d
.Lctx:
    mov eax,r13d
    sub eax,CTX
    add eax,r8d
    movzx eax,byte ptr [rbx+rax]
    lea rdi,[rip+TOK]
    mov byte ptr [rdi+r8],al
    shl eax,6
    lea rsi,[rip+E]
    add rsi,rax
    mov r9d,r8d
    shl r9d,6
    lea rdi,[rip+X]
    add rdi,r9
    lea rdx,[rip+POS]
    add rdx,r9
    xor ecx,ecx
.Lctxdim:
    movss xmm0,[rsi+rcx*4]
    addss xmm0,[rdx+rcx*4]
    movss [rdi+rcx*4],xmm0
    inc ecx
    cmp ecx,D
    jb .Lctxdim
    inc r8d
    cmp r8d,CTX
    jb .Lctx
    .ifdef TILED_PROJECTIONS
    # Q/K/V projection shares input loads across four output rows.
    lea rdi,[rip+WQ]
    lea rsi,[rip+X+448]
    lea rdx,[rip+Q]
    call matvec16_tiled
    xor r8d,r8d
.Ltiled_kv:
    mov eax,r8d
    shl eax,6
    lea rsi,[rip+X]
    add rsi,rax
    lea rdi,[rip+WK]
    lea rdx,[rip+K]
    add rdx,rax
    call matvec16_tiled
    mov eax,r8d
    shl eax,6
    lea rdi,[rip+WV]
    lea rdx,[rip+VCTX]
    add rdx,rax
    call matvec16_tiled
    inc r8d
    cmp r8d,CTX
    jb .Ltiled_kv
    .else
    # Actual *learnable*, independent query/key/value projection matrices.
    # Shape: WQ, WK, WV = [16,16] row-major float32.
    xor r8d,r8d
.Lqrow:
    mov r9d,r8d
    shl r9d,6
    lea r10,[rip+WQ]
    add r10,r9
    lea r11,[rip+X+448]
    call dot16
    lea r10,[rip+Q]
    movss [r10+r8*4],xmm0
    inc r8d
    cmp r8d,D
    jb .Lqrow
    xor r8d,r8d
.Lkv_t:
    mov r9d,r8d
    shl r9d,6
    lea rsi,[rip+X]
    add rsi,r9
    xor ebx,ebx
.Lkv_row:
    mov eax,ebx
    shl eax,6
    lea r10,[rip+WK]
    add r10,rax
    mov r11,rsi
    call dot16
    mov r9d,r8d
    shl r9d,4
    add r9d,ebx
    lea r10,[rip+K]
    movss [r10+r9*4],xmm0
    mov eax,ebx
    shl eax,6
    lea r10,[rip+WV]
    add r10,rax
    mov r11,rsi
    call dot16
    lea r10,[rip+VCTX]
    movss [r10+r9*4],xmm0
    inc ebx
    cmp ebx,D
    jb .Lkv_row
    inc r8d
    cmp r8d,CTX
    jb .Lkv_t
    .endif
    xor r8d,r8d
.Lscores:
    mov r9d,r8d
    shl r9d,6
    lea r10,[rip+K]
    add r10,r9
    lea r11,[rip+Q]
    call dot16
    mulss xmm0,[rip+quarter]
    lea r10,[rip+S]
    movss [r10+r8*4],xmm0
    inc r8d
    cmp r8d,CTX
    jb .Lscores
    lea r10,[rip+S]
    movss xmm7,[r10]
    mov ecx,1
.Lscoremax:
    maxss xmm7,[r10+rcx*4]
    inc ecx
    cmp ecx,CTX
    jb .Lscoremax
    xorps xmm8,xmm8
    xor r8d,r8d
.Lattnexp:
    lea r10,[rip+S]
    movss xmm0,[r10+r8*4]
    subss xmm0,xmm7
    call exp_f
    lea r10,[rip+A]
    movss [r10+r8*4],xmm0
    addss xmm8,xmm0
    inc r8d
    cmp r8d,CTX
    jb .Lattnexp
    ucomiss xmm8,[rip+small]
    jbe fatal
    xor ecx,ecx
.Lattnnorm:
    lea r10,[rip+A]
    movss xmm0,[r10+rcx*4]
    divss xmm0,xmm8
    movss [r10+rcx*4],xmm0
    inc ecx
    cmp ecx,CTX
    jb .Lattnnorm
    xor r9d,r9d
.Lh:
    lea r10,[rip+X+448]
    movss xmm0,[r10+r9*4]
    xor r8d,r8d
.Lhterm:
    mov ecx,r8d
    shl ecx,4
    add ecx,r9d
    lea r10,[rip+VCTX]
    movss xmm1,[r10+rcx*4]
    lea r10,[rip+A]
    mulss xmm1,[r10+r8*4]
    addss xmm0,xmm1
    inc r8d
    cmp r8d,CTX
    jb .Lhterm
    lea r10,[rip+H]
    movss [r10+r9*4],xmm0
    inc r9d
    cmp r9d,D
    jb .Lh
    xor r8d,r8d
.Lff:
    mov r9d,r8d
    shl r9d,6
    lea r10,[rip+F]
    add r10,r9
    lea r11,[rip+H]
    call dot16
    lea r10,[rip+FB]
    addss xmm0,[r10+r8*4]
    lea r10,[rip+Z]
    movss [r10+r8*4],xmm0
    xorps xmm1,xmm1
    maxss xmm0,xmm1
    lea r10,[rip+H]
    addss xmm0,[r10+r8*4]
    lea r10,[rip+Y]
    movss [r10+r8*4],xmm0
    inc r8d
    cmp r8d,D
    jb .Lff
    # output logits and finite-check
    xor r8d,r8d
.Loutv:
    lea r10,[rip+MASK]
    cmp byte ptr [r10+r8],0
    je .Loutmasked
    mov r9d,r8d
    shl r9d,6
    lea r10,[rip+O]
    add r10,r9
    lea r11,[rip+Y]
    call dot16
    lea r10,[rip+OB]
    addss xmm0,[r10+r8*4]
    movd eax,xmm0
    and eax,0x7f800000
    cmp eax,0x7f800000
    je fatal
    jmp .Loutstore
.Loutmasked:
    movss xmm0,[rip+neg]
.Loutstore:
    lea r10,[rip+L]
    movss [r10+r8*4],xmm0
    inc r8d
    cmp r8d,V
    jb .Loutv
    lea r10,[rip+L]
    movss xmm7,[rip+neg]
    xor ecx,ecx
.Loutmax:
    maxss xmm7,[r10+rcx*4]
    inc ecx
    cmp ecx,V
    jb .Loutmax
    xorps xmm8,xmm8
    xor r8d,r8d
.Loutsoft:
    xorps xmm0,xmm0
    lea r10,[rip+MASK]
    cmp byte ptr [r10+r8],0
    je .Loutzero
    lea r10,[rip+L]
    movss xmm0,[r10+r8*4]
    subss xmm0,xmm7
    call exp_f
.Loutzero:
    lea r10,[rip+PROB]
    movss [r10+r8*4],xmm0
    addss xmm8,xmm0
    inc r8d
    cmp r8d,V
    jb .Loutsoft
    ucomiss xmm8,[rip+small]
    jbe fatal
    xor ecx,ecx
.Loutnormalize:
    lea r10,[rip+PROB]
    movss xmm0,[r10+rcx*4]
    divss xmm0,xmm8
    movss [r10+rcx*4],xmm0
    inc ecx
    cmp ecx,V
    jb .Loutnormalize
    ret
# Reverse-mode derivatives of E,POS,F,FB,O,OB,WQ,WK,WV.
# All Q/K/V projections are separately trainable, initialized as identity.
backward:
    mov rbx,[rip+seqbase]
    movzx eax,byte ptr [rbx+r13]
    mov [rip+target],eax
    # zero 16+16+8+128 floats (GY, GH, GA, GX)
    xor eax,eax
    lea rdi,[rip+GY]
    mov ecx,16
    rep stosd
    lea rdi,[rip+GH]
    mov ecx,16
    rep stosd
    lea rdi,[rip+GA]
    mov ecx,8
    rep stosd
    lea rdi,[rip+GX]
    mov ecx,128
    rep stosd
    xor r8d,r8d
.Lgradlogit:
    lea r10,[rip+MASK]
    cmp byte ptr [r10+r8],0
    je .Lgradlogit_next
    lea r10,[rip+PROB]
    movss xmm0,[r10+r8*4]
    cmp r8d,[rip+target]
    jne .Lnot_true
    subss xmm0,[rip+one]
.Lnot_true:
    lea r10,[rip+GL]
    movss [r10+r8*4],xmm0
    mov r9d,r8d
    shl r9d,4
    lea r10,[rip+O]
    lea r11,[r10+r9*4]
    lea r10,[rip+GY]
    xor ecx,ecx
.Lgradoutdot:
    movss xmm1,[r11+rcx*4]
    mulss xmm1,xmm0
    addss xmm1,[r10+rcx*4]
    movss [r10+rcx*4],xmm1
    inc ecx
    cmp ecx,D
    jb .Lgradoutdot
.Lgradlogit_next:
    inc r8d
    cmp r8d,V
    jb .Lgradlogit
    # Now update output projection AFTER input-side grads computed.
    xor r8d,r8d
.Loutgrad:
    lea r10,[rip+MASK]
    cmp byte ptr [r10+r8],0
    je .Loutgradnext
    lea r10,[rip+GL]
    movss xmm0,[r10+r8*4]
    mulss xmm0,[rip+lr]
    lea r10,[rip+OB]
    movss xmm1,[r10+r8*4]
    subss xmm1,xmm0
    movss [r10+r8*4],xmm1
    mov r9d,r8d
    shl r9d,4
    lea r10,[rip+O]
    lea r11,[r10+r9*4]
    lea r10,[rip+Y]
    xor ecx,ecx
.Loutgraddim:
    movss xmm1,[r10+rcx*4]
    mulss xmm1,xmm0
    movss xmm2,[r11+rcx*4]
    subss xmm2,xmm1
    movss [r11+rcx*4],xmm2
    inc ecx
    cmp ecx,D
    jb .Loutgraddim
.Loutgradnext:
    inc r8d
    cmp r8d,V
    jb .Loutgrad
    # GH=GY+(ReLU'(Z)*GY)*F using old F, then update F/FB.
    xor ecx,ecx
.Lghinit:
    lea r10,[rip+GY]
    movss xmm0,[r10+rcx*4]
    lea r11,[rip+GH]
    movss [r11+rcx*4],xmm0
    inc ecx
    cmp ecx,D
    jb .Lghinit
    xor r8d,r8d
.Lghrow:
    lea r10,[rip+Z]
    movss xmm0,[r10+r8*4]
    xorps xmm1,xmm1
    ucomiss xmm0,xmm1
    jbe .Lghnext
    lea r10,[rip+GY]
    movss xmm0,[r10+r8*4]
    mov r9d,r8d
    shl r9d,4
    lea r10,[rip+F]
    lea r11,[r10+r9*4]
    lea r10,[rip+GH]
    xor ecx,ecx
.Lghdot:
    movss xmm1,[r11+rcx*4]
    mulss xmm1,xmm0
    addss xmm1,[r10+rcx*4]
    movss [r10+rcx*4],xmm1
    inc ecx
    cmp ecx,D
    jb .Lghdot
.Lghnext:
    inc r8d
    cmp r8d,D
    jb .Lghrow
    xor r8d,r8d
.Lffgrad:
    lea r10,[rip+Z]
    movss xmm0,[r10+r8*4]
    xorps xmm1,xmm1
    ucomiss xmm0,xmm1
    jbe .Lffgradnext
    lea r10,[rip+GY]
    movss xmm0,[r10+r8*4]
    mulss xmm0,[rip+lr]
    lea r10,[rip+FB]
    movss xmm1,[r10+r8*4]
    subss xmm1,xmm0
    movss [r10+r8*4],xmm1
    mov r9d,r8d
    shl r9d,4
    lea r10,[rip+F]
    lea r11,[r10+r9*4]
    lea r10,[rip+H]
    xor ecx,ecx
.Lffgradinner:
    movss xmm1,[r10+rcx*4]
    mulss xmm1,xmm0
    movss xmm2,[r11+rcx*4]
    subss xmm2,xmm1
    movss [r11+rcx*4],xmm2
    inc ecx
    cmp ecx,D
    jb .Lffgradinner
.Lffgradnext:
    inc r8d
    cmp r8d,D
    jb .Lffgrad
    # gA[t] = GH dot V[t]. All projection paths use OLD weights.
    xor r8d,r8d
.Lval_grad_t:
    mov r9d,r8d
    shl r9d,6
    lea r10,[rip+VCTX]
    add r10,r9
    lea r11,[rip+GH]
    call dot16
    lea r10,[rip+GA]
    movss [r10+r8*4],xmm0
    inc r8d
    cmp r8d,CTX
    jb .Lval_grad_t
    # softmax derivative: gs_t=(gA[t]-sum_k A[k]gA[k])*A[t]/sqrt(D)
    xorps xmm7,xmm7
    xor ecx,ecx
.Lnew_gsum:
    lea r10,[rip+A]
    movss xmm0,[r10+rcx*4]
    lea r10,[rip+GA]
    mulss xmm0,[r10+rcx*4]
    addss xmm7,xmm0
    inc ecx
    cmp ecx,CTX
    jb .Lnew_gsum
    xor eax,eax
    lea rdi,[rip+GQ]
    mov ecx,D
    rep stosd
    xor r8d,r8d
.Lnew_gradscore_t:
    lea r10,[rip+GA]
    movss xmm2,[r10+r8*4]
    subss xmm2,xmm7
    lea r10,[rip+A]
    mulss xmm2,[r10+r8*4]
    mulss xmm2,[rip+quarter]
    lea r10,[rip+GSC]
    movss [r10+r8*4],xmm2
    # dQ[i] += gs[t] * K[t,i]
    mov r9d,r8d
    shl r9d,6
    lea r11,[rip+K]
    add r11,r9
    lea r10,[rip+GQ]
    xor ecx,ecx
.Lnew_gradq_i:
    movss xmm0,[r11+rcx*4]
    mulss xmm0,xmm2
    addss xmm0,[r10+rcx*4]
    movss [r10+rcx*4],xmm0
    inc ecx
    cmp ecx,D
    jb .Lnew_gradq_i
    inc r8d
    cmp r8d,CTX
    jb .Lnew_gradscore_t
    # dX[t,j] += sum_i WK[i,j]*(GS[t]*Q[i])
    #             + sum_i WV[i,j]*(A[t]*GH[i])
    # GX was zeroed at backward entry. Do not update projections yet.
    xor r8d,r8d
.Lnew_gxt:
    mov r9d,r8d
    shl r9d,6
    lea rdi,[rip+GX]
    add rdi,r9
    lea r11,[rip+A]
    movss xmm4,[r11+r8*4]
    lea r11,[rip+GSC]
    movss xmm5,[r11+r8*4]
    xor ebx,ebx
.Lnew_gxi:
    lea r10,[rip+Q]
    movss xmm2,[r10+rbx*4]
    mulss xmm2,xmm5
    lea r10,[rip+GH]
    movss xmm3,[r10+rbx*4]
    mulss xmm3,xmm4
    mov eax,ebx
    shl eax,6
    lea r10,[rip+WK]
    add r10,rax
    lea r11,[rip+WV]
    add r11,rax
    xor ecx,ecx
.Lnew_gxj:
    movss xmm0,[r10+rcx*4]
    mulss xmm0,xmm2
    movss xmm1,[r11+rcx*4]
    mulss xmm1,xmm3
    addss xmm0,xmm1
    addss xmm0,[rdi+rcx*4]
    movss [rdi+rcx*4],xmm0
    inc ecx
    cmp ecx,D
    jb .Lnew_gxj
    inc ebx
    cmp ebx,D
    jb .Lnew_gxi
    inc r8d
    cmp r8d,CTX
    jb .Lnew_gxt
    # dX[last,j] += WQ^T[j,i] * dQ[i]
    xor ebx,ebx
.Lnew_gradq_row:
    lea r10,[rip+GQ]
    movss xmm2,[r10+rbx*4]
    mov eax,ebx
    shl eax,6
    lea r10,[rip+WQ]
    add r10,rax
    lea rdi,[rip+GX+448]
    xor ecx,ecx
.Lnew_gradq_j:
    movss xmm0,[r10+rcx*4]
    mulss xmm0,xmm2
    addss xmm0,[rdi+rcx*4]
    movss [rdi+rcx*4],xmm0
    inc ecx
    cmp ecx,D
    jb .Lnew_gradq_j
    inc ebx
    cmp ebx,D
    jb .Lnew_gradq_row
    # Residual gradient from Y -> H -> last input.
    xor ecx,ecx
.Lresidual:
    lea r10,[rip+GX+448]
    movss xmm0,[r10+rcx*4]
    lea r11,[rip+GH]
    addss xmm0,[r11+rcx*4]
    movss [r10+rcx*4],xmm0
    inc ecx
    cmp ecx,D
    jb .Lresidual
    # Update WQ/WK/WV only AFTER dX used the OLD projection weights.
    # dWQ[i,j]=dQ[i]*X[last,j]
    # dWK[i,j]=sum_t GS[t]*Q[i]*X[t,j]
    # dWV[i,j]=sum_t A[t]*GH[i]*X[t,j]
    xor r8d,r8d
.Lnew_proj_update_i:
    lea r10,[rip+GQ]
    movss xmm4,[r10+r8*4]
    lea r10,[rip+Q]
    movss xmm5,[r10+r8*4]
    lea r10,[rip+GH]
    movss xmm6,[r10+r8*4]
    mov r9d,r8d
    shl r9d,6
    lea r10,[rip+WQ]
    add r10,r9
    lea r11,[rip+WK]
    add r11,r9
    lea rdi,[rip+WV]
    add rdi,r9
    xor ecx,ecx
.Lnew_proj_update_j:
    lea rbx,[rip+X+448]
    movss xmm0,[rbx+rcx*4]
    mulss xmm0,xmm4
    mulss xmm0,[rip+lr]
    movss xmm1,[r10+rcx*4]
    subss xmm1,xmm0
    movss [r10+rcx*4],xmm1
    xorps xmm2,xmm2
    xorps xmm3,xmm3
    xor edx,edx
.Lnew_proj_update_t:
    mov eax,edx
    shl eax,4
    add eax,ecx
    lea rbx,[rip+X]
    movss xmm0,[rbx+rax*4]
    lea rbx,[rip+GSC]
    movss xmm1,[rbx+rdx*4]
    mulss xmm1,xmm0
    addss xmm2,xmm1
    lea rbx,[rip+A]
    movss xmm1,[rbx+rdx*4]
    mulss xmm1,xmm0
    addss xmm3,xmm1
    inc edx
    cmp edx,CTX
    jb .Lnew_proj_update_t
    mulss xmm2,xmm5
    mulss xmm3,xmm6
    mulss xmm2,[rip+lr]
    mulss xmm3,[rip+lr]
    movss xmm0,[r11+rcx*4]
    subss xmm0,xmm2
    movss [r11+rcx*4],xmm0
    movss xmm0,[rdi+rcx*4]
    subss xmm0,xmm3
    movss [rdi+rcx*4],xmm0
    inc ecx
    cmp ecx,D
    jb .Lnew_proj_update_j
    inc r8d
    cmp r8d,D
    jb .Lnew_proj_update_i
    # Update trainable embeddings and position vectors; repeated tokens accumulate.
    xor r8d,r8d
.Lembedt:
    lea r10,[rip+TOK]
    movzx eax,byte ptr [r10+r8]
    shl eax,4
    lea r10,[rip+E]
    lea r11,[r10+rax*4]
    mov r9d,r8d
    shl r9d,4
    lea r10,[rip+POS]
    lea rsi,[r10+r9*4]
    lea r10,[rip+GX]
    lea rdi,[r10+r9*4]
    xor ecx,ecx
.Lembedj:
    movss xmm0,[rdi+rcx*4]
    mulss xmm0,[rip+lr]
    movss xmm1,[r11+rcx*4]
    subss xmm1,xmm0
    movss [r11+rcx*4],xmm1
    movss xmm1,[rsi+rcx*4]
    subss xmm1,xmm0
    movss [rsi+rcx*4],xmm1
    inc ecx
    cmp ecx,D
    jb .Lembedj
    inc r8d
    cmp r8d,CTX
    jb .Lembedt
    ret
# Evaluate corpus specified by seqbase/seqlen WITHOUT any updates.
# Result xmm10 = average negative log likelihood scaled x1000.
eval_current:
    mov r13d,CTX
    xorps xmm10,xmm10
.Levalnext:
    mov eax,[rip+seqlen]
    cmp r13d,eax
    jae .Levaldone
    call forward
    mov rbx,[rip+seqbase]
    movzx eax,byte ptr [rbx+r13]
    lea r10,[rip+PROB]
    movss xmm0,[r10+rax*4]
    maxss xmm0,[rip+small]
    call ln_f
    subss xmm10,xmm0
    inc r13d
    jmp .Levalnext
.Levaldone:
    mov eax,[rip+seqlen]
    sub eax,CTX
    cvtsi2ss xmm1,eax
    divss xmm10,xmm1
    mulss xmm10,[rip+mil]
    ret
# Print xmm10 as positive unsigned milli-nats.
print_metric:
    cvtss2si eax,xmm10
    test eax,eax
    js fatal
    mov r9d,10
    call print_u
    ret
_start:
    cld
    fninit
    .ifdef AVX2_AUTO
    call detect_simd
    SAY s_kernel,l_kernel
    mov eax,[rip+simd_capable]
    mov r9d,10
    call print_u
    .endif
    .ifndef INFERENCE
    # deterministic stochastic initialization of ALL trainable float parameters
    # no imported checkpoints, weights, datasets or model calls.
    mov r14d,0x1234567
    xor r12d,r12d
    lea rbx,[rip+params_start]
.Linit:
    call rng
    movsx eax,ax
    cvtsi2ss xmm0,eax
    mulss xmm0,[rip+randdiv]
    mulss xmm0,[rip+scale]
    movss [rbx+r12*4],xmm0
    inc r12d
    cmp r12d,(params_end-params_start)/4
    jb .Linit
    # Identity initialization keeps initial Q/K/V mathematically equivalent
    # to old identity-tied attention; all three are independently trained.
    lea rdi,[rip+WQ]
    xor eax,eax
    mov ecx,768
    rep stosd
    xor ecx,ecx
.Lqkv_identity:
    mov eax,ecx
    imul eax,17
    lea rdi,[rip+WQ]
    mov dword ptr [rdi+rax*4],0x3f800000
    lea rdi,[rip+WK]
    mov dword ptr [rdi+rax*4],0x3f800000
    lea rdi,[rip+WV]
    mov dword ptr [rdi+rax*4],0x3f800000
    inc ecx
    cmp ecx,D
    jb .Lqkv_identity
    .endif
    # vocabulary includes exactly the bytes in training corpus
    lea rbx,[rip+train_bytes]
    lea rdx,[rip+MASK]
    xor ecx,ecx
.Lmaskinit:
    movzx eax,byte ptr [rbx+rcx]
    mov byte ptr [rdx+rax],1
    inc ecx
    cmp ecx,N
    jb .Lmaskinit
    .ifndef INFERENCE
    lea rax,[rip+train_bytes]
    mov [rip+seqbase],rax
    mov dword ptr [rip+seqlen],N
    call eval_current
    SAY s_init,l_init
    call print_metric
    xor r12d,r12d
    .if EPOCHS > 0
.Lepoch:
    lea rax,[rip+train_bytes]
    mov [rip+seqbase],rax
    mov dword ptr [rip+seqlen],N
    mov r13d,CTX
    xorps xmm10,xmm10
.Lnexttrain:
    call forward
    mov rbx,[rip+seqbase]
    movzx eax,byte ptr [rbx+r13]
    lea r10,[rip+PROB]
    movss xmm0,[r10+rax*4]
    maxss xmm0,[rip+small]
    call ln_f
    subss xmm10,xmm0
    call backward
    inc r13d
    .ifdef ONE_STEP
    # Test-only: exactly one update, then ordinary metrics/export.
    jmp .Lnext_epoch
    .endif
    cmp r13d,N
    jb .Lnexttrain
    mov eax,r12d
    xor edx,edx
    mov ecx,40
    div ecx
    test edx,edx
    jnz .Lnext_epoch
    SAY s_epoch,l_epoch
    mov eax,r12d
    mov r9d,' '
    call print_u
    SAY s_loss,l_loss
    mov eax,N-CTX
    cvtsi2ss xmm1,eax
    divss xmm10,xmm1
    mulss xmm10,[rip+mil]
    call print_metric
.Lnext_epoch:
    inc r12d
    cmp r12d,EPOCHS
    jb .Lepoch
    .endif
    lea rax,[rip+train_bytes]
    mov [rip+seqbase],rax
    mov dword ptr [rip+seqlen],N
    call eval_current
    SAY s_final,l_final
    call print_metric
    lea rax,[rip+valid_bytes]
    mov [rip+seqbase],rax
    mov dword ptr [rip+seqlen],NV
    call eval_current
    SAY s_valid,l_valid
    call print_metric
    # Opt-in raw float32 probability for each true validation byte. No SGD
    # updates happen in this path; eval only. A separate harness checks that
    # -log(prob) reproduces the native heldout loss before using the stream.
    .ifdef DUMP_VALID_PROBS
    .ifdef EXPORT_WEIGHTS
    .error "DUMP_VALID_PROBS and EXPORT_WEIGHTS are mutually exclusive"
    .endif
    SAY s_dump,l_dump
    lea rax,[rip+valid_bytes]
    mov [rip+seqbase],rax
    mov dword ptr [rip+seqlen],NV
    mov r13d,CTX
.Ldump_next:
    cmp r13d,NV
    jae .Ldump_end
    call forward
    lea rbx,[rip+valid_bytes]
    movzx eax,byte ptr [rbx+r13]
    lea r10,[rip+PROB]
    lea rsi,[r10+rax*4]
    mov edx,4
    call write_all
    inc r13d
    jmp .Ldump_next
.Ldump_end:
    mov eax,60
    xor edi,edi
    syscall
    .endif
    .ifdef EXPORT_WEIGHTS
    lea rsi,[rip+params_start]
    mov edx,(params_end-params_start)
    call write_all
    mov eax,60
    xor edi,edi
    syscall
    .endif
    .endif
    SAY s_gen,l_gen
    lea rsi,[rip+train_bytes]
    lea rdi,[rip+genbuf]
    mov ecx,CTX
    rep movsb
    lea rsi,[rip+genbuf]
    mov edx,CTX
    call write_all
    lea rax,[rip+genbuf]
    mov [rip+seqbase],rax
    mov r13d,CTX
    mov r14d,SEED
    mov r15d,SAMPLE_LEN
.Lsample:
    test r15d,r15d
    jz .Lend
    call forward
    call rng
    and eax,0xffffff
    cvtsi2ss xmm2,eax
    mulss xmm2,[rip+inv24]
    xorps xmm3,xmm3
    xor ebx,ebx
.Lpick:
    lea r10,[rip+MASK]
    cmp byte ptr [r10+rbx],0
    je .Lpicknext
    lea r10,[rip+PROB]
    addss xmm3,[r10+rbx*4]
    ucomiss xmm3,xmm2
    jp fatal
    ja .Lchosen
.Lpicknext:
    inc ebx
    cmp ebx,256
    jb .Lpick
    # normalized probabilities should sum to one, roundoff fallback
    lea r10,[rip+MASK]
    mov ebx,255
.Lfallback:
    cmp byte ptr [r10+rbx],0
    jne .Lchosen
    dec ebx
    jns .Lfallback
    jmp fatal
.Lchosen:
    lea r10,[rip+genbuf]
    mov byte ptr [r10+r13],bl
    lea rsi,[r10+r13]
    mov edx,1
    call write_all
    inc r13d
    dec r15d
    jmp .Lsample
.Lend:
    lea r10,[rip+number]
    mov byte ptr [r10],10
    mov rsi,r10
    mov edx,1
    call write_all
    mov eax,60
    xor edi,edi
    syscall
.section .note.GNU-stack,"",@progbits
