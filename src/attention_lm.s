# Niyah: learnable one-head causal attention + residual ReLU FFN LM
# x86-64 GNU as, Linux kernel syscalls, no libc, no PyTorch, no runtime deps.
# Educational neural language model, NOT a large language model.
.intel_syntax noprefix
.equ D, 16
.equ CTX, 8
.equ V, 256
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
s_epoch: .ascii "epoch "
.equ l_epoch, . - s_epoch
s_loss: .ascii " loss_x1000 "
.equ l_loss, . - s_loss
s_gen: .ascii "--- attention sample ---\n"
.equ l_gen, . - s_gen
.ifdef INFERENCE
.section .rodata
.balign 16
params_start:
    .incbin "models/attention_weights.f32"
params_end:
.if (params_end-params_start) != 35392
.error "checkpoint length mismatch"
.endif
.equ E, params_start
.equ POS, params_start+16384
.equ F, params_start+16896
.equ FB, params_start+17920
.equ O, params_start+17984
.equ OB, params_start+34368
.else
.section .bss
.balign 16
params_start:
E:      .skip 16384
POS:    .skip 512
F:      .skip 1024
FB:     .skip 64
O:      .skip 16384
OB:     .skip 1024
params_end:
.endif
.section .bss
.balign 16
X:      .skip 512
A:      .skip 32
S:      .skip 32
H:      .skip 64
Z:      .skip 64
Y:      .skip 64
L:      .skip 1024
PROB:   .skip 1024
GY:     .skip 64
GH:     .skip 64
GA:     .skip 32
GX:     .skip 512
GL:     .skip 1024
TOK:    .skip 8
MASK:   .skip 256
seqbase:.skip 8
seqlen: .skip 4
target: .skip 4
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
# Forward: r13 = position >= CTX, seqbase = ptr, predicts token at r13.
# Q=last positional token; K=V=context positional embeddings.
# attention = softmax(Q*K/sqrt(16)); H=Q+attention*V;
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
    xor r8d,r8d
.Lscores:
    xorps xmm0,xmm0
    mov r9d,r8d
    shl r9d,6
    lea r10,[rip+X]
    add r10,r9
    lea r11,[rip+X+448]
    xor ecx,ecx
.Lscoredim:
    movss xmm1,[r10+rcx*4]
    mulss xmm1,[r11+rcx*4]
    addss xmm0,xmm1
    inc ecx
    cmp ecx,D
    jb .Lscoredim
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
    lea r10,[rip+X]
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
    lea r10,[rip+FB]
    movss xmm0,[r10+r8*4]
    mov r9d,r8d
    shl r9d,4
    lea r10,[rip+F]
    lea r11,[r10+r9*4]
    lea r10,[rip+H]
    xor ecx,ecx
.Lffdot:
    movss xmm1,[r11+rcx*4]
    mulss xmm1,[r10+rcx*4]
    addss xmm0,xmm1
    inc ecx
    cmp ecx,D
    jb .Lffdot
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
    lea r10,[rip+OB]
    movss xmm0,[r10+r8*4]
    mov r9d,r8d
    shl r9d,4
    lea r10,[rip+O]
    lea r11,[r10+r9*4]
    lea r10,[rip+Y]
    xor ecx,ecx
.Loutdot:
    movss xmm1,[r11+rcx*4]
    mulss xmm1,[r10+rcx*4]
    addss xmm0,xmm1
    inc ecx
    cmp ecx,D
    jb .Loutdot
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
# Reverse-mode derivatives of trained E, POS, F, FB, O, OB.
# The query/key/value projections are tied identities, not learned QKV matrices.
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
    # Value path: gradX[t,j]=A[t]*GH[j], GA[t]=dot(GH,X[t]).
    xor r8d,r8d
.Lvaluet:
    mov r9d,r8d
    shl r9d,4
    lea r10,[rip+X]
    lea r11,[r10+r9*4]
    lea r10,[rip+GH]
    lea rdi,[rip+GX]
    lea rdi,[rdi+r9*4]
    lea rsi,[rip+A]
    movss xmm2,[rsi+r8*4]
    xorps xmm3,xmm3
    xor ecx,ecx
.Lvaluej:
    movss xmm0,[r10+rcx*4]
    movss xmm1,[r11+rcx*4]
    mulss xmm0,xmm2
    movss [rdi+rcx*4],xmm0
    movss xmm0,[r10+rcx*4]
    mulss xmm0,xmm1
    addss xmm3,xmm0
    inc ecx
    cmp ecx,D
    jb .Lvaluej
    lea r10,[rip+GA]
    movss [r10+r8*4],xmm3
    inc r8d
    cmp r8d,CTX
    jb .Lvaluet
    xorps xmm7,xmm7
    xor ecx,ecx
.Lgsum:
    lea r10,[rip+A]
    movss xmm0,[r10+rcx*4]
    lea r10,[rip+GA]
    mulss xmm0,[r10+rcx*4]
    addss xmm7,xmm0
    inc ecx
    cmp ecx,CTX
    jb .Lgsum
    xor r8d,r8d
.Lgradscoret:
    lea r10,[rip+GA]
    movss xmm2,[r10+r8*4]
    subss xmm2,xmm7
    lea r10,[rip+A]
    mulss xmm2,[r10+r8*4]
    mulss xmm2,[rip+quarter]
    mov r9d,r8d
    shl r9d,4
    lea r10,[rip+X]
    lea r11,[r10+r9*4]
    lea r10,[rip+GX]
    lea rdi,[r10+r9*4]
    lea rsi,[rip+GX+448]
    lea rdx,[rip+X+448]
    xor ecx,ecx
.Lgradscorej:
    movss xmm0,[rdx+rcx*4]
    mulss xmm0,xmm2
    addss xmm0,[rdi+rcx*4]
    movss [rdi+rcx*4],xmm0
    movss xmm1,[r11+rcx*4]
    mulss xmm1,xmm2
    addss xmm1,[rsi+rcx*4]
    movss [rsi+rcx*4],xmm1
    inc ecx
    cmp ecx,D
    jb .Lgradscorej
    inc r8d
    cmp r8d,CTX
    jb .Lgradscoret
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
