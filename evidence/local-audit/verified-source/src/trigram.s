.intel_syntax noprefix
.equ EPOCHS, 201
.ifndef SEED
.equ SEED, 2463534242
.endif
.ifndef SAMPLE_LEN
.equ SAMPLE_LEN, 90
.endif
.section .data
corpus:
    .incbin "data/corpus.txt"
.equ N, . - corpus
.if N < 3
.error "corpus requires at least 3 bytes"
.endif
lr: .float 0.1
one: .float 1.0
thousand: .float 1000.0
inv24: .float 5.9604644775390625e-8
s_epoch: .ascii "epoch "
s_loss: .ascii "loss*1000 "
s_gen: .ascii "--- sample ---\n"
.section .bss
.balign 16
.ifdef INFERENCE
.section .data
.balign 16
W:
    .incbin "models/weights.f32"
.if (. - W) != 67108864
.error "invalid checkpoint size"
.endif
.section .bss
.else
W: .skip 67108864
.endif
P: .skip 1024
allowed: .skip 256
tmp: .skip 16
nbuf: .skip 32
.section .text
.global _start
exp_f:
    movss [rip+tmp], xmm0
    fld dword ptr [rip+tmp]
    fldl2e
    fmulp st(1), st
    fld st(0)
    frndint
    fxch st(1)
    fsub st(0), st(1)
    f2xm1
    fld1
    faddp st(1), st
    fscale
    fstp st(1)
    fstp dword ptr [rip+tmp]
    movss xmm0, [rip+tmp]
    ret
log_f:
    movss [rip+tmp], xmm0
    fldln2
    fld dword ptr [rip+tmp]
    fyl2x
    fstp dword ptr [rip+tmp]
    movss xmm0, [rip+tmp]
    ret
numeric_fail:
    mov eax, 60
    mov edi, 3
    syscall
softmax_row:
    lea rdi, [rip+P]
    lea r10, [rip+allowed]
    xor ecx, ecx
.Lfirst:
    cmp byte ptr [r10+rcx], 0
    jne .Lfound
    inc ecx
    cmp ecx, 256
    jb .Lfirst
    jmp numeric_fail
.Lfound:
    movss xmm4, [rsi+rcx*4]
    mov r11d, ecx
.Lmax:
    cmp byte ptr [r10+rcx], 0
    je .Lmax_next
    mov eax, [rsi+rcx*4]
    and eax, 0x7f800000
    cmp eax, 0x7f800000
    je numeric_fail
    maxss xmm4, [rsi+rcx*4]
    mov r11d, ecx
.Lmax_next:
    inc ecx
    cmp ecx, 256
    jb .Lmax
    xorps xmm8, xmm8
    xor ecx, ecx
.Lexp:
    xorps xmm0, xmm0
    cmp byte ptr [r10+rcx], 0
    je .Lstore
    movss xmm0, [rsi+rcx*4]
    subss xmm0, xmm4
    movd eax, xmm0
    cmp eax, 0xff800000
    jne .Lexp_finite
    xorps xmm0, xmm0
    jmp .Lstore
.Lexp_finite:
    call exp_f
.Lstore:
    movss [rdi+rcx*4], xmm0
    addss xmm8, xmm0
    inc ecx
    cmp ecx, 256
    jb .Lexp
    xor ecx, ecx
.Lnormalize:
    movss xmm0, [rdi+rcx*4]
    divss xmm0, xmm8
    movss [rdi+rcx*4], xmm0
    inc ecx
    cmp ecx, 256
    jb .Lnormalize
    ret
write_all:
    test edx, edx
    jz .Lwrite_done
.Lwrite:
    mov eax, 1
    mov edi, 1
    syscall
    cmp rax, -4
    je .Lwrite
    test rax, rax
    jle .Lwrite_fail
    add rsi, rax
    sub rdx, rax
    jnz .Lwrite
.Lwrite_done:
    ret
.Lwrite_fail:
    mov eax, 60
    mov edi, 4
    syscall
print_u:
    lea r8, [rip+nbuf+31]
    mov byte ptr [r8], r9b
    mov ecx, 10
.Ldigit:
    xor edx, edx
    div ecx
    add dl, '0'
    dec r8
    mov [r8], dl
    test eax, eax
    jnz .Ldigit
    lea rdx, [rip+nbuf+32]
    sub rdx, r8
    mov rsi, r8
    call write_all
    ret
.macro PRINTSTR s, n
    lea rsi, [rip+\s]
    mov edx, \n
    call write_all
.endm
_start:
    fninit
    lea r8, [rip+corpus]
    lea r10, [rip+allowed]
    xor ecx, ecx
.Lvocab:
    movzx eax, byte ptr [r8+rcx]
    mov byte ptr [r10+rax], 1
    inc ecx
    cmp ecx, N
    jb .Lvocab
    .ifdef INFERENCE
    jmp sample_start
    .endif
    xor r12d, r12d
epoch:
    xorps xmm10, xmm10
    xor r13d, r13d
pair:
    cmp r13d, N-2
    jae pdone
    lea r8, [rip+corpus]
    movzx eax, byte ptr [r8+r13]
    shl eax, 8
    movzx edx, byte ptr [r8+r13+1]
    or eax, edx
    movzx ebx, byte ptr [r8+r13+2]
    shl eax, 10
    lea rsi, [rip+W]
    add rsi, rax
    call softmax_row
    movss xmm0, xmm8
    call log_f
    movss xmm1, [rsi+rbx*4]
    subss xmm1, xmm4
    subss xmm0, xmm1
    addss xmm10, xmm0
    movss xmm9, [rip+lr]
    movss xmm11, [rip+one]
    xor ecx, ecx
upd:
    movss xmm0, [rdi+rcx*4]
    cmp ecx, ebx
    jne .Lnot_target
    subss xmm0, xmm11
.Lnot_target:
    mulss xmm0, xmm9
    movss xmm1, [rsi+rcx*4]
    subss xmm1, xmm0
    movss [rsi+rcx*4], xmm1
    inc ecx
    cmp ecx, 256
    jb upd
    inc r13d
    jmp pair
pdone:
    .ifdef EXPORT_WEIGHTS
    jmp skip
    .endif
    mov eax, r12d
    xor edx, edx
    mov ecx, 20
    div ecx
    test edx, edx
    jnz skip
    mov eax, N-2
    cvtsi2ss xmm1, eax
    divss xmm10, xmm1
    mulss xmm10, [rip+thousand]
    cvtss2si r14d, xmm10
    PRINTSTR s_epoch, 6
    mov eax, r12d
    mov r9d, ' '
    call print_u
    PRINTSTR s_loss, 10
    mov eax, r14d
    mov r9d, 10
    call print_u
skip:
    inc r12d
    cmp r12d, EPOCHS
    jb epoch
    .ifdef EXPORT_WEIGHTS
checkpoint_export:
    lea rsi, [rip+W]
    mov edx, 67108864
    call write_all
    mov eax, 60
    xor edi, edi
    syscall
    .endif

sample_start:
    PRINTSTR s_gen, 15
    mov r14d, SEED
    movzx r13d, byte ptr [rip+corpus]
    shl r13d, 8
    movzx eax, byte ptr [rip+corpus+1]
    or r13d, eax
    mov r15d, SAMPLE_LEN
gen:
    test r15d, r15d
    jz .Lend_sample
    mov eax, r13d
    shl eax, 10
    lea rsi, [rip+W]
    add rsi, rax
    call softmax_row
    mov ebp, r11d
    mov eax, r14d
    mov edx, eax
    shl edx, 13
    xor eax, edx
    mov edx, eax
    shr edx, 17
    xor eax, edx
    mov edx, eax
    shl edx, 5
    xor eax, edx
    mov r14d, eax
    and eax, 0xFFFFFF
    cvtsi2ss xmm2, eax
    mulss xmm2, [rip+inv24]
    xorps xmm3, xmm3
    xor ebx, ebx
pick:
    cmp byte ptr [r10+rbx], 0
    je .Lpick_next
    addss xmm3, [rdi+rbx*4]
    ucomiss xmm3, xmm2
    jp numeric_fail
    ja picked
.Lpick_next:
    inc ebx
    cmp ebx, 256
    jb pick
    mov ebx, ebp
picked:
    mov [rip+nbuf], bl
    shl r13d, 8
    or r13d, ebx
    and r13d, 0xffff
    lea rsi, [rip+nbuf]
    mov edx, 1
    call write_all
    dec r15d
    jmp gen
.Lend_sample:
    mov byte ptr [rip+nbuf], 10
    lea rsi, [rip+nbuf]
    mov edx, 1
    call write_all
    mov eax, 60
    xor edi, edi
    syscall
.section .note.GNU-stack,"",@progbits
