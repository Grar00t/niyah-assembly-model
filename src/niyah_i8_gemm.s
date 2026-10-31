.file "niyah_i8_gemm.s"
.intel_syntax noprefix

# Blocked signed-int8 GEMM for the Qwen layer-0 linears.
# C[i,j] = sum_k A[i,k] * B[j,k] as int32, row-major.
# A is M x K, B is N x K, C is M x N. K is a positive multiple of 16 and <= 16384.
# M, N, M*N, M*K, and N*K fit in signed int32. No aliasing of C with A or B.
# Products match niyah_i8_dot: vpmovsxbw + vpmaddwd + vpaddd, total is order-independent
# inside int32 for the layer contract (K <= 12288, |product| <= 16384).
#
# A panel of 8 (then 4) rows is widened to i16 once and reused across every B row.
# That is the layer-0 case: M is the token count (<= 16) and N*K is the weight matrix,
# which does not fit in L3 and was previously re-read once per token.
# ai16 is single-threaded scratch. AVX2 is assumed; the caller checks the CPU.
# Returns 0, or 64 when the geometry is rejected.

.section .bss
.balign 64
ai16: .zero 262144

.section .text
.global niyah_i8_gemm
.type niyah_i8_gemm, @function

.macro hsum_add acc, low
    vextracti128 xmm14, \acc, 1
    vpaddd xmm14, xmm14, \low
    vpshufd xmm15, xmm14, 0x4e
    vpaddd xmm14, xmm14, xmm15
    vpshufd xmm15, xmm14, 0xb1
    vpaddd xmm14, xmm14, xmm15
    vmovd eax, xmm14
    mov dword ptr [rdi+rsi*4], eax
    add esi, r13d
.endm

niyah_i8_gemm:
    test ecx, ecx
    jle .Lbad
    test r8d, r8d
    jle .Lbad
    test r9d, r9d
    jle .Lbad
    test r9d, 15
    jnz .Lbad
    cmp r9d, 16384
    ja .Lbad
    mov eax, ecx
    imul eax, r8d
    jo .Lbad
    mov eax, ecx
    imul eax, r9d
    jo .Lbad
    mov eax, r8d
    imul eax, r9d
    jo .Lbad
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 40
    mov [rsp], rdi
    mov [rsp+8], rsi
    mov [rsp+16], rdx
    mov r12d, ecx
    mov r13d, r8d
    mov r14d, r9d
    mov dword ptr [rsp+24], 0

.Lpanel:
    mov eax, r12d
    sub eax, dword ptr [rsp+24]
    cmp eax, 8
    jae .Lpanel8
    cmp eax, 4
    jae .Lpanel4
    test eax, eax
    jz .Lok
    jmp .Lpanel1

.Lpanel8:
    mov rbx, [rsp]
    mov eax, dword ptr [rsp+24]
    imul eax, r14d
    add rbx, rax
    lea r10, [rip+ai16]
    mov r8d, 8
.Lpc8_row:
    xor ecx, ecx
.Lpc8_k:
    vpmovsxbw ymm0, XMMWORD PTR [rbx+rcx]
    vmovdqu YMMWORD PTR [r10+rcx*2], ymm0
    add ecx, 16
    cmp ecx, r14d
    jb .Lpc8_k
    add rbx, r14
    mov eax, r14d
    add eax, eax
    add r10, rax
    dec r8d
    jnz .Lpc8_row
    xor r15d, r15d
.L8n:
    mov eax, r14d
    add eax, eax
    lea rbx, [rip+ai16]
    lea r8, [rbx+rax]
    lea r9, [r8+rax]
    lea r10, [r9+rax]
    lea r11, [r10+rax]
    lea rsi, [r11+rax]
    lea rdi, [rsi+rax]
    lea rdx, [rdi+rax]
    mov rax, r15
    imul rax, r14
    add rax, [rsp+8]
    vpxor ymm0, ymm0, ymm0
    vpxor ymm1, ymm1, ymm1
    vpxor ymm2, ymm2, ymm2
    vpxor ymm3, ymm3, ymm3
    vpxor ymm4, ymm4, ymm4
    vpxor ymm5, ymm5, ymm5
    vpxor ymm6, ymm6, ymm6
    vpxor ymm7, ymm7, ymm7
    xor ecx, ecx
.p2align 4
.L8k:
    prefetcht0 BYTE PTR [rax+rcx+128]
    vpmovsxbw ymm8, XMMWORD PTR [rax+rcx]
    vpmaddwd ymm9, ymm8, YMMWORD PTR [rbx+rcx*2]
    vpaddd ymm0, ymm0, ymm9
    vpmaddwd ymm9, ymm8, YMMWORD PTR [r8+rcx*2]
    vpaddd ymm1, ymm1, ymm9
    vpmaddwd ymm9, ymm8, YMMWORD PTR [r9+rcx*2]
    vpaddd ymm2, ymm2, ymm9
    vpmaddwd ymm9, ymm8, YMMWORD PTR [r10+rcx*2]
    vpaddd ymm3, ymm3, ymm9
    vpmaddwd ymm9, ymm8, YMMWORD PTR [r11+rcx*2]
    vpaddd ymm4, ymm4, ymm9
    vpmaddwd ymm9, ymm8, YMMWORD PTR [rsi+rcx*2]
    vpaddd ymm5, ymm5, ymm9
    vpmaddwd ymm9, ymm8, YMMWORD PTR [rdi+rcx*2]
    vpaddd ymm6, ymm6, ymm9
    vpmaddwd ymm9, ymm8, YMMWORD PTR [rdx+rcx*2]
    vpaddd ymm7, ymm7, ymm9
    add ecx, 16
    cmp ecx, r14d
    jb .L8k
    mov rdi, [rsp+16]
    mov esi, dword ptr [rsp+24]
    imul esi, r13d
    add esi, r15d
    hsum_add ymm0, xmm0
    hsum_add ymm1, xmm1
    hsum_add ymm2, xmm2
    hsum_add ymm3, xmm3
    hsum_add ymm4, xmm4
    hsum_add ymm5, xmm5
    hsum_add ymm6, xmm6
    hsum_add ymm7, xmm7
    inc r15d
    cmp r15d, r13d
    jb .L8n
    add dword ptr [rsp+24], 8
    jmp .Lpanel

.Lpanel4:
    mov rbx, [rsp]
    mov eax, dword ptr [rsp+24]
    imul eax, r14d
    add rbx, rax
    lea r10, [rip+ai16]
    mov r8d, 4
.Lpc4_row:
    xor ecx, ecx
.Lpc4_k:
    vpmovsxbw ymm0, XMMWORD PTR [rbx+rcx]
    vmovdqu YMMWORD PTR [r10+rcx*2], ymm0
    add ecx, 16
    cmp ecx, r14d
    jb .Lpc4_k
    add rbx, r14
    mov eax, r14d
    add eax, eax
    add r10, rax
    dec r8d
    jnz .Lpc4_row
    xor r15d, r15d
.L4n:
    mov eax, r14d
    add eax, eax
    lea rbx, [rip+ai16]
    lea r8, [rbx+rax]
    lea r9, [r8+rax]
    lea r10, [r9+rax]
    mov rax, r15
    imul rax, r14
    add rax, [rsp+8]
    vpxor ymm0, ymm0, ymm0
    vpxor ymm1, ymm1, ymm1
    vpxor ymm2, ymm2, ymm2
    vpxor ymm3, ymm3, ymm3
    xor ecx, ecx
.p2align 4
.L4k:
    prefetcht0 BYTE PTR [rax+rcx+128]
    vpmovsxbw ymm8, XMMWORD PTR [rax+rcx]
    vpmaddwd ymm9, ymm8, YMMWORD PTR [rbx+rcx*2]
    vpaddd ymm0, ymm0, ymm9
    vpmaddwd ymm9, ymm8, YMMWORD PTR [r8+rcx*2]
    vpaddd ymm1, ymm1, ymm9
    vpmaddwd ymm9, ymm8, YMMWORD PTR [r9+rcx*2]
    vpaddd ymm2, ymm2, ymm9
    vpmaddwd ymm9, ymm8, YMMWORD PTR [r10+rcx*2]
    vpaddd ymm3, ymm3, ymm9
    add ecx, 16
    cmp ecx, r14d
    jb .L4k
    mov rdi, [rsp+16]
    mov esi, dword ptr [rsp+24]
    imul esi, r13d
    add esi, r15d
    hsum_add ymm0, xmm0
    hsum_add ymm1, xmm1
    hsum_add ymm2, xmm2
    hsum_add ymm3, xmm3
    inc r15d
    cmp r15d, r13d
    jb .L4n
    add dword ptr [rsp+24], 4
    jmp .Lpanel

.Lpanel1:
    mov rbx, [rsp]
    mov eax, dword ptr [rsp+24]
    imul eax, r14d
    add rbx, rax
    xor r15d, r15d
.L1n:
    mov rax, r15
    imul rax, r14
    add rax, [rsp+8]
    vpxor ymm2, ymm2, ymm2
    xor ecx, ecx
.L1k:
    vpmovsxbw ymm0, XMMWORD PTR [rbx+rcx]
    vpmovsxbw ymm1, XMMWORD PTR [rax+rcx]
    vpmaddwd ymm0, ymm0, ymm1
    vpaddd ymm2, ymm2, ymm0
    add ecx, 16
    cmp ecx, r14d
    jb .L1k
    vextracti128 xmm1, ymm2, 1
    vpaddd xmm0, xmm2, xmm1
    vpshufd xmm1, xmm0, 0x4e
    vpaddd xmm0, xmm0, xmm1
    vpshufd xmm1, xmm0, 0xb1
    vpaddd xmm0, xmm0, xmm1
    vmovd eax, xmm0
    mov rdi, [rsp+16]
    mov edx, dword ptr [rsp+24]
    imul edx, r13d
    add edx, r15d
    mov dword ptr [rdi+rdx*4], eax
    inc r15d
    cmp r15d, r13d
    jb .L1n
    add dword ptr [rsp+24], 1
    jmp .Lpanel

.Lok:
    xor eax, eax
    vzeroupper
    add rsp, 40
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.Lbad:
    mov eax, 64
    ret
.size niyah_i8_gemm, .-niyah_i8_gemm
.section .note.GNU-stack, "", @progbits
