# Niyah exact rational arithmetic: bounded signed 32-bit input fractions.
# Native Linux x86-64 Assembly, no libc, no malloc, no external dependencies.
# Public domain mathematics (Euclidean GCD). Original implementation.
# Result is an exactly reduced signed int64 numerator / positive int64 denominator.
# Input domain: numerators [-2147483648, 2147483647],
#               denominators [1, 2147483647]. No intermediate exceeds int64.
.intel_syntax noprefix
.section .rodata
usage_text:
    .ascii "usage: exactcalc add|sub|mul|div NUM1 DEN1 NUM2 DEN2\n"
    .ascii "NUM: signed 32-bit decimal; DEN: 1..2147483647; result: reduced exact fraction\n"
.equ usage_len, . - usage_text
invalid_text: .ascii "error: invalid operation or integer/denominator\n"
.equ invalid_len, . - invalid_text
zero_text: .ascii "error: division by zero\n"
.equ zero_len, . - zero_text
overflow_text: .ascii "error: intermediate integer overflow\n"
.equ overflow_len, . - overflow_text
minus: .ascii "-"
slash: .ascii "/"
newline: .ascii "\n"

.section .bss
opcode: .skip 1
.balign 16
digits: .skip 48
digits_end_ptr:

.section .text
.global _start
_start:
    cmp qword ptr [rsp], 2
    jne .Lnormal_args
    mov rdi, [rsp+16]
    cmp byte ptr [rdi], '-'
    jne .Lnormal_args
    cmp byte ptr [rdi+1], '-'
    jne .Lnormal_args
    cmp byte ptr [rdi+2], 'h'
    jne .Lnormal_args
    cmp byte ptr [rdi+3], 'e'
    jne .Lnormal_args
    cmp byte ptr [rdi+4], 'l'
    jne .Lnormal_args
    cmp byte ptr [rdi+5], 'p'
    jne .Lnormal_args
    cmp byte ptr [rdi+6], 0
    jne .Lnormal_args
    lea rsi, [rip+usage_text]
    mov edx, usage_len
    mov edi, 1
    call write_all
    xor edi, edi
    jmp exit

.Lnormal_args:
    cmp qword ptr [rsp], 6
    jne usage_error
    mov rdi, [rsp+16]
    call parse_op
    mov [rip+opcode], al

    mov rdi, [rsp+24]
    call parse_i32
    mov r12, rax
    mov rdi, [rsp+32]
    call parse_i32
    test rax, rax
    jle invalid
    mov r13, rax

    mov rdi, [rsp+40]
    call parse_i32
    mov r14, rax
    mov rdi, [rsp+48]
    call parse_i32
    test rax, rax
    jle invalid
    mov r15, rax

    movzx eax, byte ptr [rip+opcode]
    cmp al, 'a'
    je .Ladd
    cmp al, 's'
    je .Lsub
    cmp al, 'm'
    je .Lmul
    cmp al, 'd'
    je .Ldiv
    jmp invalid

.Ladd:
    # numerator = n1*d2 + n2*d1 (both products fit signed 64-bit).
    mov rax, r12
    imul rax, r15
    jo overflow
    mov rbx, rax
    mov rax, r14
    imul rax, r13
    jo overflow
    add rax, rbx
    jo overflow
    mov r12, rax
    jmp .Lden_mul
.Lsub:
    mov rax, r12
    imul rax, r15
    jo overflow
    mov rbx, rax
    mov rax, r14
    imul rax, r13
    jo overflow
    sub rbx, rax
    jo overflow
    mov r12, rbx
    jmp .Lden_mul
.Lmul:
    mov rax, r12
    imul rax, r14
    jo overflow
    mov r12, rax
    jmp .Lden_mul
.Ldiv:
    test r14, r14
    jz divzero
    mov rax, r12
    imul rax, r15
    jo overflow
    mov r12, rax
    mov rax, r13
    imul rax, r14
    jo overflow
    mov r13, rax
    # Divisor sign is moved onto the numerator; denominator stays positive.
    test r13, r13
    jns .Lreduce
    neg r13
    jo overflow
    neg r12
    jo overflow
    jmp .Lreduce

.Lden_mul:
    imul r13, r15
    jo overflow
.Lreduce:
    # Canonical zero is 0/1.
    test r12, r12
    jnz .Lgcd_init
    mov r13, 1
    jmp .Loutput
.Lgcd_init:
    # Euclid's GCD for nonnegative unsigned 64-bit magnitudes.
    # Each operand is strictly below 2^63 from the proven input bounds.
    mov r8, r12
    test r8, r8
    jns .Lgcd_nonnegative
    neg r8
    jo overflow
.Lgcd_nonnegative:
    mov r9, r13
.Lgcd_loop:
    mov rax, r8
    xor edx, edx
    div r9
    mov r8, r9
    mov r9, rdx
    test r9, r9
    jnz .Lgcd_loop
    # Signed numerator division and positive denominator division are exact.
    mov rax, r12
    cqo
    idiv r8
    test rdx, rdx
    jnz overflow
    mov r12, rax
    mov rax, r13
    xor edx, edx
    div r8
    test rdx, rdx
    jnz overflow
    mov r13, rax

.Loutput:
    mov rax, r12
    test rax, rax
    jns .Lpositive
    neg rax
    jo overflow
    mov rbx, rax
    lea rsi, [rip+minus]
    mov edx, 1
    mov edi, 1
    call write_all
    mov rax, rbx
.Lpositive:
    call print_u64
    lea rsi, [rip+slash]
    mov edx, 1
    mov edi, 1
    call write_all
    mov rax, r13
    call print_u64
    lea rsi, [rip+newline]
    mov edx, 1
    mov edi, 1
    call write_all
    xor edi, edi
    jmp exit

# parse_op: exact ASCII words, no prefix acceptance.
# rdi -> zero-terminated input; returns al ('a','s','m','d') or exits 2.
parse_op:
    cmp byte ptr [rdi], 'a'
    je .Lop_add
    cmp byte ptr [rdi], 's'
    je .Lop_sub
    cmp byte ptr [rdi], 'm'
    je .Lop_mul
    cmp byte ptr [rdi], 'd'
    je .Lop_div
    jmp invalid
.Lop_add:
    cmp byte ptr [rdi+1], 'd'
    jne invalid
    cmp byte ptr [rdi+2], 'd'
    jne invalid
    cmp byte ptr [rdi+3], 0
    jne invalid
    mov al, 'a'
    ret
.Lop_sub:
    cmp byte ptr [rdi+1], 'u'
    jne invalid
    cmp byte ptr [rdi+2], 'b'
    jne invalid
    cmp byte ptr [rdi+3], 0
    jne invalid
    mov al, 's'
    ret
.Lop_mul:
    cmp byte ptr [rdi+1], 'u'
    jne invalid
    cmp byte ptr [rdi+2], 'l'
    jne invalid
    cmp byte ptr [rdi+3], 0
    jne invalid
    mov al, 'm'
    ret
.Lop_div:
    cmp byte ptr [rdi+1], 'i'
    jne invalid
    cmp byte ptr [rdi+2], 'v'
    jne invalid
    cmp byte ptr [rdi+3], 0
    jne invalid
    mov al, 'd'
    ret

# parse_i32: bounded scan of <=10 ASCII decimal digits, optional '-' sign.
# rdi -> zero-terminated string; returns sign-extended int32 in rax.
# Rejects empty tokens, '+' prefix, whitespace, non-ASCII decimal digits,
# additional suffix bytes, and signed-32 overflow.
parse_i32:
    mov rsi, rdi
    xor r9d, r9d
    cmp byte ptr [rsi], '-'
    jne .Lparse_start
    inc rsi
    mov r9d, 1
.Lparse_start:
    xor eax, eax
    xor r8d, r8d
.Lparse_char:
    movzx edx, byte ptr [rsi]
    test edx, edx
    jz .Lparse_done
    cmp r8d, 10
    jae invalid
    sub edx, '0'
    cmp edx, 9
    ja invalid
    cmp rax, 214748364
    ja invalid
    jb .Lappend
    cmp edx, 7
    jbe .Lappend
    cmp r9d, 1
    jne invalid
    cmp edx, 8
    ja invalid
.Lappend:
    imul rax, rax, 10
    add rax, rdx
    inc rsi
    inc r8d
    jmp .Lparse_char
.Lparse_done:
    test r8d, r8d
    jz invalid
    test r9d, r9d
    jz .Lparse_return
    neg rax
.Lparse_return:
    ret

# print_u64: rax unsigned nonnegative (caller handles minus).
print_u64:
    lea rsi, [rip+digits_end_ptr]
    mov ecx, 10
.Ldigit:
    xor edx, edx
    div rcx
    add dl, '0'
    dec rsi
    mov [rsi], dl
    test rax, rax
    jnz .Ldigit
    lea rdx, [rip+digits_end_ptr]
    sub rdx, rsi
    mov edi, 1
    call write_all
    ret

# write_all: fd=edi, pointer=rsi, count=rdx. EINTR/short-write safe.
# A zero-byte write, non-EINTR failure, or EPIPE returns exit status 5.
write_all:
    mov r8d, edi
.Lwrite_loop:
    test rdx, rdx
    jz .Lwrite_done
    mov eax, 1
    mov edi, r8d
    syscall
    cmp rax, -4
    je .Lwrite_loop
    test rax, rax
    jle write_failure
    add rsi, rax
    sub rdx, rax
    jmp .Lwrite_loop
.Lwrite_done:
    ret

usage_error:
    lea rsi, [rip+usage_text]
    mov edx, usage_len
    mov edi, 2
    call write_all
    mov edi, 2
    jmp exit
invalid:
    lea rsi, [rip+invalid_text]
    mov edx, invalid_len
    mov edi, 2
    call write_all
    mov edi, 2
    jmp exit
divzero:
    lea rsi, [rip+zero_text]
    mov edx, zero_len
    mov edi, 2
    call write_all
    mov edi, 3
    jmp exit
overflow:
    lea rsi, [rip+overflow_text]
    mov edx, overflow_len
    mov edi, 2
    call write_all
    mov edi, 4
    jmp exit
write_failure:
    mov edi, 5
exit:
    mov eax, 60
    syscall
.section .note.GNU-stack,"",@progbits
