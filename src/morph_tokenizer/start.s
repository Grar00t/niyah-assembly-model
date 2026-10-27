.file "niyah_morph_tokenizer_start.s"
# Linux x86-64 process entry; no CRT or libc.
.intel_syntax noprefix
.global _start
.section .text
_start:
    xor ebp, ebp
    mov rdi, [rsp]
    lea rsi, [rsp+8]
    and rsp, -16
    call tokenizer_main
    mov edi, eax
    mov eax, 60
    syscall
    ud2
.section .note.GNU-stack,"",@progbits
