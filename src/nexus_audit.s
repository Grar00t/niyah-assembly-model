# Native, bounded arithmetic candidate verifier using the exactcalc sibling.
# Linux x86-64 syscalls only; no shell, PATH search, libc, or neural guarantee.
.intel_syntax noprefix
.file "nexus_audit.s"
.section .rodata
self_path: .asciz "/proc/self/exe"
calculator: .asciz "exactcalc"
verified: .ascii "VERIFIED "
.equ verified_len,.-verified
mismatch: .ascii "MISMATCH\n"
.equ mismatch_len,.-mismatch
usage: .ascii "usage: nexus_audit add|sub|mul|div N1 D1 N2 D2 CANDIDATE\n"
.equ usage_len,.-usage
failed: .ascii "calculator or native I/O protocol failure\n"
.equ failed_len,.-failed
.balign 8
ignore_sigpipe: .quad 1,0,0,0
.section .bss
.balign 8
original_args: .skip 8
exe_path: .skip 4096
child_args: .skip 56
empty_env: .skip 8
pipe_fds: .skip 8
child_pid: .skip 8
wait_status: .skip 4
answer_len: .skip 4
overflowed: .skip 4
answer: .skip 128
.section .text
.global _start
_start:
    cld
    mov [rip+original_args],rsp
    cmp qword ptr [rsp],7
    jne usage_error
    # Ignore SIGPIPE so stdout failures are observable as exit4.
    mov eax,13
    mov edi,13
    lea rsi,[rip+ignore_sigpipe]
    xor edx,edx
    mov r10d,8
    syscall
    test rax,rax
    js calc_error
    # Derive trusted-location sibling; never interpret a shell command/PATH.
    mov eax,89
    lea rdi,[rip+self_path]
    lea rsi,[rip+exe_path]
    mov edx,4096
    syscall
    test rax,rax
    jle calc_error
    cmp rax,4096
    jae calc_error
    lea rdi,[rip+exe_path]
    mov rcx,rax
.Lfind_slash:
    test rcx,rcx
    jz calc_error
    dec rcx
    cmp byte ptr [rdi+rcx],'/'
    jne .Lfind_slash
    inc rcx
    cmp ecx,4086
    ja calc_error
    add rdi,rcx
    lea rsi,[rip+calculator]
    mov ecx,10
    rep movsb
    lea rax,[rip+exe_path]
    mov [rip+child_args],rax
    mov rsi,[rip+original_args]
    lea rdi,[rip+child_args+8]
    add rsi,16
    mov ecx,5
    rep movsq
    mov eax,22
    lea rdi,[rip+pipe_fds]
    syscall
    test rax,rax
    js calc_error
    mov eax,57
    syscall
    test rax,rax
    js calc_error
    jz child
    mov [rip+child_pid],rax
    mov eax,3
    mov edi,[rip+pipe_fds+4]
    syscall
    test rax,rax
    js calc_error
    xor ebx,ebx
.Lread:
    cmp ebx,128
    jae .Loverflow
    xor eax,eax
    mov edi,[rip+pipe_fds]
    lea rsi,[rip+answer]
    add rsi,rbx
    mov edx,128
    sub edx,ebx
    syscall
    cmp rax,-4
    je .Lread
    test rax,rax
    js calc_error
    jz .Lread_done
    add ebx,eax
    jmp .Lread
.Loverflow:
    mov dword ptr [rip+overflowed],1
.Lread_done:
    mov [rip+answer_len],ebx
    mov eax,3
    mov edi,[rip+pipe_fds]
    syscall
    test rax,rax
    js calc_error
.Lwait:
    mov eax,61
    mov rdi,[rip+child_pid]
    lea rsi,[rip+wait_status]
    xor edx,edx
    xor r10d,r10d
    syscall
    cmp rax,-4
    je .Lwait
    cmp rax,[rip+child_pid]
    jne calc_error
    cmp dword ptr [rip+wait_status],0
    jne calc_error
    cmp dword ptr [rip+overflowed],0
    jne calc_error
    # Strict child framing: signed decimal numerator / positive denominator LF.
    mov edx,[rip+answer_len]
    cmp edx,4
    jb calc_error
    cmp edx,43
    ja calc_error
    lea rsi,[rip+answer]
    dec edx
    cmp byte ptr [rsi+rdx],10
    jne calc_error
    xor ecx,ecx
    cmp byte ptr [rsi],'-'
    jne .Lnumerator_start
    inc ecx
.Lnumerator_start:
    mov r8d,ecx
    movzx eax,byte ptr [rsi+rcx]
    cmp al,'0'
    jb calc_error
    cmp al,'9'
    ja calc_error
    cmp al,'0'
    jne .Lnumerator_digits
    test ecx,ecx
    jnz calc_error
    cmp byte ptr [rsi+rcx+1],'/'
    jne calc_error
.Lnumerator_digits:
    cmp ecx,edx
    jae calc_error
    movzx eax,byte ptr [rsi+rcx]
    cmp al,'/'
    je .Lslash
    sub eax,'0'
    cmp eax,9
    ja calc_error
    inc ecx
    jmp .Lnumerator_digits
.Lslash:
    cmp ecx,r8d
    je calc_error
    inc ecx
    cmp ecx,edx
    jae calc_error
    cmp byte ptr [rsi+rcx],'1'
    jb calc_error
    cmp byte ptr [rsi+rcx],'9'
    ja calc_error
.Lden_digits:
    cmp ecx,edx
    jae .Lcompare
    movzx eax,byte ptr [rsi+rcx]
    sub eax,'0'
    cmp eax,9
    ja calc_error
    inc ecx
    jmp .Lden_digits
.Lcompare:
    mov rax,[rip+original_args]
    mov rdi,[rax+56]
    xor ecx,ecx
.Lcompare_loop:
    cmp ecx,edx
    jae .Lcandidate_end
    mov al,[rdi+rcx]
    cmp al,[rsi+rcx]
    jne mismatch_exit
    inc ecx
    jmp .Lcompare_loop
.Lcandidate_end:
    cmp byte ptr [rdi+rcx],0
    jne mismatch_exit
    mov edi,1
    lea rsi,[rip+verified]
    mov edx,verified_len
    call write_all
    mov edi,1
    lea rsi,[rip+answer]
    mov edx,[rip+answer_len]
    call write_all
    xor edi,edi
    jmp exit

child:
    mov eax,3
    mov edi,[rip+pipe_fds]
    syscall
    mov eax,33
    mov edi,[rip+pipe_fds+4]
    mov esi,1
    syscall
    test rax,rax
    js child_error
    cmp dword ptr [rip+pipe_fds+4],1
    je .Lexec
    mov eax,3
    mov edi,[rip+pipe_fds+4]
    syscall
.Lexec:
    mov eax,59
    lea rdi,[rip+exe_path]
    lea rsi,[rip+child_args]
    lea rdx,[rip+empty_env]
    syscall
child_error:
    mov edi,126
    jmp exit

write_all:
    test rdx,rdx
    jz .Lwrite_done
.Lwrite_loop:
    mov eax,1
    syscall
    cmp rax,-4
    je .Lwrite_loop
    test rax,rax
    jle stdout_error
    add rsi,rax
    sub rdx,rax
    jnz .Lwrite_loop
.Lwrite_done:
    ret
mismatch_exit:
    mov edi,1
    lea rsi,[rip+mismatch]
    mov edx,mismatch_len
    call write_all
    mov edi,1
    jmp exit
usage_error:
    mov edi,2
    lea rsi,[rip+usage]
    mov edx,usage_len
    call write_all
    mov edi,2
    jmp exit
calc_error:
    mov edi,2
    lea rsi,[rip+failed]
    mov edx,failed_len
    call write_all
    mov edi,3
    jmp exit
stdout_error:
    mov edi,4
exit:
    mov eax,60
    syscall
.section .note.GNU-stack,"",@progbits
