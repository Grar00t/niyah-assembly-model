.file "tile_probe.s"
.intel_syntax noprefix
.section .rodata
s_features: .asciz "features"
s_scalar: .asciz "scalar"
s_auto: .asciz "auto"
s_avx2: .asciz "avx2"
s_avx512: .asciz "avx512"
s_amx: .asciz "amx-int8"
.section .bss
.balign 64
input: .skip 2048
output: .skip 1024
extra: .skip 1
.section .text
.global _start
.extern nx_cpu_record,nx_detect,nx_gemm_scalar,nx_gemm_avx2,nx_gemm_avx512,nx_amx_enable,nx_gemm_amx_i8
# Exact string equality, EAX=1 equal. Other inputs are rejected.
nx_streq:
 xor eax,eax
.Leq_loop:
 mov cl,[rdi+rax]
 cmp cl,[rsi+rax]
 jne .Lneq
 inc eax
 test cl,cl
 jnz .Leq_loop
 mov eax,1
 ret
.Lneq:
 xor eax,eax
 ret
exit_code:
 mov eax,60
 syscall
badarg:
 mov edi,64
 jmp exit_code
badinput:
 mov edi,65
 jmp exit_code
badoutput:
 mov edi,74
 jmp exit_code
unsupported:
 mov edi,77
 jmp exit_code
write_all:
 test rdx,rdx
 jz .Lwrite_done
.Lwrite:
 mov eax,1
 mov edi,1
 syscall
 cmp rax,-4
 je .Lwrite
 test rax,rax
 jle badoutput
 add rsi,rax
 sub rdx,rax
 jnz .Lwrite
.Lwrite_done:
 ret
_start:
 mov r14,rsp
 cmp qword ptr [r14],2
 jb badarg
 cmp qword ptr [r14],3
 ja badarg
 call nx_detect
 mov rdi,[r14+16]
 lea rsi,[rip+s_features]
 call nx_streq
 test eax,eax
 jnz .Lfeatures
 mov r12d,0
 lea rsi,[rip+s_scalar]
 call nx_streq
 test eax,eax
 jnz .Lparse_repeat
 lea rsi,[rip+s_auto]
 call nx_streq
 test eax,eax
 jnz .Lauto
 mov r12d,1
 lea rsi,[rip+s_avx2]
 call nx_streq
 test eax,eax
 jnz .Lneed_avx2
 mov r12d,2
 lea rsi,[rip+s_avx512]
 call nx_streq
 test eax,eax
 jnz .Lneed_avx512
 mov r12d,3
 lea rsi,[rip+s_amx]
 call nx_streq
 test eax,eax
 jz badarg
 call nx_amx_enable
 test eax,eax
 jnz unsupported
 jmp .Lparse_repeat
.Lauto:
 test qword ptr [rip+nx_cpu_record+8],2
 jnz .Lauto512
 test qword ptr [rip+nx_cpu_record+8],1
 jz .Lparse_repeat
 mov r12d,1
 jmp .Lparse_repeat
.Lauto512:
 mov r12d,2
 jmp .Lparse_repeat
.Lneed_avx2:
 test qword ptr [rip+nx_cpu_record+8],1
 jz unsupported
 jmp .Lparse_repeat
.Lneed_avx512:
 test qword ptr [rip+nx_cpu_record+8],2
 jz unsupported
.Lparse_repeat:
 mov r13d,1
 cmp qword ptr [r14],3
 jne .Lread_start
 mov rsi,[r14+24]
 xor eax,eax
 xor ecx,ecx
.Ldecimal:
 movzx edx,byte ptr [rsi]
 test edx,edx
 jz .Ldecimal_end
 sub edx,48
 cmp edx,9
 ja badarg
 imul eax,eax,10
 add eax,edx
 inc ecx
 cmp ecx,8
 ja badarg
 cmp eax,10000000
 ja badarg
 inc rsi
 jmp .Ldecimal
.Ldecimal_end:
 test eax,eax
 jz badarg
 mov r13d,eax
.Lread_start:
 lea rsi,[rip+input]
 mov edx,2048
.Lread_loop:
 xor eax,eax
 xor edi,edi
 syscall
 cmp rax,-4
 je .Lread_loop
 test rax,rax
 jle badinput
 add rsi,rax
 sub rdx,rax
 jnz .Lread_loop
.Lread_extra:
 xor eax,eax
 xor edi,edi
 lea rsi,[rip+extra]
 mov edx,1
 syscall
 cmp rax,-4
 je .Lread_extra
 test rax,rax
 jnz badinput
 cmp r12d,3
 je .Lrun
 lea rsi,[rip+input]
 xor ecx,ecx
.Lfinite:
 mov eax,[rsi+rcx]
 and eax,0x7f800000
 cmp eax,0x7f800000
 je badinput
 add ecx,4
 cmp ecx,2048
 jb .Lfinite
.Lrun:
 lea rdi,[rip+input]
 lea rsi,[rip+input+1024]
 lea rdx,[rip+output]
 cmp r12d,3
 je .Lrun_amx
 cmp r12d,2
 je .Lrun512
 cmp r12d,1
 je .Lrun256
 call nx_gemm_scalar
 jmp .Lnext
.Lrun256:
 call nx_gemm_avx2
 jmp .Lnext
.Lrun512:
 call nx_gemm_avx512
 jmp .Lnext
.Lrun_amx:
 call nx_gemm_amx_i8
.Lnext:
 dec r13d
 jnz .Lrun
 lea rsi,[rip+output]
 mov edx,1024
 call write_all
 xor edi,edi
 jmp exit_code
.Lfeatures:
 cmp qword ptr [r14],2
 jne badarg
 lea rsi,[rip+nx_cpu_record]
 mov edx,64
 call write_all
 xor edi,edi
 jmp exit_code
.section .note.GNU-stack,"",@progbits
