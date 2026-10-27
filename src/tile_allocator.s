.file "tile_allocator.s"
.intel_syntax noprefix
# Bounded offline AMX tile-register interval graph coloring.
# Host ABI: Linux x86-64 raw syscalls. The engine executes no AMX instruction.
# Input 16 + N*8 bytes: "NTILE001", U8 N(1..64), U8 reserved mask,
# 6 zero bytes, then N intervals (U32 begin, U32 end), little endian.
# Starts must be nondecreasing, end > begin. Intervals are half-open.
# Output 8 + N bytes: "TMM1", U8 N, 3 zero bytes, N register IDs [0,7].
# 0 success, 64 CLI, 65 malformed input, 74 write failure,
# 77 insufficient registers. No partial output on rejection.
# Validation precedes allocation; SIGPIPE is handled as an output failure.
.section .rodata
.balign 8
sigpipe_ignore: .quad 1,0,0,0
.section .bss
.balign 16
header: .skip 16
intervals: .skip 512
extra: .skip 1
.balign 16
live_end: .skip 32
output: .skip 72
.section .text
.global _start
_start:
 cmp qword ptr [rsp],1
 jne .Lbadarg
 lea rsi,[rip+header]
 mov edx,16
 call .Lread_exact
 movabs rax,0x313030454c49544e
 cmp qword ptr [rip+header],rax
 jne .Lbadinput
 movzx r12d,byte ptr [rip+header+8]
 test r12d,r12d
 jz .Lbadinput
 cmp r12d,64
 ja .Lbadinput
 movzx r13d,byte ptr [rip+header+9]
 lea rdi,[rip+header+10]
 xor ecx,ecx
.Lreserved_loop:
 cmp byte ptr [rdi+rcx],0
 jne .Lbadinput
 inc ecx
 cmp ecx,6
 jb .Lreserved_loop
 lea rsi,[rip+intervals]
 mov edx,r12d
 shl edx,3
 call .Lread_exact
.Lcheck_extra:
 xor eax,eax
 xor edi,edi
 lea rsi,[rip+extra]
 mov edx,1
 syscall
 cmp rax,-4
 je .Lcheck_extra
 test rax,rax
 jne .Lbadinput
 # Validate the complete stream before allocating any register.
 lea rdi,[rip+intervals]
 xor r8d,r8d
 xor r9d,r9d
.Lvalidate:
 mov eax,[rdi+r8*8]
 cmp eax,[rdi+r8*8+4]
 jae .Lbadinput
 cmp eax,r9d
 jb .Lbadinput
 mov r9d,eax
 inc r8d
 cmp r8d,r12d
 jb .Lvalidate
 mov dword ptr [rip+output],0x314d4d54
 mov byte ptr [rip+output+4],r12b
 .global nx_color_begin,nx_color_end
nx_color_begin:
 lea rdi,[rip+intervals]
 lea rdx,[rip+live_end]
 lea rsi,[rip+output+8]
 xor r8d,r8d
.Linterval:
 mov r15d,[rdi+r8*8]
 mov r14d,[rdi+r8*8+4]
 xor r10d,r10d
.Ltry_color:
 bt r13d,r10d
 jc .Lnext_color
 cmp dword ptr [rdx+r10*4],r15d
 ja .Lnext_color
 mov [rdx+r10*4],r14d
 mov [rsi+r8],r10b
 jmp .Lassigned
.Lnext_color:
 inc r10d
 cmp r10d,8
 jb .Ltry_color
 jmp .Lpressure
.Lassigned:
 inc r8d
 cmp r8d,r12d
 jb .Linterval
nx_color_end:
 # rt_sigaction(SIGPIPE,SIG_IGN,NULL,8) so EPIPE returns exit 74.
 mov eax,13
 mov edi,13
 lea rsi,[rip+sigpipe_ignore]
 xor edx,edx
 mov r10d,8
 syscall
 test rax,rax
 js .Lbadoutput
 lea rsi,[rip+output]
 mov edx,r12d
 add edx,8
.Lwrite_all:
 test edx,edx
 jz .Lok
 mov eax,1
 mov edi,1
 syscall
 cmp rax,-4
 je .Lwrite_all
 test rax,rax
 jle .Lbadoutput
 add rsi,rax
 sub rdx,rax
 jmp .Lwrite_all
.Lread_exact:
 test rdx,rdx
 jz .Lread_done
.Lread_loop:
 xor eax,eax
 xor edi,edi
 syscall
 cmp rax,-4
 je .Lread_loop
 test rax,rax
 jle .Lbadinput
 add rsi,rax
 sub rdx,rax
 jnz .Lread_loop
.Lread_done:
 ret
.Lbadarg:
 mov edi,64
 jmp .Lexit
.Lbadinput:
 mov edi,65
 jmp .Lexit
.Lbadoutput:
 mov edi,74
 jmp .Lexit
.Lpressure:
 mov edi,77
 jmp .Lexit
.Lok:
 xor edi,edi
.Lexit:
 mov eax,60
 syscall
 ud2
.section .note.GNU-stack,"",@progbits
