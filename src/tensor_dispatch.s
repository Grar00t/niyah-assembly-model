.file "tensor_dispatch.s"
.intel_syntax noprefix
.section .bss
.balign 64
.global nx_cpu
nx_cpu: .skip 64
# Separate six-u32 snapshot; NXISA001 remains exactly 64 bytes.
# max basic leaf, max palette, palette1 EAX/EBX/ECX, TMUL EBX.
.global nx_amx_limits
nx_amx_limits: .skip 24
nx_supported_xcomp: .skip 8
nx_permitted_xcomp: .skip 8
.section .text
.global nx_detect_int8,nx_amx_permission,nx_policy,nx_decode_caps,nx_decode_amx_limits
# Pure policy: EDI usable ISA bits (1 AVX2,2 AVX512F,4 AMX-INT8),
# ESI process AMX permission (exactly 1). No privileged/ISA-dependent instruction.
# Return 0 scalar,1 AVX2,2 AVX512,3 AMX.
nx_policy:
 test edi,4
 jz .policy_vector
 cmp esi,1
 jne .policy_vector
 mov eax,3
 ret
.policy_vector:
 test edi,2
 jz .policy_avx2
 mov eax,2
 ret
.policy_avx2:
 xor eax,eax
 test edi,1
 setnz al
 ret
# All OS state checks precede execution of an extended instruction.
nx_detect_int8:
 push rbx
 lea rdi,[rip+nx_cpu]
 xor eax,eax
 mov ecx,11
 rep stosq
 movabs rax,0x313030415349584e
 mov [rip+nx_cpu],rax
 xor eax,eax
 cpuid
 mov r8d,eax
 mov [rip+nx_amx_limits],eax
 cmp eax,1
 jb .detect_done
 mov eax,1
 cpuid
 mov [rip+nx_cpu+24],ecx
 mov r9d,ecx
 mov eax,ecx
 and eax,0x0c000000
 cmp eax,0x0c000000
 jne .detect_leaf7
 xor ecx,ecx
 xgetbv
 shl rdx,32
 or rax,rdx
 mov [rip+nx_cpu+16],rax
.detect_leaf7:
 cmp r8d,7
 jb .detect_done
 mov eax,7
 xor ecx,ecx
 cpuid
 mov [rip+nx_cpu+32],ebx
 mov [rip+nx_cpu+40],edx
 # Palette/TMUL leaves are valid only for an advertised AMX candidate.
 mov eax,edx
 and eax,0x03000000
 cmp eax,0x03000000
 jne .detect_done
 cmp r8d,0x1e
 jb .detect_done
 mov eax,0x1d
 xor ecx,ecx
 cpuid
 mov [rip+nx_amx_limits+4],eax
 cmp eax,1
 jb .detect_done
 mov eax,0x1d
 mov ecx,1
 cpuid
 mov [rip+nx_amx_limits+8],eax
 mov [rip+nx_amx_limits+12],ebx
 mov [rip+nx_amx_limits+16],ecx
 mov eax,0x1e
 xor ecx,ecx
 cpuid
 mov [rip+nx_amx_limits+20],ebx
.detect_done:
 call nx_decode_caps
 pop rbx
 ret
# Decode the captured CPUID/XCR0 snapshot. No ISA beyond baseline x86-64.
# Exported separately so missing-state combinations can be tested without
# executing unsupported instructions. No synthetic path reaches a kernel.
nx_decode_caps:
 mov qword ptr [rip+nx_cpu+8],0
 mov r9d,[rip+nx_cpu+24]
 mov ecx,[rip+nx_cpu+32]
 mov r10d,[rip+nx_cpu+40]
 mov eax,r9d
 and eax,0x1c000000
 cmp eax,0x1c000000
 jne .decode_amx
 mov rax,[rip+nx_cpu+16]
 and eax,6
 cmp eax,6
 jne .decode_amx
 bt ecx,5
 jnc .decode_512
 or qword ptr [rip+nx_cpu+8],1
.decode_512:
 bt ecx,16
 jnc .decode_amx
 mov rax,[rip+nx_cpu+16]
 and eax,0xe6
 cmp eax,0xe6
 jne .decode_amx
 or qword ptr [rip+nx_cpu+8],2
.decode_amx:
 mov eax,r9d
 and eax,0x0c000000
 cmp eax,0x0c000000
 jne .decode_done
 and r10d,0x03000000
 cmp r10d,0x03000000
 jne .decode_done
 mov rax,[rip+nx_cpu+16]
 and eax,0x60000
 cmp eax,0x60000
 jne .decode_done
 lea rdi,[rip+nx_amx_limits]
 sub rsp,8
 call nx_decode_amx_limits
 add rsp,8
 test eax,eax
 jnz .decode_done
 or qword ptr [rip+nx_cpu+8],4
.decode_done:
 mov eax,[rip+nx_cpu+8]
 ret
# Pure palette/TMUL decoder. RDI points to the bounded six-u32 snapshot.
# Return 0 if palette 1 supports eight 16x64-byte tiles and the INT8 TMUL;
# otherwise 77. No CPUID, syscall, vector, or tile instruction executes here.
# Fields follow Intel CPUID 1DH.01H and 1EH.00H, Table 1-3.
nx_decode_amx_limits:
 cmp dword ptr [rdi],0x1e
 jb .limits_no
 cmp dword ptr [rdi+4],1
 jb .limits_no
 mov edx,[rdi+8]
 movzx eax,dx
 cmp eax,8192
 jb .limits_no
 shr edx,16
 cmp edx,1024
 jb .limits_no
 mov edx,[rdi+12]
 movzx eax,dx
 cmp eax,64
 jb .limits_no
 shr edx,16
 cmp edx,8
 jb .limits_no
 movzx eax,word ptr [rdi+16]
 cmp eax,16
 jb .limits_no
 mov edx,[rdi+20]
 movzx eax,dl
 cmp eax,16
 jb .limits_no
 shr edx,8
 and edx,0xffff
 cmp edx,64
 jb .limits_no
 xor eax,eax
 ret
.limits_no:
 mov eax,77
 ret
# Call in the final process (permission is cleared by exec).
# No tile instruction is executed by this function.
nx_amx_permission:
 test qword ptr [rip+nx_cpu+8],4
 jz .permission_no
 mov eax,158
 mov edi,0x1021
 lea rsi,[rip+nx_supported_xcomp]
 syscall
 test rax,rax
 js .permission_no
 mov eax,[rip+nx_supported_xcomp]
 and eax,0x60000
 cmp eax,0x60000
 jne .permission_no
 mov eax,158
 mov edi,0x1023
 mov esi,18
 syscall
 test rax,rax
 js .permission_no
 mov eax,158
 mov edi,0x1022
 lea rsi,[rip+nx_permitted_xcomp]
 syscall
 test rax,rax
 js .permission_no
 mov eax,[rip+nx_permitted_xcomp]
 and eax,0x60000
 cmp eax,0x60000
 jne .permission_no
 mov qword ptr [rip+nx_cpu+56],1
 xor eax,eax
 ret
.permission_no:
 mov eax,77
 ret
.section .note.GNU-stack,"",@progbits
