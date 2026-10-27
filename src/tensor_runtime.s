.file "tensor_runtime.s"
.intel_syntax noprefix
.section .rodata
mode_auto: .asciz "auto"
mode_scalar: .asciz "scalar"
mode_avx2: .asciz "avx2"
mode_avx512: .asciz "avx512"
mode_amx: .asciz "amx"
mode_features: .asciz "features"
.balign 8
sigpipe_ignore: .quad 1,0,0,0
.balign 64
.global nx_tilecfg
nx_tilecfg:
.byte 1,0
.zero 14
.rept 8
.word 64
.endr
.zero 16
.rept 8
.byte 16
.endr
.zero 8
.section .bss
extra: .skip 1
requested: .skip 4
features_only: .skip 4
.section .text
.global _start,nx_pack_b,nx_i8_scalar,nx_i8_avx2,nx_i8_avx512
.extern nx_plan,nx_colors,nx_inputs,nx_slots,nx_outputs,nx_packbuf
.extern nx_vector_program,nx_amx_program,nx_cpu
.extern nx_detect_int8,nx_amx_permission,nx_policy
_start:
 cld
 mov r14,rsp
 and rsp,-16
 mov eax,13
 mov edi,13
 lea rsi,[rip+sigpipe_ignore]
 xor edx,edx
 mov r10d,8
 syscall
 test rax,rax
 js output_error
 call nx_detect_int8
 mov dword ptr [rip+requested],-1
 cmp qword ptr [r14],1
 je choose
 cmp qword ptr [r14],2
 jne usage_error
 mov rdi,[r14+16]
 lea rsi,[rip+mode_auto]
 call str_equal
 test eax,eax
 jnz choose
 lea rsi,[rip+mode_features]
 call str_equal
 test eax,eax
 jnz .features
 mov dword ptr [rip+requested],0
 lea rsi,[rip+mode_scalar]
 call str_equal
 test eax,eax
 jnz choose
 mov dword ptr [rip+requested],1
 lea rsi,[rip+mode_avx2]
 call str_equal
 test eax,eax
 jnz choose
 mov dword ptr [rip+requested],2
 lea rsi,[rip+mode_avx512]
 call str_equal
 test eax,eax
 jnz choose
 mov dword ptr [rip+requested],3
 lea rsi,[rip+mode_amx]
 call str_equal
 test eax,eax
 jz usage_error
 jmp choose
.features:
 mov dword ptr [rip+requested],-1
 mov dword ptr [rip+features_only],1
choose:
 mov eax,[rip+requested]
 cmp eax,0
 je selected
 cmp eax,1
 je .force_avx2
 cmp eax,2
 je .force_avx512
 cmp eax,3
 je .force_amx
 call nx_amx_permission
 xor esi,esi
 test eax,eax
 sete sil
 mov edi,[rip+nx_cpu+8]
 call nx_policy
 jmp selected
.force_avx2:
 test qword ptr [rip+nx_cpu+8],1
 jz unsupported
 mov eax,1
 jmp selected
.force_avx512:
 test qword ptr [rip+nx_cpu+8],2
 jz unsupported
 mov eax,2
 jmp selected
.force_amx:
 call nx_amx_permission
 test eax,eax
 jnz unsupported
 mov eax,3
selected:
 mov [rip+nx_cpu+48],rax
 cmp dword ptr [rip+features_only],0
 jne .write_features
 lea r15,[rip+nx_i8_scalar]
 cmp eax,1
 jne .not_avx2
 lea r15,[rip+nx_i8_avx2]
.not_avx2:
 cmp eax,2
 jne .read_inputs
 lea r15,[rip+nx_i8_avx512]
.read_inputs:
 lea rsi,[rip+nx_inputs]
 mov edx,[rip+nx_plan+8]
 shl edx,10
 call read_exact
.read_extra:
 xor eax,eax
 xor edi,edi
 lea rsi,[rip+extra]
 mov edx,1
 syscall
 cmp rax,-4
 je .read_extra
 test rax,rax
 jnz input_error
 cmp qword ptr [rip+nx_cpu+48],3
 jne .run_vector
 call nx_amx_program
 jmp .write_output
.run_vector:
 call nx_vector_program
.write_output:
 lea rsi,[rip+nx_outputs]
 mov edx,[rip+nx_plan+12]
 shl edx,10
 jmp write_all
.write_features:
 lea rsi,[rip+nx_cpu]
 mov edx,64
 jmp write_all
str_equal:
 xor eax,eax
.eq_loop:
 mov cl,[rdi+rax]
 cmp cl,[rsi+rax]
 jne .eq_no
 inc eax
 test cl,cl
 jnz .eq_loop
 mov eax,1
 ret
.eq_no:
 xor eax,eax
 ret
read_exact:
 test edx,edx
 jz .read_done
.read_loop:
 xor eax,eax
 xor edi,edi
 syscall
 cmp rax,-4
 je .read_loop
 test rax,rax
 jle input_error
 add rsi,rax
 sub rdx,rax
 jnz .read_loop
.read_done:
 ret
write_all:
 test edx,edx
 jz success
.write_loop:
 mov eax,1
 mov edi,1
 syscall
 cmp rax,-4
 je .write_loop
 test rax,rax
 jle output_error
 add rsi,rax
 sub rdx,rax
 jnz .write_loop
success:
 xor edi,edi
 jmp exit_code
usage_error:
 mov edi,64
 jmp exit_code
input_error:
 mov edi,65
 jmp exit_code
output_error:
 mov edi,74
 jmp exit_code
unsupported:
 mov edi,77
exit_code:
 mov eax,60
 syscall
 ud2

# C16x16 i32 = A16x64 signed i8 * B64x16 signed i8. No aliasing.
nx_i8_scalar:
 push rbx
 push r12
 xor r8d,r8d
.scalar_i:
 xor r9d,r9d
.scalar_j:
 lea r10,[rdi+r8]
 lea r11,[rsi+r9]
 xor eax,eax
 xor ecx,ecx
.scalar_k:
 movsx ebx,byte ptr [r10+rcx]
 movsx r12d,byte ptr [r11]
 imul ebx,r12d
 add eax,ebx
 add r11,16
 inc ecx
 cmp ecx,64
 jb .scalar_k
 lea r10,[rdx+r8]
 mov [r10+r9*4],eax
 inc r9d
 cmp r9d,16
 jb .scalar_j
 add r8,64
 cmp r8,1024
 jb .scalar_i
 pop r12
 pop rbx
 ret

# Eight output columns, signed widening (no saturating intermediate).
nx_i8_avx2:
 xor r8d,r8d
.avx2_i:
 xor r9d,r9d
.avx2_j:
 vpxor ymm0,ymm0,ymm0
 lea r10,[rdi+r8]
 mov eax,r9d
 shr eax,2
 lea r11,[rsi+rax]
 xor ecx,ecx
.avx2_k:
 movsx eax,byte ptr [r10+rcx]
 vmovd xmm1,eax
 vpbroadcastd ymm1,xmm1
 vpmovsxbd ymm2,QWORD PTR [r11]
 vpmulld ymm1,ymm1,ymm2
 vpaddd ymm0,ymm0,ymm1
 add r11,16
 inc ecx
 cmp ecx,64
 jb .avx2_k
 lea rax,[rdx+r8]
 vmovdqu [rax+r9],ymm0
 add r9,32
 cmp r9,64
 jb .avx2_j
 add r8,64
 cmp r8,1024
 jb .avx2_i
 vzeroupper
 ret

# AVX512F only: signed i8 -> 16 i32 lanes, exact multiplication and accumulation.
nx_i8_avx512:
 xor r8d,r8d
.avx512_i:
 vpxord zmm0,zmm0,zmm0
 lea r10,[rdi+r8]
 mov r11,rsi
 xor ecx,ecx
.avx512_k:
 movsx eax,byte ptr [r10+rcx]
 vmovd xmm1,eax
 vpbroadcastd zmm1,xmm1
 vpmovsxbd zmm2,XMMWORD PTR [r11]
 vpmulld zmm1,zmm1,zmm2
 vpaddd zmm0,zmm0,zmm1
 add r11,16
 inc ecx
 cmp ecx,64
 jb .avx512_k
 vmovdqu32 [rdx+r8],zmm0
 add r8,64
 cmp r8,1024
 jb .avx512_i
 vzeroupper
 ret

# P[g,4*j+l] = B[4*g+l,j]. RDI=B row-major, RSI=packed B.
nx_pack_b:
 xor r8d,r8d
.pack_g:
 xor r9d,r9d
.pack_j:
 xor r10d,r10d
.pack_l:
 mov eax,r8d
 shl eax,2
 add eax,r10d
 shl eax,4
 add eax,r9d
 mov cl,[rdi+rax]
 mov eax,r8d
 shl eax,6
 lea eax,[rax+r9*4]
 add eax,r10d
 mov [rsi+rax],cl
 inc r10d
 cmp r10d,4
 jb .pack_l
 inc r9d
 cmp r9d,16
 jb .pack_j
 inc r8d
 cmp r8d,16
 jb .pack_g
 ret
.section .note.GNU-stack,"",@progbits
