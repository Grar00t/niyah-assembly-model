.file "tile_kernels.s"
.intel_syntax noprefix
# Native kernel interface: Linux x86-64, no libc. CPUID before any AVX/AMX.
.section .bss
.balign 64
.global nx_cpu_record
nx_cpu_record: .skip 64
nx_amx_packed: .skip 1024
nx_xcomp: .skip 8
.section .rodata
.balign 64
nx_tilecfg:
.byte 1,0
.zero 14
.word 64,64,64
.zero 26
.byte 16,16,16
.zero 13
.section .text
.global nx_detect,nx_gemm_scalar,nx_gemm_avx2,nx_gemm_avx512,nx_amx_enable,nx_gemm_amx_i8
# Returns bit0 AVX2+FMA+OS, bit1 AVX512F+FMA+OS, bit2 AMX-TILE+INT8+XCR0.
# bit2 does NOT mean Linux process permission has been granted.
nx_detect:
 push rbx
 mov rax,0x314449555043594e
 mov [rip+nx_cpu_record],rax
 xor eax,eax
 cpuid
 mov r8d,eax
 mov [rip+nx_cpu_record+48],rax
 cmp eax,1
 jb .Ldetect_done
 mov eax,1
 cpuid
 mov [rip+nx_cpu_record+16],ecx
 mov r9d,ecx
 bt ecx,27
 jnc .Ldetect_leaf7
 xor ecx,ecx
 xgetbv
 shl rdx,32
 or rax,rdx
 mov [rip+nx_cpu_record+40],rax
.Ldetect_leaf7:
 cmp r8d,7
 jb .Ldetect_done
 mov eax,7
 xor ecx,ecx
 cpuid
 mov [rip+nx_cpu_record+24],ebx
 mov [rip+nx_cpu_record+32],edx
 mov r10d,edx
 mov eax,r9d
 and eax,0x18001000
 cmp eax,0x18001000
 jne .Ldetect_amx
 mov rax,[rip+nx_cpu_record+40]
 and eax,6
 cmp eax,6
 jne .Ldetect_amx
 bt ebx,5
 jnc .Ldetect_avx512
 or qword ptr [rip+nx_cpu_record+8],1
.Ldetect_avx512:
 bt ebx,16
 jnc .Ldetect_amx
 mov rax,[rip+nx_cpu_record+40]
 and eax,0xe6
 cmp eax,0xe6
 jne .Ldetect_amx
 or qword ptr [rip+nx_cpu_record+8],2
.Ldetect_amx:
 and r10d,0x03000000
 cmp r10d,0x03000000
 jne .Ldetect_done
 mov rax,[rip+nx_cpu_record+40]
 and eax,0x60000
 cmp eax,0x60000
 jne .Ldetect_done
 or qword ptr [rip+nx_cpu_record+8],4
.Ldetect_done:
 mov eax,[rip+nx_cpu_record+8]
 pop rbx
 ret
# FP32 C16x16=A16x16 * B16x16, all contiguous row-major.
# Scalar accumulation is non-fused. No aliasing of A/B with C.
nx_gemm_scalar:
 xor r8d,r8d
.Lscalar_i:
 xor r9d,r9d
.Lscalar_j:
 xorps xmm0,xmm0
 lea rax,[rdi+r8]
 lea r11,[rsi+r9]
 xor r10d,r10d
.Lscalar_k:
 movss xmm1,[rax+r10]
 mulss xmm1,[r11]
 addss xmm0,xmm1
 add r11,64
 add r10,4
 cmp r10,64
 jb .Lscalar_k
 lea rax,[rdx+r8]
 movss [rax+r9],xmm0
 add r9,4
 cmp r9,64
 jb .Lscalar_j
 add r8,64
 cmp r8,1024
 jb .Lscalar_i
 ret
# Four output rows by eight columns: B vector reused across four FMA chains.
nx_gemm_avx2:
 xor r8d,r8d
.Lavx2_i:
 xor r9d,r9d
.Lavx2_j:
 vxorps ymm0,ymm0,ymm0
 vxorps ymm1,ymm1,ymm1
 vxorps ymm2,ymm2,ymm2
 vxorps ymm3,ymm3,ymm3
 lea rax,[rdi+r8]
 lea r11,[rsi+r9]
 xor r10d,r10d
.Lavx2_k:
 vmovups ymm4,[r11]
 vbroadcastss ymm5,[rax+r10]
 vfmadd231ps ymm0,ymm4,ymm5
 vbroadcastss ymm5,[rax+r10+64]
 vfmadd231ps ymm1,ymm4,ymm5
 vbroadcastss ymm5,[rax+r10+128]
 vfmadd231ps ymm2,ymm4,ymm5
 vbroadcastss ymm5,[rax+r10+192]
 vfmadd231ps ymm3,ymm4,ymm5
 add r11,64
 add r10,4
 cmp r10,64
 jb .Lavx2_k
 lea rax,[rdx+r8]
 add rax,r9
 vmovups [rax],ymm0
 vmovups [rax+64],ymm1
 vmovups [rax+128],ymm2
 vmovups [rax+192],ymm3
 add r9,32
 cmp r9,64
 jb .Lavx2_j
 add r8,256
 cmp r8,1024
 jb .Lavx2_i
 vzeroupper
 ret
# Four rows by sixteen columns; reachable only after AVX512F/XCR0 gate.
nx_gemm_avx512:
 xor r8d,r8d
.Lavx512_i:
 vpxord zmm0,zmm0,zmm0
 vpxord zmm1,zmm1,zmm1
 vpxord zmm2,zmm2,zmm2
 vpxord zmm3,zmm3,zmm3
 lea rax,[rdi+r8]
 mov r11,rsi
 xor r10d,r10d
.Lavx512_k:
 vmovups zmm4,[r11]
 vbroadcastss zmm5,[rax+r10]
 vfmadd231ps zmm0,zmm4,zmm5
 vbroadcastss zmm5,[rax+r10+64]
 vfmadd231ps zmm1,zmm4,zmm5
 vbroadcastss zmm5,[rax+r10+128]
 vfmadd231ps zmm2,zmm4,zmm5
 vbroadcastss zmm5,[rax+r10+192]
 vfmadd231ps zmm3,zmm4,zmm5
 add r11,64
 add r10,4
 cmp r10,64
 jb .Lavx512_k
 lea rax,[rdx+r8]
 vmovups [rax],zmm0
 vmovups [rax+64],zmm1
 vmovups [rax+128],zmm2
 vmovups [rax+192],zmm3
 add r8,256
 cmp r8,1024
 jb .Lavx512_i
 vzeroupper
 ret
# Return 0 on usable AMX INT8, 77 on unsupported/refused. Never executes tile here.
nx_amx_enable:
 test qword ptr [rip+nx_cpu_record+8],4
 jz .Lamx_unsupported
 mov eax,158
 mov edi,0x1021
 lea rsi,[rip+nx_xcomp]
 syscall
 test rax,rax
 js .Lamx_unsupported
 mov eax,[rip+nx_xcomp]
 and eax,0x60000
 cmp eax,0x60000
 jne .Lamx_unsupported
 mov eax,158
 mov edi,0x1023
 mov esi,18
 syscall
 test rax,rax
 js .Lamx_unsupported
 xor eax,eax
 ret
.Lamx_unsupported:
 mov eax,77
 ret
# Signed INT8 C16x16=A16x64 * B64x16; packs B to groups of four K lanes.
# Caller must have successful nx_amx_enable in this same process.
nx_gemm_amx_i8:
 push rbx
 lea rbx,[rip+nx_amx_packed]
 xor r8d,r8d
.Lpack_group:
 xor r9d,r9d
.Lpack_col:
 xor r10d,r10d
.Lpack_lane:
 mov eax,r8d
 shl eax,2
 add eax,r10d
 shl eax,4
 add eax,r9d
 mov cl,[rsi+rax]
 mov eax,r8d
 shl eax,6
 lea eax,[rax+r9*4]
 add eax,r10d
 mov [rbx+rax],cl
 inc r10d
 cmp r10d,4
 jb .Lpack_lane
 inc r9d
 cmp r9d,16
 jb .Lpack_col
 inc r8d
 cmp r8d,16
 jb .Lpack_group
 ldtilecfg [rip+nx_tilecfg]
 tilezero tmm0
 mov eax,64
 tileloadd tmm1,[rdi+rax*1]
 tileloadd tmm2,[rbx+rax*1]
 tdpbssd tmm0,tmm1,tmm2
 tilestored [rdx+rax*1],tmm0
 tilerelease
 pop rbx
 ret
.section .note.GNU-stack,"",@progbits
