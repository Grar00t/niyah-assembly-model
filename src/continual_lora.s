# Bounded rank-2 continual adaptation of the native model's frozen WQ.
# Linux x86-64, GNU as, raw syscalls only. See docs/CONTINUAL_LORA.md.
.intel_syntax noprefix
.file "continual_lora.s"
.equ BASE_BYTES,38464
.equ WQ_OFFSET,35392
.equ PARAM_OFFSET,64+BASE_BYTES
.equ CHECKSUM_OFFSET,PARAM_OFFSET+256
.equ CHECKPOINT_BYTES,CHECKSUM_OFFSET+8
.equ MAX_RECORDS,16
.section .rodata
cmd_init: .asciz "init"
cmd_predict: .asciz "predict"
cmd_update: .asciz "update"
.ifdef LORA_TEST_STEP
cmd_gradient: .asciz "gradient"
.endif
lr: .float 0.03125
one: .float 1.0
minus_one: .float -1.0
half: .float 0.5
epsilon: .float 0.0000001
retention_budget: .float 0.005
default_mxcsr: .long 0x1f80
msg_invalid: .ascii "invalid command, format, base identity or numeric range\n"
.equ len_invalid,.-msg_invalid
msg_io: .ascii "input/output failure; existing adapter preserved\n"
.equ len_io,.-msg_io
msg_reject: .ascii "candidate rejected; existing adapter preserved\n"
.equ len_reject,.-msg_reject
.section .bss
.balign 16
base: .skip BASE_BYTES+1
checkpoint: .skip CHECKPOINT_BYTES+1
params: .skip 256
backup: .skip 256
grads: .skip 256
prediction: .skip 64
hidden: .skip 8
error: .skip 64
hidden_error: .skip 8
input: .skip 129
new_data: .skip MAX_RECORDS*128+1
replay_data: .skip MAX_RECORDS*128+1
stats: .skip 16
args: .skip 8
mode: .skip 4
new_count: .skip 4
replay_count: .skip 4
epochs: .skip 4
updates: .skip 8
sgd_steps: .skip 8
output_path: .skip 8
output_fd: .skip 8
.section .text
.global _start

# rdi, rsi are NUL strings; eax=1 iff equal.
equal_string:
    xor ecx,ecx
.Les_loop:
    mov al,[rdi+rcx]
    cmp al,[rsi+rcx]
    jne .Les_no
    test al,al
    jz .Les_yes
    inc ecx
    jmp .Les_loop
.Les_no:
    xor eax,eax
    ret
.Les_yes:
    mov eax,1
    ret

# Reads a snapshot, at most rdx bytes; caller uses a one-byte overflow margin.
# rdi=path, rsi=destination, rdx=capacity, rax=bytes. No symlink following.
read_file:
    push r12
    push r13
    push r14
    push r15
    mov r13,rsi
    mov r14,rdx
    mov eax,2
    mov esi,0x20000
    xor edx,edx
    syscall
    test rax,rax
    js io_error
    mov r12,rax
    xor r15d,r15d
.Lread_loop:
    mov rdi,r12
    lea rsi,[r13+r15]
    mov rdx,r14
    sub rdx,r15
    jz .Lread_done
    xor eax,eax
    syscall
    cmp rax,-4
    je .Lread_loop
    test rax,rax
    js io_error
    jz .Lread_done
    add r15,rax
    jmp .Lread_loop
.Lread_done:
    mov eax,3
    mov rdi,r12
    syscall
    test rax,rax
    js io_error
    mov rax,r15
    pop r15
    pop r14
    pop r13
    pop r12
    ret

# rdi=fd, rsi=buffer, rdx=bytes. Returns zero or a negative errno.
write_all:
    test rdx,rdx
    jz .Lwrite_ok
.Lwrite_loop:
    mov eax,1
    syscall
    cmp rax,-4
    je .Lwrite_loop
    test rax,rax
    jle .Lwrite_error
    add rsi,rax
    sub rdx,rax
    jnz .Lwrite_loop
.Lwrite_ok:
    xor eax,eax
    ret
.Lwrite_error:
    test rax,rax
    jnz .Lwrite_return
    mov rax,-5
.Lwrite_return:
    ret

# Reject nonfinite values and absolute magnitude above positive float bits edx.
# Preserves rdi/ecx/edx and r9..r15.
validate_floats:
    xor r8d,r8d
.Lvf_loop:
    mov eax,[rdi+r8*4]
    and eax,0x7fffffff
    cmp eax,edx
    ja invalid
    inc r8d
    cmp r8d,ecx
    jb .Lvf_loop
    ret

# Noncryptographic FNV-1a corruption check, never an authenticity claim.
# Full base identity is separately compared byte for byte.
fnv64:
    movabs rax,0xcbf29ce484222325
    movabs rdx,0x100000001b3
    xor ecx,ecx
.Lfnv_loop:
    movzx edi,byte ptr [rsi+rcx]
    xor rax,rdi
    imul rax,rdx
    inc ecx
    cmp ecx,CHECKSUM_OFFSET
    jb .Lfnv_loop
    ret

load_base:
    mov rax,[rip+args]
    mov rdi,[rax+24]
    lea rsi,[rip+base]
    mov edx,BASE_BYTES+1
    call read_file
    cmp rax,BASE_BYTES
    jne invalid
    lea rdi,[rip+base]
    mov ecx,BASE_BYTES/4
    mov edx,0x42800000
    call validate_floats
    ret

load_adapter:
    mov rax,[rip+args]
    mov rdi,[rax+32]
    lea rsi,[rip+checkpoint]
    mov edx,CHECKPOINT_BYTES+1
    call read_file
    cmp rax,CHECKPOINT_BYTES
    jne invalid
    lea rsi,[rip+checkpoint]
    movabs rax,0x31303041524f4c4e
    cmp [rsi],rax
    jne invalid
    cmp dword ptr [rsi+8],1
    jne invalid
    cmp dword ptr [rsi+12],16
    jne invalid
    cmp dword ptr [rsi+16],2
    jne invalid
    cmp dword ptr [rsi+20],BASE_BYTES
    jne invalid
    cmp dword ptr [rsi+24],64
    jne invalid
    cmp dword ptr [rsi+28],0
    jne invalid
    cmp qword ptr [rsi+48],0
    jne invalid
    cmp qword ptr [rsi+56],0
    jne invalid
    mov rax,[rsi+32]
    mov [rip+updates],rax
    mov rax,[rsi+40]
    mov [rip+sgd_steps],rax
    call fnv64
    cmp rax,[rip+checkpoint+CHECKSUM_OFFSET]
    jne invalid
    lea rsi,[rip+checkpoint+64]
    lea rdi,[rip+base]
    mov ecx,BASE_BYTES
    repe cmpsb
    jne invalid
    lea rdi,[rip+checkpoint+PARAM_OFFSET]
    mov ecx,64
    mov edx,0x41000000
    call validate_floats
    lea rsi,[rip+checkpoint+PARAM_OFFSET]
    lea rdi,[rip+params]
    mov ecx,32
    rep movsq
    ret

# Output is always a NEW file, never an in-place replacement of an adapter.
# Full copied base permits exact binding without a crypto runtime dependency.
save_adapter:
    lea rdi,[rip+checkpoint]
    xor eax,eax
    mov ecx,8
    rep stosq
    lea rdi,[rip+checkpoint]
    movabs rax,0x31303041524f4c4e
    mov [rdi],rax
    mov dword ptr [rdi+8],1
    mov dword ptr [rdi+12],16
    mov dword ptr [rdi+16],2
    mov dword ptr [rdi+20],BASE_BYTES
    mov dword ptr [rdi+24],64
    mov rax,[rip+updates]
    mov [rdi+32],rax
    mov rax,[rip+sgd_steps]
    mov [rdi+40],rax
    lea rdi,[rip+checkpoint+64]
    lea rsi,[rip+base]
    mov ecx,BASE_BYTES/8
    rep movsq
    lea rsi,[rip+params]
    mov ecx,32
    rep movsq
    lea rsi,[rip+checkpoint]
    call fnv64
    mov [rip+checkpoint+CHECKSUM_OFFSET],rax
    mov eax,2
    mov rdi,[rip+output_path]
    mov esi,0x200c1
    mov edx,0x180
    syscall
    test rax,rax
    js io_error
    mov [rip+output_fd],rax
    mov rdi,rax
    lea rsi,[rip+checkpoint]
    mov edx,CHECKPOINT_BYTES
    call write_all
    test rax,rax
    js .Lsave_cleanup
    mov eax,74
    mov rdi,[rip+output_fd]
    syscall
    test rax,rax
    js .Lsave_cleanup
    mov eax,3
    mov rdi,[rip+output_fd]
    syscall
    test rax,rax
    js .Lsave_unlink
    ret
.Lsave_cleanup:
    mov eax,3
    mov rdi,[rip+output_fd]
    syscall
.Lsave_unlink:
    mov eax,87
    mov rdi,[rip+output_path]
    syscall
    jmp io_error

# y = WQ*x + B*(A*x), A[2,16], B[16,2], alpha/r=1.
# rsi=x[16], output prediction[16]; only xmm0..xmm5 are used.
forward:
    lea r10,[rip+params]
    lea r11,[rip+hidden]
    xor r8d,r8d
.Lf_hrow:
    xorps xmm0,xmm0
    xor ecx,ecx
.Lf_hdim:
    movss xmm1,[r10+rcx*4]
    mulss xmm1,[rsi+rcx*4]
    addss xmm0,xmm1
    inc ecx
    cmp ecx,16
    jb .Lf_hdim
    movss [r11+r8*4],xmm0
    add r10,64
    inc r8d
    cmp r8d,2
    jb .Lf_hrow
    lea r10,[rip+base+WQ_OFFSET]
    lea r11,[rip+params+128]
    lea rdx,[rip+prediction]
    xor r8d,r8d
.Lf_yrow:
    xorps xmm0,xmm0
    xor ecx,ecx
.Lf_ydim:
    movss xmm1,[r10+rcx*4]
    mulss xmm1,[rsi+rcx*4]
    addss xmm0,xmm1
    inc ecx
    cmp ecx,16
    jb .Lf_ydim
    movss xmm1,[r11]
    mulss xmm1,[rip+hidden]
    addss xmm0,xmm1
    movss xmm1,[r11+4]
    mulss xmm1,[rip+hidden+4]
    addss xmm0,xmm1
    movss [rdx+r8*4],xmm0
    add r10,64
    add r11,8
    inc r8d
    cmp r8d,16
    jb .Lf_yrow
    ret

# rdi=dataset, eax=record count; mean half squared error in xmm0.
loss_dataset:
    push r12
    push r13
    push r14
    mov r12,rdi
    mov r13d,eax
    xor r14d,r14d
    xorps xmm15,xmm15
.Lloss_record:
    mov rsi,r12
    call forward
    lea r10,[rip+prediction]
    xor ecx,ecx
.Lloss_dim:
    movss xmm0,[r10+rcx*4]
    subss xmm0,[r12+rcx*4+64]
    mulss xmm0,xmm0
    addss xmm15,xmm0
    inc ecx
    cmp ecx,16
    jb .Lloss_dim
    add r12,128
    inc r14d
    cmp r14d,r13d
    jb .Lloss_record
    cvtsi2ss xmm1,r13d
    movaps xmm0,xmm15
    mulss xmm0,[rip+half]
    divss xmm0,xmm1
    pop r14
    pop r13
    pop r12
    ret

# SGD for one record, using OLD A and B for BOTH derivatives.
# Componentwise clip gradient to [-1,1], lr=1/32. Any bad candidate rejects.
sgd_record:
    push r12
    mov r12,rsi
    call forward
    lea r10,[rip+prediction]
    lea r11,[rip+error]
    xor ecx,ecx
.Lsgd_err:
    movss xmm0,[r10+rcx*4]
    subss xmm0,[r12+rcx*4+64]
    movss [r11+rcx*4],xmm0
    inc ecx
    cmp ecx,16
    jb .Lsgd_err
    # hidden_error = B^T * error, using the original B.
    xor r8d,r8d
    lea rdx,[rip+hidden_error]
.Lsgd_hr:
    lea r10,[rip+params+128]
    lea r10,[r10+r8*4]
    xorps xmm0,xmm0
    xor ecx,ecx
.Lsgd_hdim:
    movss xmm1,[r10]
    mulss xmm1,[r11+rcx*4]
    addss xmm0,xmm1
    add r10,8
    inc ecx
    cmp ecx,16
    jb .Lsgd_hdim
    movss [rdx+r8*4],xmm0
    inc r8d
    cmp r8d,2
    jb .Lsgd_hr
    lea r10,[rip+grads]
    xor r8d,r8d
.Lsgd_ar:
    movss xmm2,[rdx+r8*4]
    xor ecx,ecx
.Lsgd_adim:
    movss xmm0,[r12+rcx*4]
    mulss xmm0,xmm2
    movss [r10+rcx*4],xmm0
    inc ecx
    cmp ecx,16
    jb .Lsgd_adim
    add r10,64
    inc r8d
    cmp r8d,2
    jb .Lsgd_ar
    # grad B = error outer hidden.
    xor ecx,ecx
.Lsgd_bdim:
    movss xmm0,[r11+rcx*4]
    movaps xmm1,xmm0
    mulss xmm0,[rip+hidden]
    mulss xmm1,[rip+hidden+4]
    movss [r10],xmm0
    movss [r10+4],xmm1
    add r10,8
    inc ecx
    cmp ecx,16
    jb .Lsgd_bdim
    lea r10,[rip+grads]
    lea r11,[rip+params]
    xor ecx,ecx
.Lsgd_update:
    mov eax,[r10+rcx*4]
    and eax,0x7fffffff
    cmp eax,0x7f800000
    jae reject_candidate
    movss xmm0,[r10+rcx*4]
    maxss xmm0,[rip+minus_one]
    minss xmm0,[rip+one]
    movss [r10+rcx*4],xmm0
    mulss xmm0,[rip+lr]
    movss xmm1,[r11+rcx*4]
    subss xmm1,xmm0
    movd eax,xmm1
    and eax,0x7fffffff
    cmp eax,0x41000000
    ja reject_candidate
    movss [r11+rcx*4],xmm1
    inc ecx
    cmp ecx,64
    jb .Lsgd_update
    pop r12
    ret

# Validate 1..16 records of 16 features bounded by4 +16 targets bounded by64.
# rsi=dataset, rax=bytes. Returns eax=record count.
validate_dataset:
    test eax,eax
    jz invalid
    cmp eax,MAX_RECORDS*128
    ja invalid
    test eax,127
    jnz invalid
    mov r10d,eax
    shr r10d,7
    mov r9,rsi
    xor r11d,r11d
.Lvd_record:
    mov rdi,r9
    mov ecx,16
    mov edx,0x40800000
    call validate_floats
    lea rdi,[r9+64]
    mov edx,0x42800000
    call validate_floats
    add r9,128
    inc r11d
    cmp r11d,r10d
    jb .Lvd_record
    mov eax,r10d
    ret

_start:
    cld
    ldmxcsr [rip+default_mxcsr]
    mov [rip+args],rsp
    cmp qword ptr [rsp],4
    jb invalid
    mov rdi,[rsp+16]
    lea rsi,[rip+cmd_init]
    call equal_string
    test eax,eax
    jnz .Linit_cmd
    mov rdi,[rsp+16]
    lea rsi,[rip+cmd_predict]
    call equal_string
    test eax,eax
    jnz .Lpredict_cmd
    mov rdi,[rsp+16]
    lea rsi,[rip+cmd_update]
    call equal_string
    test eax,eax
    jnz .Lupdate_cmd
    .ifdef LORA_TEST_STEP
    mov rdi,[rsp+16]
    lea rsi,[rip+cmd_gradient]
    call equal_string
    test eax,eax
    jnz .Lgradient_cmd
    .endif
    jmp invalid

.Linit_cmd:
    cmp qword ptr [rsp],4
    jne invalid
    call load_base
    # Nonzero orthogonal-ish rows in A, zero B: exactly unchanged base output.
    lea rdi,[rip+params]
    xor ecx,ecx
.Linit_a:
    mov eax,0x3d800000
    test ecx,1
    jz .Linit_a_positive
    or eax,0x80000000
.Linit_a_positive:
    mov [rdi+rcx*4],eax
    mov eax,0x3d800000
    test ecx,2
    jz .Linit_b_positive
    or eax,0x80000000
.Linit_b_positive:
    mov [rdi+rcx*4+64],eax
    inc ecx
    cmp ecx,16
    jb .Linit_a
    mov rax,[rip+args]
    mov rax,[rax+32]
    mov [rip+output_path],rax
    call save_adapter
    jmp success

.Lpredict_cmd:
    cmp qword ptr [rsp],5
    jne invalid
    call load_base
    call load_adapter
    mov rax,[rip+args]
    mov rdi,[rax+40]
    lea rsi,[rip+input]
    mov edx,65
    call read_file
    cmp eax,64
    jne invalid
    lea rdi,[rip+input]
    mov ecx,16
    mov edx,0x40800000
    call validate_floats
    lea rsi,[rip+input]
    call forward
    mov edi,1
    lea rsi,[rip+prediction]
    mov edx,64
    call write_all
    test rax,rax
    js io_error
    jmp success

.Lupdate_cmd:
    cmp qword ptr [rsp],8
    jne invalid
    call load_base
    call load_adapter
    lea rsi,[rip+params]
    lea rdi,[rip+backup]
    mov ecx,32
    rep movsq
    mov rax,[rip+args]
    mov rdi,[rax+40]
    lea rsi,[rip+new_data]
    mov edx,MAX_RECORDS*128+1
    call read_file
    lea rsi,[rip+new_data]
    call validate_dataset
    mov [rip+new_count],eax
    mov rax,[rip+args]
    mov rdi,[rax+48]
    lea rsi,[rip+replay_data]
    mov edx,MAX_RECORDS*128+1
    call read_file
    lea rsi,[rip+replay_data]
    call validate_dataset
    mov [rip+replay_count],eax
    mov rax,[rip+args]
    mov rsi,[rax+64]
    xor eax,eax
    xor ecx,ecx
.Lparse_epochs:
    movzx edx,byte ptr [rsi+rcx]
    test edx,edx
    jz .Lparsed_epochs
    sub edx,'0'
    cmp edx,9
    ja invalid
    imul eax,10
    add eax,edx
    cmp eax,256
    ja invalid
    inc ecx
    cmp ecx,3
    ja invalid
    jmp .Lparse_epochs
.Lparsed_epochs:
    test eax,eax
    jz invalid
    mov [rip+epochs],eax
    lea rdi,[rip+new_data]
    mov eax,[rip+new_count]
    call loss_dataset
    movss [rip+stats],xmm0
    lea rdi,[rip+replay_data]
    mov eax,[rip+replay_count]
    call loss_dataset
    movss [rip+stats+8],xmm0
    xor r13d,r13d
.Ltrain_epoch:
    lea r14,[rip+new_data]
    xor r15d,r15d
.Ltrain_new:
    mov rsi,r14
    call sgd_record
    add r14,128
    inc r15d
    cmp r15d,[rip+new_count]
    jb .Ltrain_new
    lea r14,[rip+replay_data]
    xor r15d,r15d
.Ltrain_replay:
    mov rsi,r14
    call sgd_record
    add r14,128
    inc r15d
    cmp r15d,[rip+replay_count]
    jb .Ltrain_replay
    inc r13d
    cmp r13d,[rip+epochs]
    jb .Ltrain_epoch
    lea rdi,[rip+new_data]
    mov eax,[rip+new_count]
    call loss_dataset
    movss [rip+stats+4],xmm0
    lea rdi,[rip+replay_data]
    mov eax,[rip+replay_count]
    call loss_dataset
    movss [rip+stats+12],xmm0
    movss xmm0,[rip+stats+12]
    movss xmm1,[rip+stats+8]
    addss xmm1,[rip+retention_budget]
    ucomiss xmm0,xmm1
    jp reject_candidate
    ja reject_candidate
    movss xmm0,[rip+stats+4]
    addss xmm0,[rip+epsilon]
    ucomiss xmm0,[rip+stats]
    jp reject_candidate
    jae reject_candidate
    add qword ptr [rip+updates],1
    jc reject_candidate
    mov eax,[rip+new_count]
    add eax,[rip+replay_count]
    imul eax,[rip+epochs]
    add [rip+sgd_steps],rax
    jc reject_candidate
    mov rax,[rip+args]
    mov rax,[rax+56]
    mov [rip+output_path],rax
    call save_adapter
    mov edi,1
    lea rsi,[rip+stats]
    mov edx,16
    call write_all
    test rax,rax
    js io_error
    jmp success

.ifdef LORA_TEST_STEP
# Testing build only: one native gradient step, no persistent write/promotion.
.Lgradient_cmd:
    cmp qword ptr [rsp],5
    jne invalid
    call load_base
    call load_adapter
    mov rax,[rip+args]
    mov rdi,[rax+40]
    lea rsi,[rip+input]
    mov edx,129
    call read_file
    cmp eax,128
    jne invalid
    lea rsi,[rip+input]
    call validate_dataset
    lea rsi,[rip+params]
    lea rdi,[rip+backup]
    mov ecx,32
    rep movsq
    lea rsi,[rip+input]
    call sgd_record
    mov edi,1
    lea rsi,[rip+grads]
    mov edx,256
    call write_all
    test rax,rax
    js io_error
    mov edi,1
    lea rsi,[rip+params]
    mov edx,256
    call write_all
    test rax,rax
    js io_error
    jmp success
.endif

reject_candidate:
    lea rsi,[rip+backup]
    lea rdi,[rip+params]
    mov ecx,32
    rep movsq
    mov edi,2
    lea rsi,[rip+msg_reject]
    mov edx,len_reject
    call write_all
    mov edi,10
    jmp exit
invalid:
    mov edi,2
    lea rsi,[rip+msg_invalid]
    mov edx,len_invalid
    call write_all
    mov edi,2
    jmp exit
io_error:
    mov edi,2
    lea rsi,[rip+msg_io]
    mov edx,len_io
    call write_all
    mov edi,3
    jmp exit
success:
    xor edi,edi
exit:
    mov eax,60
    syscall
.section .note.GNU-stack,"",@progbits
