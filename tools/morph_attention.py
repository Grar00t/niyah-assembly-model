#!/usr/bin/env python3
"""Build/verify an isolated u16 surface-token assembly attention experiment.

Python is a build/checkpoint harness only. Training and inference execute
independent static assembly binaries with Linux syscalls and no libc.
"""
import hashlib, json, math, os, pathlib, re, shutil, struct, subprocess, sys, tempfile, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
WORK=ROOT/'build/morph_native'
BASE=ROOT/'src/morph_tokenizer/upstream/attention_byte_v2.s'
VOCAB=ROOT/'src/morph_tokenizer/vocab.json'
CODEC=ROOT/'build/morph-tokenizer'
BASE_SHA='f7e66d703c388fe8a110e95569481ea869ca4cd43a7029c69e5f040ff92a0450'
VOCAB_SHA='1939a48174247d0012e934206e31e02f2d6d795ee622647faf07c1209e010d43'
FORMAT='niyah-morph-attention-f32-v1'
V=294; D=16; CTX=8; EPOCHS=240; SAMPLE=64
WEIGHT_BYTES=V*132+4672
TRAIN=('والكتاب كتاب. سيكتبها طالب. فبالكتاب علم. سوف يكتب كتابه. '
       'طالب فهم كتابه. والكتاب علم. سيكتب كتابه. كتابه كتاب.\n').encode()
VALID=('سيكتب كتابه. والكتاب علم. طالب فهم كتابه.\n').encode()
COMMANDS=[]
def sha(data): return hashlib.sha256(data).hexdigest()
def command(argv,*,cwd=ROOT,data=None,expect=0):
    started=time.perf_counter()
    p=subprocess.run([str(x) for x in argv],cwd=cwd,input=data,capture_output=True,timeout=180)
    COMMANDS.append({'argv':[str(x) for x in argv],'cwd':str(cwd),'exit':p.returncode,
      'seconds':time.perf_counter()-started,'stdout_bytes':len(p.stdout),'stdout_sha256':sha(p.stdout),
      'stderr_bytes':len(p.stderr),'stderr_sha256':sha(p.stderr)})
    if p.returncode!=expect: raise RuntimeError(f'{argv}: exit {p.returncode}: {p.stderr[:500]!r}')
    return p.stdout
def token_ids(raw):
    encoded=command([CODEC,'encode'],data=raw)
    if len(encoded)<64 or encoded[:8]!=b'NYMORPH\0': raise ValueError('token header')
    version,vid,count,nbytes=struct.unpack_from('<IIQQ',encoded,8)
    if (version,vid,nbytes)!=(1,1,len(raw)) or encoded[32:64].hex()!=VOCAB_SHA or len(encoded)!=64+2*count:
        raise ValueError('token identity/length')
    ids=struct.unpack_from('<'+'H'*count,encoded,64)
    if any(i>=V for i in ids) or command([CODEC,'decode'],data=encoded)!=raw: raise ValueError('token roundtrip')
    return encoded[64:],ids
def replace_exact(source,old,new,count=1):
    if source.count(old)!=count: raise ValueError(f'source adaptation count: {old!r} -> {source.count(old)} expected {count}')
    return source.replace(old,new)
def assembly_source():
    raw=BASE.read_bytes()
    if sha(raw)!=BASE_SHA or sha(VOCAB.read_bytes())!=VOCAB_SHA: raise ValueError('pinned source/vocabulary changed')
    source=raw.decode(); vocab=json.loads(VOCAB.read_bytes()); pieces=[s.encode() for s in vocab['pieces']]
    if len(pieces)+256!=V: raise ValueError('vocabulary shape')
    changes=[
      ('.equ V, 256','.equ V, 294',1),
      ('train_bytes: .incbin "data/corpus.txt"','train_bytes: .incbin "train.ids"',1),
      ('.equ N, . - train_bytes','.equ N, (. - train_bytes)/2',1),
      ('valid_bytes: .incbin "data/heldout.txt"','valid_bytes: .incbin "valid.ids"',1),
      ('.equ NV, . - valid_bytes','.equ NV, (. - valid_bytes)/2',1),
      ('.incbin "models/attention_weights.f32"','.incbin "weights.f32"',1),
      ('!= 38464','!= (V*132+4672)',1),
      ('.equ POS, params_start+16384','.equ POS, params_start+V*64',1),
      ('.equ F, params_start+16896','.equ F, params_start+V*64+512',1),
      ('.equ FB, params_start+17920','.equ FB, params_start+V*64+1536',1),
      ('.equ O, params_start+17984','.equ O, params_start+V*64+1600',1),
      ('.equ OB, params_start+34368','.equ OB, params_start+V*128+1600',1),
      ('.equ WQ, params_start+35392','.equ WQ, params_start+V*132+1600',1),
      ('.equ WK, params_start+36416','.equ WK, params_start+V*132+2624',1),
      ('.equ WV, params_start+37440','.equ WV, params_start+V*132+3648',1),
      ('E:      .skip 16384','E:      .skip V*64',1),
      ('O:      .skip 16384','O:      .skip V*64',1),
      ('OB:     .skip 1024','OB:     .skip V*4',1),
      ('L:      .skip 1024','L:      .skip V*4',1),
      ('PROB:   .skip 1024','PROB:   .skip V*4',1),
      ('GL:     .skip 1024','GL:     .skip V*4',1),
      ('TOK:    .skip 8','TOK:    .skip CTX*2',1),
      ('MASK:   .skip 256','MASK:   .skip V',1),
      ('genbuf: .skip SAMPLE_LEN+CTX+1','genbuf: .skip (SAMPLE_LEN+CTX)*2',1),
      ('movzx eax,byte ptr [rbx+rax]','movzx eax,word ptr [rbx+rax*2]',1),
      ('mov byte ptr [rdi+r8],al','mov word ptr [rdi+r8*2],ax',1),
      ('movzx eax,byte ptr [rbx+r13]','movzx eax,word ptr [rbx+r13*2]',4),
      ('movzx eax,byte ptr [r10+r8]','movzx eax,word ptr [r10+r8*2]',1),
      ('movzx eax,byte ptr [rbx+rcx]','movzx eax,word ptr [rbx+rcx*2]',1),
      ('cmp ebx,256','cmp ebx,V',1),('mov ebx,255','mov ebx,V-1',1),
      ('# vocabulary includes exactly the bytes in training corpus','# vocabulary includes exactly the surface token IDs in training corpus',1),
    ]
    for old,new,count in changes: source=replace_exact(source,old,new,count)
    old="""    mov ecx,CTX
    rep movsb
    lea rsi,[rip+genbuf]
    mov edx,CTX
    call write_all
"""
    new="""    mov ecx,CTX
    rep movsw
    xor r12d,r12d
.Lmorph_print_context:
    lea rax,[rip+genbuf]
    movzx edi,word ptr [rax+r12*2]
    call morph_append_token
    inc r12d
    cmp r12d,CTX
    jb .Lmorph_print_context
"""
    source=replace_exact(source,old,new)
    old="""    mov byte ptr [r10+r13],bl
    lea rsi,[r10+r13]
    mov edx,1
    call write_all
"""
    new="""    mov word ptr [r10+r13*2],bx
    mov edi,ebx
    call morph_append_token
"""
    source=replace_exact(source,old,new)
    source=replace_exact(source,'_start:\n    cld','_start:\n    cld\n    .ifdef MORPH_UTF8_SELFTEST\n    jmp morph_utf8_selftest\n    .endif\n    SAY morph_identity,88')
    source=replace_exact(source,'    SAY s_gen,l_gen\n','')
    source='.file \"niyah_morph_attention_v1.s\"\n# GENERATED VERSIONED U16 SURFACE-TOKEN VARIANT; base SHA256 '+BASE_SHA+'\n'+source
    source+='\n.section .rodata\nmorph_identity: .ascii "tokenizer_vocab_sha256 '+VOCAB_SHA+'\\n"\n.equ morph_identity_len,.-morph_identity\n'
    source+='morph_byte_values: .byte '+','.join(str(i) for i in range(256))+'\n'
    for i,piece in enumerate(pieces): source+='morph_surface_'+str(i)+': .byte '+','.join(str(x) for x in piece)+'\n'
    source+='.balign 8\nmorph_piece_ptrs: .quad '+','.join('morph_surface_'+str(i) for i in range(len(pieces)))+'\n'
    source+='morph_piece_lengths: .long '+','.join(str(len(p)) for p in pieces)+'\n'
    source+=""".section .bss
.balign 16
morph_gen_bytes: .skip (SAMPLE_LEN+CTX)*16
morph_gen_len: .skip 8
.ifdef MORPH_UTF8_SELFTEST
morph_utf8_test_buffer: .skip 513
.endif
.section .text
# Append an exact surface without writing partial tokens to stdout.
morph_append_token:
    cmp edi,V
    jae fatal
    cmp edi,256
    jae .Lmorph_surface
    mov edx,edi
    mov rcx,[rip+morph_gen_len]
    lea rax,[rip+morph_gen_bytes]
    mov byte ptr [rax+rcx],dl
    inc rcx
    mov [rip+morph_gen_len],rcx
    ret
.Lmorph_surface:
    sub edi,256
    lea rax,[rip+morph_piece_ptrs]
    mov rsi,[rax+rdi*8]
    lea rax,[rip+morph_piece_lengths]
    mov ecx,[rax+rdi*4]
    lea rdi,[rip+morph_gen_bytes]
    add rdi,[rip+morph_gen_len]
    add [rip+morph_gen_len],rcx
    rep movsb
    ret
# rsi=bytes, rcx=length. Returns0 for strict UTF-8,1 otherwise.
# No byte is emitted. ASCII/NUL allowed; rejects overlong, surrogate,
# out-of-range scalars and truncated sequences.
morph_validate_utf8:
    test rcx,rcx
    jz .Lmorph_utf8_ok
    movzx eax,byte ptr [rsi]
    cmp eax,0x80
    jb .Lmorph_utf8_one
    cmp eax,0xc2
    jb .Lmorph_utf8_bad
    cmp eax,0xdf
    jbe .Lmorph_utf8_two
    cmp eax,0xef
    jbe .Lmorph_utf8_three
    cmp eax,0xf4
    jbe .Lmorph_utf8_four
    jmp .Lmorph_utf8_bad
.Lmorph_utf8_one:
    inc rsi
    dec rcx
    jmp morph_validate_utf8
.Lmorph_utf8_two:
    cmp rcx,2
    jb .Lmorph_utf8_bad
    movzx r8d,byte ptr [rsi+1]
    sub r8d,0x80
    cmp r8d,0x3f
    ja .Lmorph_utf8_bad
    add rsi,2
    sub rcx,2
    jmp morph_validate_utf8
.Lmorph_utf8_three:
    cmp rcx,3
    jb .Lmorph_utf8_bad
    movzx r8d,byte ptr [rsi+1]
    mov r9d,r8d
    sub r8d,0x80
    cmp r8d,0x3f
    ja .Lmorph_utf8_bad
    movzx r8d,byte ptr [rsi+2]
    sub r8d,0x80
    cmp r8d,0x3f
    ja .Lmorph_utf8_bad
    cmp eax,0xe0
    jne .Lmorph_utf8_no_overlong3
    cmp r9d,0xa0
    jb .Lmorph_utf8_bad
.Lmorph_utf8_no_overlong3:
    cmp eax,0xed
    jne .Lmorph_utf8_advance3
    cmp r9d,0xa0
    jae .Lmorph_utf8_bad
.Lmorph_utf8_advance3:
    add rsi,3
    sub rcx,3
    jmp morph_validate_utf8
.Lmorph_utf8_four:
    cmp rcx,4
    jb .Lmorph_utf8_bad
    movzx r8d,byte ptr [rsi+1]
    mov r9d,r8d
    sub r8d,0x80
    cmp r8d,0x3f
    ja .Lmorph_utf8_bad
    movzx r8d,byte ptr [rsi+2]
    sub r8d,0x80
    cmp r8d,0x3f
    ja .Lmorph_utf8_bad
    movzx r8d,byte ptr [rsi+3]
    sub r8d,0x80
    cmp r8d,0x3f
    ja .Lmorph_utf8_bad
    cmp eax,0xf0
    jne .Lmorph_utf8_no_overlong4
    cmp r9d,0x90
    jb .Lmorph_utf8_bad
.Lmorph_utf8_no_overlong4:
    cmp eax,0xf4
    jne .Lmorph_utf8_advance4
    cmp r9d,0x8f
    ja .Lmorph_utf8_bad
.Lmorph_utf8_advance4:
    add rsi,4
    sub rcx,4
    jmp morph_validate_utf8
.Lmorph_utf8_ok:
    xor eax,eax
    ret
.Lmorph_utf8_bad:
    mov eax,1
    ret
.Lmorph_generation_invalid:
    mov eax,60
    mov edi,2
    syscall
.ifdef MORPH_UTF8_SELFTEST
# Test-only native stdin driver exercises the same validator atomically.
morph_utf8_selftest:
    xor r12d,r12d
.Lmorph_utf8_read:
    xor eax,eax
    xor edi,edi
    lea rsi,[rip+morph_utf8_test_buffer]
    add rsi,r12
    mov edx,513
    sub rdx,r12
    syscall
    cmp rax,-4
    je .Lmorph_utf8_read
    test rax,rax
    js fatal
    jz .Lmorph_utf8_test_validate
    add r12,rax
    cmp r12,512
    jbe .Lmorph_utf8_read
    mov eax,60
    mov edi,4
    syscall
.Lmorph_utf8_test_validate:
    lea rsi,[rip+morph_utf8_test_buffer]
    mov rcx,r12
    call morph_validate_utf8
    test eax,eax
    jnz .Lmorph_generation_invalid
    lea rsi,[rip+morph_utf8_test_buffer]
    mov rdx,r12
    call write_all
    mov eax,60
    xor edi,edi
    syscall
.endif
.section .note.GNU-stack,"",@progbits
"""
    source=replace_exact(source,".Lend:\n    lea r10,[rip+number]", """.Lend:
    lea rsi,[rip+morph_gen_bytes]
    mov rcx,[rip+morph_gen_len]
    call morph_validate_utf8
    test eax,eax
    jnz .Lmorph_generation_invalid
    SAY s_gen,l_gen
    lea rsi,[rip+morph_gen_bytes]
    mov rdx,[rip+morph_gen_len]
    call write_all
    lea r10,[rip+number]""")
    return source.encode()
def prepare():
    WORK.mkdir(parents=True,exist_ok=True)
    train,train_ids=token_ids(TRAIN); valid,valid_ids=token_ids(VALID)
    if len(train_ids)<=CTX or len(valid_ids)<=CTX or not set(valid_ids).issubset(set(train_ids)):
        raise ValueError('toy corpus contract: length or unseen validation IDs')
    for name,data in [('train.ids',train),('valid.ids',valid),('train.txt',TRAIN),('valid.txt',VALID),('vocab.json',VOCAB.read_bytes()),('morph_attention.s',assembly_source())]:
        (WORK/name).write_bytes(data)
    return train_ids,valid_ids
def compile_native(destination,*,cwd=WORK,**opts):
    flags=[f'-Wa,--defsym,{key}={value}' for key,value in opts.items()]
    command(['cc','-nostdlib','-static','-no-pie','-Wl,--build-id=none',*flags,'-o',destination,'morph_attention.s'],cwd=cwd)
    if b'INTERP' in command(['readelf','-lW',destination]): raise ValueError('interpreter')
    if b'NEEDED' in command(['readelf','-dW',destination]): raise ValueError('dynamic dependency')
    if command(['nm','-u',destination]).strip(): raise ValueError('undefined symbol')
def finite_weights(raw):
    if len(raw)!=WEIGHT_BYTES: raise ValueError('incompatible checkpoint geometry')
    if not all(math.isfinite(v) for (v,) in struct.iter_unpack('<f',raw)): raise ValueError('non-finite checkpoint')
    return sha(raw)
def validate(info,payloads):
    if info.get('format')!=FORMAT or info.get('vocab_sha256')!=VOCAB_SHA or info.get('vocab_size')!=V or info.get('weights_bytes')!=WEIGHT_BYTES or info.get('context_tokens')!=CTX:
        raise ValueError('incompatible checkpoint/tokenizer contract')
    finite_weights(payloads['weights.f32'])
    if set(info.get('files',{}))!=set(payloads): raise ValueError('checkpoint member set')
    for name,data in payloads.items():
        if info['files'][name]!=sha(data): raise ValueError('checkpoint member mismatch: '+name)
    if sha(payloads['vocab.json'])!=VOCAB_SHA: raise ValueError('vocabulary mismatch')
def verified_snapshot(destination):
    info=json.loads((WORK/'manifest.json').read_bytes())
    # Fixed names: manifest-provided paths never influence filesystem access.
    names=('weights.f32','morph_attention.s','train.ids','valid.ids','vocab.json')
    payloads={name:(WORK/name).read_bytes() for name in names}
    validate(info,payloads)
    for name,data in payloads.items():
        p=destination/name; p.write_bytes(data)
    validate(info,{name:(destination/name).read_bytes() for name in names})
    for name in names: (destination/name).chmod(0o400)
    return info
def metric(data,key):
    m=re.search(re.escape(key.encode())+rb' (\d+)\n',data)
    if not m: raise ValueError('missing metric '+key)
    return int(m.group(1))
def sample(data):
    marker=b'--- attention sample ---\n'
    if data.count(marker)!=1 or not data.endswith(b'\n'): raise ValueError('sample framing')
    return data.split(marker,1)[1][:-1]
def build():
    started=time.perf_counter(); train_ids,valid_ids=prepare()
    trainer=WORK/'morph-train'; exporter=WORK/'morph-export'; infer=WORK/'morph-infer'
    compile_native(trainer,EPOCHS=EPOCHS,SEED=42,SAMPLE_LEN=SAMPLE)
    trained=command([trainer]); (WORK/'train.stdout').write_bytes(trained)
    compile_native(exporter,EPOCHS=EPOCHS,SEED=42,SAMPLE_LEN=0,EXPORT_WEIGHTS=1)
    exported=command([exporter]); weights=exported[-WEIGHT_BYTES:]
    finite_weights(weights); (WORK/'weights.f32').write_bytes(weights)
    if not exported[:-WEIGHT_BYTES].endswith(b'\n') or metric(exported[:-WEIGHT_BYTES],'final_train_loss_x1000')!=metric(trained,'final_train_loss_x1000'):
        raise ValueError('training export framing/mismatch')
    names=('weights.f32','morph_attention.s','train.ids','valid.ids','vocab.json')
    manifest={'format':FORMAT,'vocab_sha256':VOCAB_SHA,'vocab_size':V,'context_tokens':CTX,
      'hidden_width':D,'epochs':EPOCHS,'weights_bytes':WEIGHT_BYTES,'base_assembly_sha256':BASE_SHA,
      'files':{name:sha((WORK/name).read_bytes()) for name in names},
      'training_origin':'Original tiny demonstration phrases authored for this test; no interaction data',
      'limitations':['Tiny related train/validation phrases, not general Arabic evaluation','Standard SGD, not continual LoRA','Weights are integrity-bound, not authenticated']}
    (WORK/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='niyah-morph-infer-') as td:
        sandbox=pathlib.Path(td); verified_snapshot(sandbox)
        compile_native(sandbox/'infer',cwd=sandbox,INFERENCE=1,SEED=42,SAMPLE_LEN=SAMPLE)
        shutil.copyfile(sandbox/'infer',infer); infer.chmod(0o755)
    generated=command([infer]); (WORK/'infer.stdout').write_bytes(generated)
    if sample(trained)!=sample(generated): raise ValueError('frozen inference replay mismatch')
    summary={'schema':'niyah-morph-attention-build-v1','result':'PASS','exit':0,
      'workdir':str(WORK),'elapsed_seconds':time.perf_counter()-started,
      'vocab_size':V,'vocab_sha256':VOCAB_SHA,'weight_bytes':WEIGHT_BYTES,
      'train_bytes':len(TRAIN),'train_tokens':len(train_ids),'valid_bytes':len(VALID),'valid_tokens':len(valid_ids),
      'initial_train_millinats_per_token':metric(trained,'initial_train_loss_x1000'),
      'final_train_millinats_per_token':metric(trained,'final_train_loss_x1000'),
      'heldout_millinats_per_token':metric(trained,'heldout_loss_x1000'),
      'binary_sha256':{p.name:sha(p.read_bytes()) for p in (trainer,exporter,infer)},
      'weights_sha256':sha(weights),'manifest_sha256':sha((WORK/'manifest.json').read_bytes()),
      'generated_sample':sample(generated).decode('utf-8'),'old_byte_checkpoint_compatible':False,
      'no_libc':True,'runtime_dependencies':['Linux x86-64 syscall ABI'],'commands':COMMANDS.copy()}
    (WORK/'build_receipt.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary
if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1]!='build': raise SystemExit('usage: morph_attention.py build')
    summary=build(); print(json.dumps({k:v for k,v in summary.items() if k!='commands'},ensure_ascii=False))
