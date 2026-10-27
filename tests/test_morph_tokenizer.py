#!/usr/bin/env python3
"""External stdlib reference and adversarial tests; no runtime dependency."""
import hashlib, json, os, pathlib, random, struct, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
EXE=ROOT/'build/morph-tokenizer'
VOC=ROOT/'src/morph_tokenizer/vocab.json'
vocab=json.loads(VOC.read_bytes()); digest=hashlib.sha256(VOC.read_bytes()).digest()
pieces=[bytes([i]) for i in range(256)]+[s.encode() for s in vocab['pieces']]
records=[]; start=time.perf_counter()
def run(mode,data=b'',expect=0):
    p=subprocess.run([str(EXE),mode],input=data,capture_output=True,timeout=20)
    assert p.returncode==expect,(mode,p.returncode,expect,p.stderr)
    return p.stdout
def parse(blob):
    assert blob[:8]==b'NYMORPH\0' and len(blob)>=64
    version,vocab_version,count,length=struct.unpack_from('<IIQQ',blob,8)
    assert (version,vocab_version)==(1,1) and blob[32:64]==digest
    assert len(blob)==64+count*2
    ids=struct.unpack_from('<'+'H'*count,blob,64)
    assert all(i<len(pieces) for i in ids)
    raw=b''.join(pieces[i] for i in ids)
    assert len(raw)==length
    return raw,ids
def roundtrip(name,data):
    blob=run('encode',data); raw,ids=parse(blob)
    assert raw==data and run('decode',blob)==data
    assert run('encode',data)==blob
    records.append({'name':name,'bytes':len(data),'tokens':len(ids),'exit':0})
    return blob,ids
fixtures={
 'empty':b'', 'clitics':'والكتاب فبالكتاب سيكتبها للكتاب كتابه فسوف يكتب'.encode(),
 'diacritics':'وَالْكِتَابُ سَيَكْتُبُهَا كـتـاب  أكتبُ'.encode(),
 'unicode_whitespace':'كتاب\u00a0والكتاب\u2003سيكتبها\u2028لغة\u202fنية\u3000حساب\r\n'.encode(),
 'punctuation':'«والكتاب»،سيكتبها؛ كتاب؟(سلام)!'.encode(),
 'mixed':'hello, مرحبا بالعالم; Al-kitab 123 😀 日本語\x00'.encode(),
 'invalid_utf8':bytes([0,0xc0,0xaf,0xed,0xa0,0x80,0xf4,0x90,0x80,0x80,0xff,0x80])+b'\xe2\x82',
 'all_bytes':bytes(range(256)),
 'long_word':('ك'*1000).encode(),
 'maximum':b'x'*1048576,
}
for name,data in fixtures.items(): roundtrip(name,data)
random.seed(20261027)
for i in range(80):
    n=random.randrange(0,1025); data=bytes(random.randrange(256) for _ in range(n))
    roundtrip('random_%02d'%i,data)
blob,ids=roundtrip('segmentation_contract','والكتاب سيكتبها'.encode())
assert len(ids)==8,ids  # [و,ال,كتاب,space,س,ي,كتب,ها]
expected=['و','ال','كتاب',' ','س','ي','كتب','ها']
assert [pieces[i].decode() for i in ids]==expected
valid=run('encode','كتاب'.encode())
malformed=[]
for n in [0,1,7,8,31,63,len(valid)-1]: malformed.append(('truncate_%d'%n,valid[:n]))
for offset in [0,8,12,16,24,32,63]:
    x=bytearray(valid); x[offset]^=0xff; malformed.append(('header_flip_%d'%offset,bytes(x)))
x=bytearray(valid); x[64:66]=b'\xff\xff'; malformed.append(('invalid_id',bytes(x)))
malformed.append(('extra_byte',valid+b'x'))
x=bytearray(valid); struct.pack_into('<Q',x,16,2**64-1); malformed.append(('huge_count',bytes(x)))
x=bytearray(valid); struct.pack_into('<Q',x,24,0); malformed.append(('underflow_length',bytes(x)))
for name,blob in malformed:
    assert run('decode',blob,2)==b'',name
    records.append({'name':name,'exit':2,'stdout_bytes':0})
assert run('encode',b'x'*1048577,4)==b''
assert run('decode',b'x'*(64+2*1048576+1),4)==b''
assert run('invalid-mode',b'',64)==b''
assert run('vocab')==VOC.read_bytes()
with open('/dev/full','wb') as full:
    p=subprocess.run([str(EXE),'encode'],input=b'abc',stdout=full,stderr=subprocess.PIPE)
    assert p.returncode==3,p.returncode
for command in [['readelf','-lW',str(EXE)],['readelf','-dW',str(EXE)],['nm','-u',str(EXE)]]:
    p=subprocess.run(command,capture_output=True,text=True); assert p.returncode==0
    if command[1]=='-lW': assert 'INTERP' not in p.stdout
    elif command[1]=='-dW': assert 'NEEDED' not in p.stdout
    else: assert not p.stdout.strip()
receipt={'schema':'niyah-morph-tokenizer-tests-v1','result':'PASS','exit':0,
 'elapsed_seconds':time.perf_counter()-start,'vocab_size':len(pieces),'vocab_sha256':digest.hex(),
 'binary_sha256':hashlib.sha256(EXE.read_bytes()).hexdigest(),'binary_bytes':EXE.stat().st_size,
 'roundtrip_cases':91,'malformed_cases':len(malformed),'records':records,
 'runtime_dependencies':['Linux x86-64 syscall ABI'],'dynamic_dependencies':[],
 'old_byte_checkpoint_compatible':False,'arabic_generalization_verified':False,
 'kv_cache_reduction_measured':False}
out=ROOT/'evidence/nexus-morph-tokenizer.json'; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='records'},ensure_ascii=False))
