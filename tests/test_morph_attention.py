#!/usr/bin/env python3
"""Native u16 model integration, independent oracle, and checkpoint negatives."""
import copy, hashlib, importlib.util, json, math, pathlib, struct, tempfile, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m=load('morph_native_builder',ROOT/'tools/morph_attention.py')
oracle=load('byte_model_reference_oracle',ROOT/'tests/test_attention.py')
start=time.perf_counter(); info=m.build()
train_ids=tuple(x[0] for x in struct.iter_unpack('<H',(m.WORK/'train.ids').read_bytes()))
valid_ids=tuple(x[0] for x in struct.iter_unpack('<H',(m.WORK/'valid.ids').read_bytes()))
assert max(train_ids)>255 and max(valid_ids)>255,'must exercise actual u16 vocabulary indices'
assert set(valid_ids).issubset(set(train_ids))
trained=(m.WORK/'train.stdout').read_bytes(); infer=(m.WORK/'infer.stdout').read_bytes()
assert info['final_train_millinats_per_token']<info['initial_train_millinats_per_token']
assert m.command([m.WORK/'morph-train'])==trained,'deterministic native training'
assert m.command([m.WORK/'morph-infer'])==infer,'deterministic frozen inference'
assert m.sample(trained)==m.sample(infer)
assert m.sample(infer).decode('utf-8')
raw=(m.WORK/'weights.f32').read_bytes(); vals=struct.unpack('<'+'f'*(len(raw)//4),raw)
arrays=[]; off=0
for count in (m.V*16,128,256,16,m.V*16,m.V,256,256,256):
    arrays.append(vals[off:off+count]); off+=count
assert off==len(vals)
oracle.ALLOWED=set(train_ids)
oracle_train=oracle.oracle_loss(train_ids,arrays)*1000
oracle_valid=oracle.oracle_loss(valid_ids,arrays)*1000
assert abs(oracle_train-info['final_train_millinats_per_token'])<3,(oracle_train,info)
assert abs(oracle_valid-info['heldout_millinats_per_token'])<3,(oracle_valid,info)
for matrix in arrays[6:]:
    assert max(abs(x-(1.0 if k//16==k%16 else 0.0)) for k,x in enumerate(matrix))>1e-6
manifest=json.loads((m.WORK/'manifest.json').read_bytes())
payloads={name:(m.WORK/name).read_bytes() for name in manifest['files']}
m.validate(manifest,payloads)
negatives=[]
def reject(name,metadata,files):
    try: m.validate(metadata,files)
    except ValueError as error: negatives.append({'name':name,'rejected':True,'reason':str(error)})
    else: raise AssertionError('accepted '+name)
for field,value in [('format','niyah-attention-f32-v2-learned-qkv'),('vocab_sha256','0'*64),('vocab_size',256),('context_tokens',16),('weights_bytes',38464)]:
    bad=copy.deepcopy(manifest); bad[field]=value; reject('contract_'+field,bad,payloads)
for name,data in [('weights.f32',raw[:-4]),('weights.f32',b'\x00\x00\xc0\x7f'+raw[4:]),('weights.f32',b'\0'*38464),('train.ids',payloads['train.ids']+b'\0\0'),('valid.ids',b'\xff\xff'+payloads['valid.ids'][2:]),('morph_attention.s',payloads['morph_attention.s']+b'\n'),('vocab.json',payloads['vocab.json']+b'\n')]:
    bad=payloads.copy(); bad[name]=data; reject('member_'+name,manifest,bad)
with tempfile.TemporaryDirectory(prefix='morph-snapshot-test-') as td:
    sandbox=pathlib.Path(td); m.verified_snapshot(sandbox)
    assert all((sandbox/name).read_bytes()==data for name,data in payloads.items())

# Report training overfit explicitly instead of treating it as quality progress.
m.compile_native(m.WORK/'morph-untrained',EPOCHS=0,SEED=42,SAMPLE_LEN=0)
untrained=m.command([m.WORK/'morph-untrained'])
untrained_heldout=m.metric(untrained,'heldout_loss_x1000')
# Native shared validator: oracle is Python's strict decoder.
m.compile_native(m.WORK/'morph-utf8-test',EPOCHS=0,SAMPLE_LEN=0,MORPH_UTF8_SELFTEST=1)
utf8_vectors=[b'',b'ASCII\0', 'مرحبا'.encode(),bytes([0x7f,0xc2,0x80,0xdf,0xbf,0xe0,0xa0,0x80,0xed,0x9f,0xbf,0xee,0x80,0x80,0xf0,0x90,0x80,0x80,0xf4,0x8f,0xbf,0xbf]),
 b'\x80',b'\xc0\xaf',b'\xc1\xbf',b'\xc2',b'\xc2A',b'\xe0\x9f\xbf',b'\xed\xa0\x80',b'\xed\xbf\xbf',b'\xe2\x82',b'\xf0\x8f\xbf\xbf',b'\xf4\x90\x80\x80',b'\xf5\x80\x80\x80',b'\xf0\x90\x80',b'\xff',b'\xef\xbf\xbf',b'good'+b'\xff']
utf8_receipts=[]
for data in utf8_vectors:
    try: data.decode('utf-8'); expected=0
    except UnicodeDecodeError: expected=2
    result=m.command([m.WORK/'morph-utf8-test'],data=data,expect=expected)
    assert result==(data if expected==0 else b'')
    utf8_receipts.append({'input_hex':data.hex(),'exit':expected,'stdout_bytes':len(result)})
assert m.command([m.WORK/'morph-utf8-test'],data=b'a'*513,expect=4)==b''

report={**{k:v for k,v in info.items() if k!='commands'},
 'schema':'niyah-morph-attention-tests-v1','result':'PASS','test_exit':0,
 'test_elapsed_seconds':time.perf_counter()-start,
 'independent_oracle_train_millinats_per_token':oracle_train,
 'independent_oracle_heldout_millinats_per_token':oracle_valid,
 'rejected_incompatible_checkpoints':negatives,
 'roundtrip_train_to_frozen_inference':True,'u16_ids_above_255_exercised':True,
 'qkv_parameters_changed_by_native_sgd':True,
 'untrained_heldout_millinats_per_token':untrained_heldout,
 'heldout_worse_than_untrained':info['heldout_millinats_per_token']>untrained_heldout,
 'native_atomic_utf8_validation':utf8_receipts,
 'generation_utf8_policy':'validate complete generated surface bytes before sample stdout; invalid exits2',
 'token_reduction_vs_raw_byte_fixture_only':{'train_ratio':len(m.TRAIN)/len(train_ids),'valid_ratio':len(m.VALID)/len(valid_ids)},
 'memory_continual_lora_implemented':False,'kv_cache_reduction_measured':False,
 'arabic_quality_or_bpe_advantage_verified':False,'commands':m.COMMANDS}
out=ROOT/'evidence/nexus-morph-attention.json'; out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='commands'},ensure_ascii=False))
