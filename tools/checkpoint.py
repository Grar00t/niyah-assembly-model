from pathlib import Path
import argparse, array, hashlib, json, math, os, subprocess, sys, tempfile
ROOT = Path(__file__).resolve().parents[1]
SIZE = 67108864

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def identity():
    return {name: digest(ROOT/name) for name in ('src/trigram.s','data/corpus.txt')}

def compile_model(output, flag=None):
    cmd = ['cc','-nostdlib','-static','-no-pie']
    if flag:
        cmd += ['-Wa,--defsym,'+flag+'=1']
    subprocess.run(cmd+['-o',str(output),'src/trigram.s'],cwd=ROOT,check=True,timeout=60)

def validate_weights(path):
    if path.stat().st_size != SIZE:
        raise ValueError('invalid weight file size')
    with path.open('rb') as f:
        while raw := f.read(1048576):
            values = array.array('f')
            if values.itemsize != 4:
                raise ValueError('host float size unsupported')
            values.frombytes(raw)
            if sys.byteorder != 'little':
                values.byteswap()
            if not all(math.isfinite(x) for x in values):
                raise ValueError('nonfinite weight')

def verify():
    m = json.loads((ROOT/'models/checkpoint.json').read_text())
    if m['format'] != 'raw-float32-le' or m['weight_bytes'] != SIZE:
        raise ValueError('unsupported checkpoint format')
    if m['files'] != identity():
        raise ValueError('source or corpus changed; run make checkpoint')
    weights = ROOT/'models/weights.f32'
    if digest(weights) != m['weights_sha256']:
        raise ValueError('weight checksum mismatch')
    validate_weights(weights)
    return m

def train():
    data = (ROOT/'data/corpus.txt').read_bytes()
    data.decode('utf-8',errors='strict')
    if len(data) < 3:
        raise ValueError('corpus requires at least three bytes')
    before = identity()
    models = ROOT/'models'
    models.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models) as td:
        td = Path(td)
        exe = td/'exporter'
        raw = td/'weights.f32'
        compile_model(exe,'EXPORT_WEIGHTS')
        with raw.open('wb') as f:
            subprocess.run([str(exe)],cwd=ROOT,stdout=f,check=True,timeout=300)
        validate_weights(raw)
        if before != identity():
            raise ValueError('source or corpus changed during training')
        manifest = {'format':'raw-float32-le','weight_bytes':SIZE,
                    'files':before,'weights_sha256':digest(raw),
                    'optimizer_state_saved':False}
        meta = td/'checkpoint.json'
        meta.write_text(json.dumps(manifest,indent=2)+'\n')
        os.replace(raw,models/'weights.f32')
        os.replace(meta,models/'checkpoint.json')
    verify()
    print('PASS: checkpoint saved and verified')

def generate():
    verify()
    with tempfile.TemporaryDirectory() as td:
        exe = Path(td)/'generate'
        compile_model(exe,'INFERENCE')
        result = subprocess.run([str(exe)],cwd=ROOT,capture_output=True,check=True,timeout=60)
    return result.stdout

def test():
    restored = generate()
    with tempfile.TemporaryDirectory() as td:
        exe = Path(td)/'trainer'
        compile_model(exe)
        trained = subprocess.run([str(exe)],cwd=ROOT,capture_output=True,check=True,timeout=300).stdout
    marker = b'--- sample ---\n'
    if restored.count(marker)!=1 or trained.count(marker)!=1:
        raise ValueError('sample marker mismatch')
    if restored.split(marker,1)[1] != trained.split(marker,1)[1]:
        raise ValueError('checkpoint replay differs')
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td)/'bad.f32'
        bad.write_bytes(b'x')
        try:
            validate_weights(bad)
        except ValueError:
            pass
        else:
            raise ValueError('truncated checkpoint accepted')
    print('PASS: checkpoint replay and truncated-file rejection')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['train','verify','generate','test'])
    command = parser.parse_args().command
    try:
        (ROOT/'build').mkdir(exist_ok=True)
        if command=='train': train()
        elif command=='verify': verify(); print('PASS: checkpoint identity')
        elif command=='generate': sys.stdout.buffer.write(generate())
        else: test()
    except (ValueError,OSError,KeyError,subprocess.SubprocessError) as e:
        print('FAIL: '+str(e),file=sys.stderr)
        raise SystemExit(1)
