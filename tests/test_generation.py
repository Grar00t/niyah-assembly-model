from pathlib import Path
import subprocess, tempfile, json, re
root=Path(__file__).resolve().parents[1]
allowed=set((root/'data/corpus.txt').read_bytes())
results=[]
with tempfile.TemporaryDirectory() as tmp:
    exe=Path(tmp)/'model'
    for seed in (1,42,2463534242,4294967295):
        subprocess.run(['cc','-nostdlib','-static','-no-pie',
            '-Wa,--defsym,SEED='+str(seed),'-Wa,--defsym,SAMPLE_LEN=4096',
            '-o',str(exe),str(root/'src/trigram.s')],check=True,timeout=30)
        p=subprocess.run([str(exe)],capture_output=True,timeout=60)
        parts=p.stdout.split(b'--- sample ---\n')
        sample=parts[1][:-1] if len(parts)==2 and parts[1].endswith(b'\n') else b''
        lines=re.findall(rb'^epoch (\d+) loss\*1000 (\d+)$',parts[0],re.M)
        outside=sum(b not in allowed for b in sample)
        ok=p.returncode==0 and not p.stderr and len(sample)==4096 and outside==0 and len(lines)==11
        results.append({'test':'seed '+str(seed),'pass':ok,'length':len(sample),'outside':outside})
        if seed==42:
            q=subprocess.run([str(exe)],capture_output=True,timeout=60)
            results.append({'test':'replay','pass':q.returncode==0 and q.stdout==p.stdout})
report={'tests':results,'passed':sum(x['pass'] for x in results),'total':len(results),'memory_safety_proven':False}
(root/'verification_results.json').write_text(json.dumps(report,indent=2)+'\n')
for r in results:
    print(('PASS' if r['pass'] else 'FAIL')+': '+r['test'])
print(str(report['passed'])+'/'+str(report['total']))
raise SystemExit(0 if all(r['pass'] for r in results) else 1)
