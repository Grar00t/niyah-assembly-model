import csv,gzip,json,hashlib,collections,datetime
from pathlib import Path
o=Path.home()/'qwen-layer0-quant-verify-v1'
home=o/'inventory_v1.tsv'
engine=o/'inventory_engine_windows_v1.tsv.gz'
dest=o/'inventory_complete_v1.tsv.gz'
summary=o/'inventory_complete_summary_v1.json'
counts=collections.Counter()
archive=collections.Counter()
home_roots=0
hsource=json.loads((o/'inventory_partial_summary_v1.json').read_text())
dsource=json.loads((o/'inventory_engine_windows_summary_v1.json').read_text())
if hsource.get('home_matching_roots_finished')!=41 or dsource.get('status')!='COMPLETE':
 raise SystemExit('STOP inventory source incomplete')
if dest.exists() or summary.exists():raise SystemExit('STOP: destination exists')
with gzip.open(dest,'xt',encoding='utf-8',newline='',compresslevel=4) as fout:
 writer=csv.writer(fout,delimiter='\t',lineterminator='\n')
 with home.open('r',encoding='utf-8',newline='') as f:
  reader=csv.DictReader(f,delimiter='\t')
  writer.writerow(reader.fieldnames)
  for row in reader:
   if row['path'].startswith('/mnt/d/Niyah.Engine'): continue
   writer.writerow([row[k] for k in reader.fieldnames])
   counts['home_'+row['kind']]+=1
   if row['archive_candidate_reason']:archive['home']+=1
 with gzip.open(engine,'rt',encoding='utf-8',newline='') as f:
  reader=csv.DictReader(f,delimiter='\t')
  for row in reader:
   writer.writerow([row[k] for k in reader.fieldnames])
   counts['engine_'+row['kind']]+=1
   if row['archive_candidate_reason']:archive['engine']+=1
sha=hashlib.sha256(dest.read_bytes()).hexdigest()
data={'status':'COMPLETE','manifest':str(dest),'manifest_sha256':sha,
      'source_home':str(home),'source_engine':str(engine),
      'home_root_count':41,'counts':dict(counts),'archive_candidates':dict(archive),
      'engine_root_bytes':dsource['total_file_bytes'],
      'engine_scan_errors':dsource['counts']['errors'],
      'home_root_summaries':hsource['home_root_summaries'],
      'engine_top_level_directories':dsource['top_level_directories'],
      'creation_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with summary.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,ensure_ascii=False)
print('COMPLETE_MANIFEST',str(dest),'SHA256',sha)
print('COUNTS',dict(counts),'CANDIDATES',dict(archive))
print('HOME_ROOTS',len(data['home_root_summaries']))
print('D_BYTES',data['engine_root_bytes'])
