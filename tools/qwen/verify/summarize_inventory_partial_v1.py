#!/usr/bin/env python3
import csv
import json
from pathlib import Path
ROOT=Path("/home/a/qwen-layer0-quant-verify-v1")
SRC=ROOT/"inventory_v1.tsv"
OUT=ROOT/"inventory_partial_summary_v1.json"
stats={"files":0,"directories":0,"symlinks":0,"rows":0,"archive_candidate_rows":0}
roots=[]
last=""
with SRC.open("r",encoding="utf-8",newline="") as input:
    for row in csv.DictReader(input,delimiter="\t"):
        stats["rows"]+=1
        last=row.get("path","")
        kind=row.get("kind","")
        if kind=="file":stats["files"]+=1
        elif kind=="directory":stats["directories"]+=1
        else:stats["symlinks"]+=1
        if row.get("archive_candidate_reason"):
            stats["archive_candidate_rows"]+=1
        p=Path(last)
        if kind=="directory" and p.parent==Path("/home/a") and ("niyah" in p.name.lower() or "qwen" in p.name.lower()):
            roots.append({k:row.get(k,"") for k in
                 ("path","size_bytes","git_branch","repo_dirty","last_repo_commit_date","purpose","archive_candidate_reason")})
        if kind=="directory" and p==Path("/home/a/models/Qwen3-8B-Base"):
            roots.append({k:row.get(k,"") for k in
                 ("path","size_bytes","git_branch","repo_dirty","last_repo_commit_date","purpose","archive_candidate_reason")})
result={
    "status":"PARTIAL",
    "reason":"Read-only recursive scan interrupted after slow DrvFS enumeration of D:/Niyah.Engine corpus; rows already written are preserved, not a complete inventory",
    "table":str(SRC),
    "last_recorded_path":last,
    "counts":stats,
    "home_matching_roots_finished":len(roots),
    "home_root_summaries":sorted(roots,key=lambda x:x["path"]),
    "windows_engine_repo": {
       "path":"D:/Niyah.Engine",
       "git_branch":"feature/oracle-engine-three-lobe-20261002",
       "last_repo_commit":"2a0ec79, 2026-10-12T02:16:31+03:00",
       "status":"DIRTY (observed 2026-10-30)",
       "folder_size_bytes":"UNKNOWN: full scan incomplete",
    },
}
with OUT.open("x",encoding="utf-8") as output:
    json.dump(result,output,indent=2,ensure_ascii=False)
print(json.dumps({"summary":str(OUT),"counts":stats,"roots":len(roots),"last_recorded_path":last},ensure_ascii=False))
for r in sorted(roots,key=lambda x:int(x["size_bytes"] or 0),reverse=True)[:30]:
    print("|".join((r["path"],r["size_bytes"],r["git_branch"],r["repo_dirty"],r["last_repo_commit_date"],r["archive_candidate_reason"][:50])))
