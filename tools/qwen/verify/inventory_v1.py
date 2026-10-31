#!/usr/bin/env python3
"""Read-only Niyah/Qwen inventory. New evidence files only; never change repositories."""
import csv
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time

HOME = Path("/home/a")
ENGINE = Path("/mnt/d/Niyah.Engine")
OUT = Path("/home/a/qwen-layer0-quant-verify-v1")
TABLE = OUT / "inventory_v1.tsv"
SUMMARY = OUT / "inventory_summary_v1.json"
PATTERN = re.compile(r"niyah|qwen", re.IGNORECASE)
errors = []
found_roots = []
found_files = []
start = time.monotonic()

def error(e):
    errors.append(str(e))

def executable(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired) as e:
        error(f"{cmd}: {e}")
        return ""

def repo_info(path):
    root = executable(["git", "-C", str(path), "rev-parse", "--show-toplevel"])
    if not root:
        return dict(root="", branch="-", dirty="N/A", last_commit_date="")
    branch = executable(["git", "-C", root, "branch", "--show-current"]) or "DETACHED"
    dirty = executable(["git", "-C", root, "status", "--porcelain=v1", "--untracked-files=normal"])
    last = executable(["git", "-C", root, "log", "-1", "--format=%cI"])
    return dict(root=root, branch=branch, dirty="DIRTY" if dirty else "CLEAN", last_commit_date=last)

def candidate(name):
    s = name.lower()
    if "lang-release" in s or "lang-output" in s or "lang-publish-verify" in s:
        return "ARCHIVE-CANDIDATE: dated compiler release/output copy; verify git HEAD and hashes before any decision"
    if "qwen-layer0-check-" in s or "qwen-working-" in s:
        return "ARCHIVE-CANDIDATE: dated diagnostic output; preserve receipts and compare before any decision"
    if ("niyah-" in s and ("-audit" in s or "-verification" in s or "-inspect" in s)):
        return "ARCHIVE-CANDIDATE: historical audit/work folder; evidence may be unique, do not move before inspection"
    return ""

def root_purpose(path):
    n = path.name.lower()
    if path == ENGINE:
        return "Native Windows C11 model runtime/training checkout"
    if "assembly-model" in n:
        return "Assembly neural experiments and native tensor/GEMM kernels; active"
    if "lang-integrated" in n:
        return "Arabic compiler with experimental tensor IR/backend; active dirty checkout"
    if n == "qwen3-8b-base":
        return "Qwen3-8B-Base model config, tokenizer and original pretrained weights"
    if "lang-" in n:
        return "Arabic language/compiler worktree, snapshot or test"
    if "qwen-layer0-quant" in n:
        return "Current independent Qwen layer-0 verification evidence"
    if "qwen" in n:
        return "Qwen model test or evidence work folder"
    if "niyah" in n:
        return "Niyah experiment/test/work folder; classify contents before archiving"
    return "Related path"

def file_purpose(path):
    n = path.name.lower()
    parts = {p.lower() for p in path.parts}
    if ".git" in parts:
        return "Git metadata (preserve)"
    if n.endswith(".safetensors") or n.endswith((".gguf", ".ckpt", ".bin", ".pt", ".pth")):
        return "Model/tensor weights or binary payload"
    if "tests" in parts:
        return "Test code/fixture"
    if "evidence" in parts or "reports" in parts:
        return "Measurement/evidence/receipt"
    if "docs" in parts:
        return "Documentation"
    if "build" in parts or n.endswith((".o", ".obj", ".so", ".dll", ".a", ".exe", ".elf")):
        return "Built artifact (not assumed disposable)"
    if "src" in parts or n.endswith((".c", ".h", ".s", ".cc", ".cpp", ".cu", ".py")):
        return "Source code"
    if ".venv" in parts or "site-packages" in parts:
        return "Local Python environment/dependency"
    if n.endswith((".log", ".json", ".txt", ".tsv", ".csv")):
        return "Configuration/data/log; exact purpose not verified"
    return "Related directory member; inspect before classification"

for parent, dirs, files in os.walk(HOME, followlinks=False, onerror=error):
    parent_path = Path(parent)
    for name in list(dirs):
        if PATTERN.search(name):
            child = parent_path / name
            found_roots.append(child)
            dirs.remove(name)
    for name in files:
        if PATTERN.search(name):
            found_files.append(parent_path / name)

found_roots.append(ENGINE)
found_roots.sort(key=str)
found_files.sort(key=str)
if not ENGINE.is_dir():
    error("Windows root not available: " + str(ENGINE))

headers = [
    "path", "kind", "size_bytes", "git_root", "git_branch", "repo_dirty",
    "last_repo_commit_date", "purpose", "archive_candidate_reason",
]
summary = dict(
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope="ALL matching niyah/qwen names under /home/a, every nested path in their matched directories, and all of D:/Niyah.Engine",
    size_definition="Regular files: st_size, directories: recursively summed regular-file sizes, symlinks: lstat size",
    last_commit_definition="Last commit date of containing Git repository; not per-file blame date",
    git_dirty_definition="Whole-repository porcelain status; not individual file status",
    roots=[], matched_files_outside_roots=[], errors=errors, counts={}, archive_candidate_count=0,
)
counts = dict(files=0, folders=0, symlinks=0, candidates=0, bytes=0)
with TABLE.open("x", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(headers)

    def add(path, typ, size, git, purpose, archive):
        writer.writerow([
            str(path), typ, size, git["root"], git["branch"], git["dirty"],
            git["last_commit_date"], purpose, archive,
        ])
        if typ == "file":
            counts["files"] += 1
            counts["bytes"] += size
        elif typ == "directory":
            counts["folders"] += 1
        else:
            counts["symlinks"] += 1
        if archive:
            counts["candidates"] += 1

    for root in found_roots:
        if not root.is_dir() or root.is_symlink():
            continue
        git = repo_info(root)
        archive = candidate(root.name)
        sizes = {}
        nfiles = 0
        ndirs = 0
        for parent, dirs, files in os.walk(root, topdown=False, followlinks=False, onerror=error):
            here = Path(parent)
            total = 0
            for name in files:
                item = here / name
                try:
                    st = item.lstat()
                    typ = "symlink" if item.is_symlink() else "file"
                    if typ == "file":
                        total += st.st_size
                        nfiles += 1
                    add(item, typ, st.st_size, git, file_purpose(item), archive)
                except OSError as e:
                    error(f"{item}: {e}")
            for name in dirs:
                item = here / name
                if item.is_symlink():
                    try:
                        add(item, "symlink", item.lstat().st_size, git, "Symlink; not followed", archive)
                    except OSError as e:
                        error(f"{item}: {e}")
                else:
                    total += sizes.get(item, 0)
            sizes[here] = total
            ndirs += 1
            add(here, "directory", total, git,
                root_purpose(root) if here == root else "Contained directory in " + root.name,
                archive)
        summary["roots"].append({
            "path": str(root), "size_bytes": sizes.get(root, 0),
            "file_count": nfiles, "directory_count": ndirs,
            "purpose": root_purpose(root), "git": git,
            "archive_candidate_reason": archive,
        })

    for item in found_files:
        try:
            st = item.lstat()
            git = repo_info(item.parent)
            purpose = file_purpose(item)
            add(item, "symlink" if item.is_symlink() else "file", st.st_size, git, purpose, "")
            summary["matched_files_outside_roots"].append(str(item))
        except OSError as e:
            error(f"{item}: {e}")

summary["counts"] = counts
summary["duration_seconds"] = round(time.monotonic() - start, 2)
summary["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
summary["archive_candidate_root_count"] = sum(bool(r["archive_candidate_reason"]) for r in summary["roots"])
with SUMMARY.open("x", encoding="utf-8") as out:
    json.dump(summary, out, ensure_ascii=False, indent=2)
print(json.dumps({
    "table": str(TABLE), "summary": str(SUMMARY), "counts": counts,
    "roots": len(summary["roots"]), "errors": len(errors),
    "archive_candidate_roots": summary["archive_candidate_root_count"],
    "duration_seconds": summary["duration_seconds"],
}, ensure_ascii=False), flush=True)
