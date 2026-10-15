#!/usr/bin/env python3
"""Fail-closed checkpoint corruption tests with copies, never live weights."""
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("niyah_checkpoint",ROOT/"tools"/"checkpoint_attention.py")
checkpoint=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)

def must_reject(label,action):
    try:
        action()
    except (ValueError,FileNotFoundError,json.JSONDecodeError) as exc:
        print(f"PASS negative {label}: rejected ({type(exc).__name__})")
    else:
        raise AssertionError(f"FAIL-OPEN: {label}")

def main():
    source_bytes=checkpoint.WEIGHTS.read_bytes()
    source_manifest=checkpoint.MANIFEST.read_bytes()
    checkpoint.verify()
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        checkpoint.WEIGHTS=path/"weights.f32"
        checkpoint.MANIFEST=path/"manifest.json"
        checkpoint.WEIGHTS.write_bytes(source_bytes)
        checkpoint.MANIFEST.write_bytes(source_manifest)
        checkpoint.verify()

        original=bytearray(source_bytes)
        original[60]^=0x01
        checkpoint.WEIGHTS.write_bytes(original)
        must_reject("modified training parameter",checkpoint.verify)

        checkpoint.WEIGHTS.write_bytes(source_bytes[:-4])
        must_reject("truncated parameters",checkpoint.verify)

        checkpoint.WEIGHTS.write_bytes(source_bytes)
        info=json.loads(source_manifest)
        info["sha256_assembly"]="0"*64
        checkpoint.MANIFEST.write_text(json.dumps(info))
        must_reject("tampered source digest",checkpoint.verify)

        checkpoint.MANIFEST.write_bytes(source_manifest)
        checkpoint.WEIGHTS.unlink()
        must_reject("missing parameters",checkpoint.verify)
    print("PASS corruption tests used only disposable weight/manifest copies")

if __name__=="__main__":main()
