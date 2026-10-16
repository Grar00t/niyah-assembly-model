#!/usr/bin/env python3
"""TOCTOU regression: never build inference from mutable original weight path."""
from pathlib import Path
import hashlib
import importlib.util
import tempfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("chk",ROOT/"tools"/"checkpoint_attention.py")
chk=importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

def main():
    # All fault injection targets disposable files, never the live checkpoint.
    realweights=chk.WEIGHTS
    original_sha=hashlib.sha256(realweights.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="niyah-snapshot-regression-") as td:
        tmp=Path(td)
        copy_weight=tmp/"weight.f32"
        copy_manifest=tmp/"manifest.json"
        original_weight=chk.WEIGHTS.read_bytes()
        copy_weight.write_bytes(original_weight)
        copy_manifest.write_bytes(chk.MANIFEST.read_bytes())
        chk.WEIGHTS=copy_weight
        chk.MANIFEST=copy_manifest
        chk.verify()
        baseline=chk.generate(sample=16,auto=True)
        old_build=chk.build
        touches=0
        def mutate_between_snapshot_and_assembler(exe,**opts):
            nonlocal touches
            touches+=1
            corrupt=bytearray(original_weight)
            corrupt[128]^=0x55
            copy_weight.write_bytes(corrupt)
            try:
                return old_build(exe,**opts)
            finally:
                copy_weight.write_bytes(original_weight)
        chk.build=mutate_between_snapshot_and_assembler
        try:
            actual=chk.generate(sample=16,auto=True)
        finally:
            chk.build=old_build
        assert touches==1
        assert actual==baseline, "inference consumed changed mutable original after verification"
        print("PASS: post-verification original weight modification does NOT alter inference")
        assert copy_weight.read_bytes()==original_weight

        # Distinct failure: original weights changed *after* verify()
        # but before their own snapshot capture; fail closed on digest.
        old_verify=chk.verify
        def change_after_verify():
            old_verify()
            altered=bytearray(original_weight)
            altered[80]^=0x10
            copy_weight.write_bytes(altered)
        chk.verify=change_after_verify
        try:
            try:
                chk.generate(sample=16)
            except ValueError as ex:
                assert "snapshot mismatch" in str(ex),str(ex)
            else:
                raise AssertionError("FAIL OPEN: checkpoint mutated after verification")
        finally:
            chk.verify=old_verify
            copy_weight.write_bytes(original_weight)
        print("PASS: changed original between verification and snapshot fails closed")
    assert hashlib.sha256(realweights.read_bytes()).hexdigest()==original_sha
    print("PASS: production model weights are unchanged; all attacks used private copies")

if __name__=="__main__":
    main()
