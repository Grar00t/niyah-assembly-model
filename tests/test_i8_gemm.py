#!/usr/bin/env python3
"""Blocked AVX2 int8 GEMM against the current one-row niyah_i8_dot oracle.

The dot sequence below is the kernel embedded in qwen_native_layer0.py.
Real Qwen3-8B q_proj rows are checked when safetensors and numpy import.
"""
import hashlib, json, pathlib, subprocess, tempfile, textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
QWEN = pathlib.Path.home() / "qwen_native_layer0.py"
DOT = textwrap.dedent("""\
    .intel_syntax noprefix
    .text
    .global niyah_i8_dot
    .type niyah_i8_dot,@function
    niyah_i8_dot:
     vpxor ymm2,ymm2,ymm2
     xor ecx,ecx
    .Li8:
     vpmovsxbw ymm0,XMMWORD PTR [rdi+rcx]
     vpmovsxbw ymm1,XMMWORD PTR [rsi+rcx]
     vpmaddwd ymm0,ymm0,ymm1
     vpaddd ymm2,ymm2,ymm0
     add ecx,16
     cmp ecx,edx
     jb .Li8
     vextracti128 xmm1,ymm2,1
     vpaddd xmm0,xmm2,xmm1
     vpshufd xmm1,xmm0,0x4e
     vpaddd xmm0,xmm0,xmm1
     vpshufd xmm1,xmm0,0xb1
     vpaddd xmm0,xmm0,xmm1
     vmovd eax,xmm0
     vzeroupper
     ret
    .size niyah_i8_dot,.-niyah_i8_dot
    .section .note.GNU-stack,"",@progbits
    """)

def quant_rows(rows):
    import numpy as np
    out = []
    for row in rows:
        x = np.asarray(row, dtype=np.float64)
        m = float(np.max(np.abs(x))) if x.size else 0.0
        s = (m / 127.0) if m else 1.0
        q = np.clip(np.rint(x / s), -127, 127).astype(np.int8)
        out.append(np.ascontiguousarray(q))
    return out

def main():
    qwen = QWEN.read_text()
    needle = r"vpmaddwd ymm0,ymm0,ymm1\n vpaddd ymm2,ymm2,ymm0\n add ecx,16\n cmp ecx,edx\n jb .Li8"
    if needle not in qwen:
        raise SystemExit("current niyah_i8_dot is not the qwen_native_layer0.py kernel")
    if "niyah_i8_gemm(aq,m->w,acc,t,m->rows,m->cols)" not in qwen:
        raise SystemExit("qwen_native_layer0.py does not call niyah_i8_gemm")
    src = ROOT / "src" / "niyah_i8_gemm.s"
    harness = ROOT / "tests" / "i8_gemm_harness.c"
    with tempfile.TemporaryDirectory(prefix="niyah-i8-gemm-") as tmp:
        work = pathlib.Path(tmp)
        (work / "dot.s").write_text(DOT)
        exe = work / "harness"
        cmd = ["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
               "-fno-strict-aliasing", "-o", str(exe), str(harness), str(src), str(work / "dot.s")]
        build = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if build.returncode != 0:
            raise SystemExit(build.stderr)
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=180)
        if run.returncode != 0:
            raise SystemExit(run.stderr + run.stdout)
        model = pathlib.Path.home() / "models" / "Qwen3-8B-Base" / "model-00001-of-00005.safetensors"
        if not model.is_file():
            raw = {"status": "SKIPPED", "reason": "Qwen weights not present"}
        else:
            try:
                import numpy as np
                import torch
                from safetensors import safe_open
            except ModuleNotFoundError as exc:
                raw = {"status": "SKIPPED", "reason": "import failed: " + exc.name}
            else:
                with safe_open(str(model), framework="pt", device="cpu") as handle:
                    weight = handle.get_slice("model.layers.0.self_attn.q_proj.weight")[:64].float().contiguous().numpy()
                rows = quant_rows(weight)
                b = np.ascontiguousarray(np.stack(rows))
                rng = np.random.default_rng(20261029)
                a = np.ascontiguousarray(rng.integers(-128, 128, size=(16, b.shape[1]), dtype=np.int8))
                ap = work / "A.bin"
                bp = work / "B.bin"
                ap.write_bytes(a.tobytes())
                bp.write_bytes(b.tobytes())
                check = subprocess.run([str(exe), str(ap), str(bp), "16", str(b.shape[0]), str(b.shape[1])],
                                       capture_output=True, text=True, timeout=180)
                if check.returncode != 0:
                    raise SystemExit(check.stderr + check.stdout)
                raw = json.loads(check.stdout)
                raw["weight"] = "model.layers.0.self_attn.q_proj.weight[:64] per-row absmax int8"
                raw["activation"] = "16xK synthetic int8"
                raw["sha256_q"] = hashlib.sha256(b.tobytes()).hexdigest()
        report = {
            "kernel": "niyah_i8_gemm",
            "source": str(src),
            "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "oracle": "niyah_i8_dot from qwen_native_layer0.py plus scalar int32",
            "float_epilogue": "bit-identical to the current scale formula when int32 matches",
            "harness_stdout": run.stdout,
            "qwen_q_proj": raw,
            "limit": "RMSNorm, RoPE, attention, SiLU, residuals, activation quant, and the FP scale epilogue stay in C. This is not full-model inference.",
        }
    out = ROOT / "evidence" / "i8-gemm-blocked.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(run.stdout)
    print(json.dumps(raw))
    print("wrote", out)

if __name__ == "__main__":
    main()
