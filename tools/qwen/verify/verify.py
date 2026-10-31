#!/usr/bin/env python3
"""Independent layer-0 numerical verification; diagnostic hooks do not modify the original."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

HOME = Path("/home/a")
OUT = HOME / "qwen-layer0-quant-verify-v1"
ORIGINAL = HOME / "qwen_native_layer0_fixed_v2.py"
KERNEL = HOME / "niyah-assembly-model/src/niyah_i8_gemm.s"
MODEL = HOME / "models/Qwen3-8B-Base"
# Predeclared before tests. FP32 casts have ~1e-7 relative rounding;
# 3e-5 absolute + 2e-5 relative per element accommodates small sequential
# FP64 reduction/libm differences without accepting INT8-level differences.
ATOL = 3e-5
RTOL = 2e-5
REL_L2_LIMIT = 1e-5
CHUNK = 128
STAGES = [
    "input_norm", "q", "k", "v", "q_norm", "k_norm",
    "q_rope", "k_rope", "attention", "o_proj", "first_residual",
    "post_norm", "gate", "up", "silu_product", "down", "final_residual",
]
LINEARS = ("q", "k", "v", "o_proj", "gate", "up", "down")
FLOAT_TAGS = dict(zip(
    (1, 200, 201, 202, 5, 6, 7, 8, 9, 203, 11, 12, 204, 205, 15, 206, 17),
    STAGES,
))
LOG = OUT / "run.log"
REPORT = OUT / "report.json"

def emit(message):
    line = str(message)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def arr_sha(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()

def save_report(report):
    tmp = OUT / "report.json.writing"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, REPORT)

def exact_patch(src, old, new):
    count = src.count(old)
    if count != 1:
        raise RuntimeError(f"Diagnostic patch pattern count={count}: {old[:85]!r}")
    return src.replace(old, new, 1)

def diagnostic_source(source):
    # Inject observation-only hooks into a transient copy of the original C,
    # leaving ALL arithmetic, rounding, native kernels and source files intact.
    src = exact_patch(source, "#include <string.h>\n", """#include <string.h>
typedef void (*NIYAH_TRACE_F)(int,const float*,int,int);
typedef void (*NIYAH_TRACE_I)(int,const int32_t*,int,int);
static NIYAH_TRACE_F trace_f = NULL;
static NIYAH_TRACE_I trace_i = NULL;
static int linear_number = 0;
void niyah_set_trace(NIYAH_TRACE_F f, NIYAH_TRACE_I i){trace_f=f;trace_i=i;}
#define TRACE_F(tag,p,r,c) do{if(trace_f)trace_f(tag,p,r,c);}while(0)
#define TRACE_I(tag,p,r,c) do{if(trace_i)trace_i(tag,p,r,c);}while(0)
""")
    src = exact_patch(
        src,
        " if(niyah_i8_gemm(aq,m->w,acc,t,m->rows,m->cols))return 64;",
        " if(niyah_i8_gemm(aq,m->w,acc,t,m->rows,m->cols))return 64;\n TRACE_I(100+linear_number,acc,t,m->rows);",
    )
    src = exact_patch(
        src,
        " return 0;\n}\nstatic int norm(",
        " TRACE_F(200+linear_number,y,t,m->rows);\n ++linear_number;\n return 0;\n}\nstatic int norm(",
    )
    src = exact_patch(src, " size_t n=(size_t)t*4096", " linear_number=0;\n size_t n=(size_t)t*4096")
    src = exact_patch(src, " CHECK(norm(x,inorm,h,t,4096,eps));",
                      " CHECK(norm(x,inorm,h,t,4096,eps)); TRACE_F(1,h,t,4096);")
    src = exact_patch(src, " CHECK(norm(q,qn,q,t*32,128,eps));",
                      " CHECK(norm(q,qn,q,t*32,128,eps)); TRACE_F(5,q,t,4096);")
    src = exact_patch(src, "CHECK(norm(k,kn,k,t*8,128,eps));",
                      "CHECK(norm(k,kn,k,t*8,128,eps)); TRACE_F(6,k,t,1024);")
    src = exact_patch(src, " rope(q,cosine,sine,t,32);rope(k,cosine,sine,t,8);",
                      " rope(q,cosine,sine,t,32);rope(k,cosine,sine,t,8); TRACE_F(7,q,t,4096); TRACE_F(8,k,t,1024);")
    src = exact_patch(src, " CHECK(attention(q,k,v,a,t));",
                      " CHECK(attention(q,k,v,a,t)); TRACE_F(9,a,t,4096);")
    src = exact_patch(src, " for(size_t i=0;i<n;++i)res[i]=x[i]+p[i];",
                      " for(size_t i=0;i<n;++i)res[i]=x[i]+p[i]; TRACE_F(11,res,t,4096);")
    src = exact_patch(src, " CHECK(norm(res,pnorm,h,t,4096,eps));",
                      " CHECK(norm(res,pnorm,h,t,4096,eps)); TRACE_F(12,h,t,4096);")
    src = exact_patch(src, " CHECK(linear(z,p,m+6,t,aq,acc));",
                      " TRACE_F(15,z,t,12288); CHECK(linear(z,p,m+6,t,aq,acc));")
    src = exact_patch(src,
        " for(size_t i=0;i<n;++i){y[i]=res[i]+p[i];if(!isfinite(y[i])){rc=65;goto cleanup;}}\ncleanup:",
        " for(size_t i=0;i<n;++i){y[i]=res[i]+p[i];if(!isfinite(y[i])){rc=65;goto cleanup;}}\n TRACE_F(17,y,t,4096);\ncleanup:")
    return src

def quant_rows(data):
    """Independent FP64 max/divide/nearest-even/clip with stored FP32 scales."""
    x = np.asarray(data, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Quantization requires finite two-dimensional FP32")
    xd = x.astype(np.float64)
    maxabs = np.max(np.abs(xd), axis=1, keepdims=True)
    scale = np.where(maxabs == 0.0, 1.0, maxabs / 127.0).astype(np.float32)
    if not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Bad quantization scale")
    quantized = np.clip(np.rint(xd / scale.astype(np.float64)), -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized), np.ascontiguousarray(scale)

def fp32_rms(x, weight, eps):
    rows, cols = x.shape
    v = x.astype(np.float64)
    w = np.asarray(weight, dtype=np.float32).astype(np.float64).reshape(1, cols)
    square_sum = np.sum(v * v, axis=1, dtype=np.float64)
    inv = 1.0 / np.sqrt(square_sum / cols + float(np.float32(eps)))
    return np.ascontiguousarray((v * inv[:, None] * w).astype(np.float32))

def rope(x, cos, sin, heads):
    n = x.shape[0]
    p = np.asarray(x, dtype=np.float32).reshape(n, heads, 128)
    a = p[:, :, :64].copy()
    b = p[:, :, 64:].copy()
    c = cos[:, None, :64]
    s = sin[:, None, :64]
    result = np.empty_like(p)
    # Each multiply and add/subtract is explicitly FP32, like the native C.
    result[:, :, :64] = a * c - b * s
    result[:, :, 64:] = b * c + a * s
    return np.ascontiguousarray(result.reshape(n, heads * 128))

def causal_attention(q, k, v):
    t = len(q)
    q = q.reshape(t, 32, 128)
    k = k.reshape(t, 8, 128)
    v = v.reshape(t, 8, 128)
    out = np.empty((t, 32, 128), dtype=np.float32)
    factor = 1.0 / np.sqrt(128.0)
    for i in range(t):
        for h in range(32):
            key_idx = h // 4
            qd = q[i, h].astype(np.float64)
            kd = k[:i+1, key_idx].astype(np.float64)
            vd = v[:i+1, key_idx].astype(np.float64)
            scores = np.sum(kd * qd[None, :], axis=1, dtype=np.float64) * factor
            probs = np.exp(scores - scores.max())
            denom = probs.sum(dtype=np.float64)
            out[i, h] = (np.sum(probs[:, None] * vd, axis=0, dtype=np.float64) / denom).astype(np.float32)
    return np.ascontiguousarray(out.reshape(t, 4096))

def fp32_silu_product(g, u):
    a = g.astype(np.float64)
    sig = np.empty_like(a)
    mask = a >= 0
    sig[mask] = 1.0 / (1.0 + np.exp(-a[mask]))
    e = np.exp(a[~mask])
    sig[~mask] = e / (1.0 + e)
    return np.ascontiguousarray((a * sig * u.astype(np.float64)).astype(np.float32))

def int64_matmul_reference(activations, weights):
    # This independently computes exact integer products in NumPy.
    # A full int64 weight copy is avoided; each 128-row tile uses int64.
    x = np.asarray(activations, dtype=np.int8).astype(np.int64)
    out_rows = weights.shape[0]
    acc = np.empty((x.shape[0], out_rows), dtype=np.int64)
    for begin in range(0, out_rows, CHUNK):
        end = min(begin + CHUNK, out_rows)
        w = weights[begin:end].astype(np.int64)
        acc[:, begin:end] = x @ w.T
    if np.any(acc > np.iinfo(np.int32).max) or np.any(acc < np.iinfo(np.int32).min):
        raise OverflowError("int64 accumulation is not representable in int32")
    return acc

def matched_reference(x, cos, sin, matrices, norms, eps):
    states = {}
    products = {}
    w_in, w_post, w_q, w_k = norms

    def linear(key, a, index):
        q, activation_scale = quant_rows(a)
        w_int8, weight_scale = matrices[index]
        accumulation = int64_matmul_reference(q, w_int8)
        products[key] = accumulation
        # C expression: (float)((double)acc * (double)token_scale * (double)row_scale)
        y = accumulation.astype(np.float64) * activation_scale.astype(np.float64)
        y *= weight_scale.reshape(1, -1).astype(np.float64)
        states[key] = np.ascontiguousarray(y.astype(np.float32))
        return states[key]

    states["input_norm"] = fp32_rms(x, w_in, eps)
    q = linear("q", states["input_norm"], 0)
    k = linear("k", states["input_norm"], 1)
    v = linear("v", states["input_norm"], 2)
    states["q_norm"] = fp32_rms(q.reshape(-1, 128), w_q, eps).reshape(len(x), 4096)
    states["k_norm"] = fp32_rms(k.reshape(-1, 128), w_k, eps).reshape(len(x), 1024)
    states["q_rope"] = rope(states["q_norm"], cos, sin, 32)
    states["k_rope"] = rope(states["k_norm"], cos, sin, 8)
    states["attention"] = causal_attention(states["q_rope"], states["k_rope"], v)
    projected = linear("o_proj", states["attention"], 3)
    states["first_residual"] = np.ascontiguousarray((x + projected).astype(np.float32))
    states["post_norm"] = fp32_rms(states["first_residual"], w_post, eps)
    gate = linear("gate", states["post_norm"], 4)
    up = linear("up", states["post_norm"], 5)
    states["silu_product"] = fp32_silu_product(gate, up)
    down = linear("down", states["silu_product"], 6)
    states["final_residual"] = np.ascontiguousarray((states["first_residual"] + down).astype(np.float32))
    return states, products

def metrics(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} != {b.shape}")
    delta = a.astype(np.float64) - b.astype(np.float64)
    aa = np.abs(delta)
    bb = np.abs(b.astype(np.float64))
    denom = np.linalg.norm(b.astype(np.float64))
    return {
        "max_abs": float(aa.max(initial=0)),
        "relative_l2": float(np.linalg.norm(delta) / denom) if denom != 0 else None,
        "failed_elements": int(np.count_nonzero(aa > ATOL + RTOL * bb)),
        "elements": int(a.size),
    }

def first_failure(comparison):
    for name in STAGES:
        if comparison[name]["failed_elements"]:
            return name
    return None

def fp32_transformers_trace(layer, x, cos, sin, eps):
    captured = {}
    modules = dict(layer.named_modules())
    specs = [
        ("input_layernorm", "input_norm", False),
        ("self_attn.q_proj", "q", False),
        ("self_attn.k_proj", "k", False),
        ("self_attn.v_proj", "v", False),
        ("self_attn.q_norm", "q_norm", False),
        ("self_attn.k_norm", "k_norm", False),
        ("self_attn.o_proj", "attention", True),
        ("self_attn.o_proj", "o_proj", False),
        ("post_attention_layernorm", "first_residual", True),
        ("post_attention_layernorm", "post_norm", False),
        ("mlp.gate_proj", "gate", False),
        ("mlp.up_proj", "up", False),
        ("mlp.down_proj", "silu_product", True),
        ("mlp.down_proj", "down", False),
    ]
    handles = []
    try:
        for module_name, stage, take_input in specs:
            if module_name not in modules:
                raise RuntimeError(f"Missing FP32 hook target {module_name}")
            def callback(mod, args, output, stage=stage, take_input=take_input):
                value = args[0] if take_input else output
                captured[stage] = np.ascontiguousarray(
                    value.detach().cpu().numpy().reshape(len(x), -1), dtype=np.float32
                ).copy()
            handles.append(modules[module_name].register_forward_hook(callback))
        torch_x = torch.from_numpy(x).unsqueeze(0)
        n = len(x)
        mask = torch.full((1, 1, n, n), float("-inf"), dtype=torch.float32).triu(1)
        positions = torch.arange(n, dtype=torch.long).unsqueeze(0)
        with torch.inference_mode():
            result = layer(
                torch_x, attention_mask=mask, position_ids=positions,
                position_embeddings=(torch.from_numpy(cos).unsqueeze(0),
                                     torch.from_numpy(sin).unsqueeze(0)),
                use_cache=False,
            )
            if isinstance(result, tuple):
                result = result[0]
        captured["final_residual"] = np.ascontiguousarray(result[0].numpy().reshape(n, 4096)).copy()
        # RoPE is applied inside attention, not a torch.nn.Module;
        # reconstruct it from the actual FP32 q_norm/k_norm hook values,
        # with precisely the shared position cos/sin tensors.
        captured["q_rope"] = rope(captured["q_norm"], cos, sin, 32)
        captured["k_rope"] = rope(captured["k_norm"], cos, sin, 8)
        missing = set(STAGES) - set(captured)
        if missing:
            raise RuntimeError(f"FP32 missing intermediates: {missing}")
        return captured
    finally:
        for handle in handles:
            handle.remove()

class Matrix(ctypes.Structure):
    _fields_ = [("w", ctypes.c_void_p), ("s", ctypes.c_void_p),
                ("rows", ctypes.c_int), ("cols", ctypes.c_int)]

def compile_native(original):
    gemm_source = KERNEL.read_text()
    src = diagnostic_source(original.C_SOURCE)
    asm = original.ASM + gemm_source.replace('.section .note.GNU-stack, "", @progbits\n', '')
    work = tempfile.TemporaryDirectory(prefix="native-build-", dir=OUT)
    dest = Path(work.name)
    (dest / "layer.c").write_text(src)
    (dest / "kernels.s").write_text(asm)
    subprocess.run([
        "gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
        "-fno-fast-math", "-ffp-contract=off", "-Wl,-z,defs,-z,noexecstack",
        "-o", str(dest / "layer.so"), str(dest / "layer.c"),
        str(dest / "kernels.s"), "-lm",
    ], check=True, timeout=90)
    lib = ctypes.CDLL(str(dest / "layer.so"))
    lib.available.argtypes = []
    lib.available.restype = ctypes.c_int
    if not lib.available():
        raise RuntimeError("AVX2 is not available")
    return work, lib

def native_int64_kernel_check(lib):
    gemm = lib.niyah_i8_gemm
    gemm.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    gemm.restype = ctypes.c_int
    gen = np.random.default_rng(20261029)
    cases = [(1, 1, 16), (3, 5, 32), (7, 6, 48), (8, 4, 64),
             (9, 6, 80), (16, 17, 128)]
    for m, n, k in cases:
        a = np.ascontiguousarray(gen.integers(-128, 128, (m, k), dtype=np.int8))
        b = np.ascontiguousarray(gen.integers(-128, 128, (n, k), dtype=np.int8))
        c = np.empty((m, n), dtype=np.int32)
        rc = gemm(a.ctypes.data, b.ctypes.data, c.ctypes.data, m, n, k)
        if rc:
            raise RuntimeError(f"native standalone GEMM rejected {(m,n,k)}: {rc}")
        expected = int64_matmul_reference(a, b)
        if not np.array_equal(c.astype(np.int64), expected):
            raise AssertionError(f"integer discrepancy in native GEMM {(m,n,k)}")
    emit(f"Standalone GEMM vs independent INT64: {len(cases)} cases PASS")
    return len(cases)

def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    report = {
        "status": "RUNNING", "implementation_gate": "NOT_ESTABLISHED",
        "quantization_quality_gate": "NOT_ESTABLISHED",
        "scope": "Qwen3-8B-Base decoder layer 0 only",
        "tolerances_predeclared": {
            "float_atol": ATOL, "float_rtol": RTOL,
            "relative_l2_limit": REL_L2_LIMIT,
            "rationale": "FP32 storage/casts plus FP64 reductions and libm rounding; exact integer products have zero tolerance. No FP32 quantization-quality PASS threshold is assumed.",
        },
        "inputs": {
            "original_script": str(ORIGINAL), "original_script_sha256": sha(ORIGINAL),
            "kernel": str(KERNEL), "kernel_sha256": sha(KERNEL),
            "model": str(MODEL),
            "mode": "reuse same x, weights, positions and RoPE for native, NumPy and Transformers",
            "variants": "original 16-token embedding; deterministic altered embedding",
        },
        "comparisons": [],
        "limitations": [
            "Only one decoder layer, no full 36-layer runtime", "No KV cache",
            "No text generation", "Runtime includes copied C for normalization/attention and Assembly for INT8 products",
            "Transformer FP32 is the reference for quality measurement, not a quantization-matched correctness oracle",
        ],
    }
    save_report(report)
    work = None
    try:
        spec = importlib.util.spec_from_file_location("existing_layer0_source", ORIGINAL)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        cfg = AutoConfig.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
        expected = (4096, 12288, 32, 8, 128, 36)
        actual = (cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads,
                  cfg.num_key_value_heads, cfg.head_dim, cfg.num_hidden_layers)
        if actual != expected or cfg.hidden_act != "silu" or cfg.attention_bias:
            raise RuntimeError(f"Unsupported model geometry: {actual}")
        if getattr(cfg, "layer_types", None) and cfg.layer_types[0] != "full_attention":
            raise RuntimeError("Layer 0 is not full attention")
        cfg._attn_implementation = "eager"
        mapping_path = MODEL / "model.safetensors.index.json"
        mapping = json.loads(mapping_path.read_text())["weight_map"]
        def load(name, selection=slice(None)):
            path = MODEL / mapping[name]
            with safe_open(str(path), framework="pt", device="cpu") as f:
                return f.get_slice(name)[selection].float()
        work, lib = compile_native(original)
        count = native_int64_kernel_check(lib)
        report["standalone_gemm_int64_checks"] = count
        save_report(report)

        with torch.device("meta"):
            layer = Qwen3DecoderLayer(cfg, layer_idx=0)
        state = {}
        for name in layer.state_dict():
            state[name] = load("model.layers.0." + name)
            emit("loaded " + name)
        layer.load_state_dict(state, strict=True, assign=True)
        layer.eval()
        if any(param.is_meta for param in layer.parameters()):
            raise RuntimeError("Unloaded FP32 model weights")

        names = [
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        ]
        packed = []
        for name in names:
            w = state[name].numpy()
            q = np.empty(w.shape, dtype=np.int8)
            s = np.empty((len(w), 1), dtype=np.float32)
            for begin in range(0, len(w), CHUNK):
                end = min(begin + CHUNK, len(w))
                original_q, original_s = original.quant(w[begin:end])
                independent_q, independent_s = quant_rows(w[begin:end])
                if not np.array_equal(independent_q, original_q):
                    raise AssertionError(f"Weight integer quantization mismatch {name} row {begin}")
                if not np.array_equal(independent_s, original_s):
                    raise AssertionError(f"Weight FP32 scale mismatch {name} row {begin}")
                q[begin:end] = original_q
                s[begin:end] = original_s
            packed.append((q, s))
            emit(f"quantize/independent cross-check PASS: {name} {q.shape}")
        norm_names = ("input_layernorm.weight", "post_attention_layernorm.weight",
                      "self_attn.q_norm.weight", "self_attn.k_norm.weight")
        norms = [np.ascontiguousarray(state[name].numpy()) for name in norm_names]
        hashes_before = [arr_sha(a) for pair in packed for a in pair] + [arr_sha(w) for w in norms]
        matrices = (Matrix * 7)(*[Matrix(q.ctypes.data, s.ctypes.data, *q.shape)
                                 for q, s in packed])

        native_fn = lib.niyah_layer0
        native_fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int, ctypes.c_float]
        native_fn.restype = ctypes.c_int
        fcb_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                    ctypes.c_int, ctypes.c_int)
        icb_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
                                    ctypes.c_int, ctypes.c_int)
        set_trace = lib.niyah_set_trace
        set_trace.argtypes = [fcb_type, icb_type]
        set_trace.restype = None

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True,
                                                   trust_remote_code=False)
        prompt = "نريد تنفيذ حساب حقيقي من نموذج كوين باستخدام الأوزان الأصلية ومقارنة نتيجة الاسمبلي بمرجع مستقل قبل دمجها في التشغيل الكامل."
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:16]
        if len(ids) != 16:
            raise RuntimeError("Fewer than 16 reference tokens")
        report["input_token_ids"] = ids
        original_embedding = np.ascontiguousarray(
            torch.cat([load("model.embed_tokens.weight", slice(i, i+1)) for i in ids]).numpy()
        )
        variants = {
            "embedding": original_embedding,
            "deterministic_perturbation": np.ascontiguousarray(
                (original_embedding.astype(np.float64) * 0.93
                 + 0.015 * np.sin(np.arange(original_embedding.size, dtype=np.float64).reshape(original_embedding.shape) * 0.017)).astype(np.float32)
            ),
        }
        rotary = Qwen3RotaryEmbedding(config=cfg)
        for case_name, case_base in variants.items():
            for t in (1, 7, 16):
                started = time.perf_counter()
                x = np.ascontiguousarray(case_base[:t])
                x_hash_before = arr_sha(x)
                pos = torch.arange(t, dtype=torch.long).unsqueeze(0)
                with torch.inference_mode():
                    cos_t, sin_t = rotary(torch.from_numpy(x).unsqueeze(0), pos)
                cos = np.ascontiguousarray(cos_t[0].numpy())
                sin = np.ascontiguousarray(sin_t[0].numpy())
                cos_hash = arr_sha(cos)
                sin_hash = arr_sha(sin)
                if cos.shape != (t, 128) or sin.shape != (t, 128):
                    raise RuntimeError("Unexpected rotary shape")
                if not np.array_equal(cos[:, :64], cos[:, 64:]) or not np.array_equal(sin[:, :64], sin[:, 64:]):
                    raise RuntimeError("Unexpected rotary split")

                native_float, native_int, trace_errors = {}, {}, []
                @fcb_type
                def trace_float(tag, data, rows, cols):
                    try:
                        name = FLOAT_TAGS[tag]
                        native_float[name] = np.ctypeslib.as_array(
                            data, shape=(rows * cols,)
                        ).reshape(rows, cols).copy()
                    except Exception as ex:
                        trace_errors.append(str(ex))
                @icb_type
                def trace_integer(tag, data, rows, cols):
                    try:
                        name = LINEARS[tag - 100]
                        native_int[name] = np.ctypeslib.as_array(
                            data, shape=(rows * cols,)
                        ).reshape(rows, cols).copy()
                    except Exception as ex:
                        trace_errors.append(str(ex))
                set_trace(trace_float, trace_integer)
                guard = np.full(x.size + 32, np.nan, np.float32)
                native_output = guard[16:-16].reshape(t, 4096)
                def run_native(inp, dest=native_output):
                    args = [inp.ctypes.data, dest.ctypes.data, ctypes.addressof(matrices)]
                    args += [n.ctypes.data for n in norms]
                    args += [cos.ctypes.data, sin.ctypes.data]
                    rc = native_fn(*args, t, cfg.rms_norm_eps)
                    if rc != 0:
                        raise RuntimeError(f"Native layer returned {rc}")

                run_native(x)
                if trace_errors:
                    raise RuntimeError(f"Trace callback errors: {trace_errors}")
                if not np.isnan(guard[:16]).all() or not np.isnan(guard[-16:]).all():
                    raise AssertionError("Native output canary overwritten")
                if not np.isfinite(native_output).all():
                    raise AssertionError("Native nonfinite output")
                if set(native_float) != set(STAGES) or set(native_int) != set(LINEARS):
                    raise AssertionError(f"Incomplete native trace: {set(STAGES)-set(native_float)}, {set(LINEARS)-set(native_int)}")
                states, exact_acc = matched_reference(x, cos, sin, packed, norms, cfg.rms_norm_eps)
                fp32_states = fp32_transformers_trace(layer, x, cos, sin, cfg.rms_norm_eps)
                for name in LINEARS:
                    expected64 = exact_acc[name]
                    actual32 = native_int[name]
                    if not np.array_equal(expected64, actual32.astype(np.int64)):
                        idx = tuple(np.argwhere(expected64 != actual32.astype(np.int64))[0])
                        raise AssertionError(f"INT64 != native INT32 at {case_name}/{t}/{name}/{idx}: {int(expected64[idx])} vs {int(actual32[idx])}")
                a = {key: metrics(native_float[key], states[key]) for key in STAGES}
                b = {key: metrics(states[key], fp32_states[key]) for key in STAGES}
                fail_a = first_failure(a)
                fail_b = first_failure(b)
                native_l2 = a["final_residual"]["relative_l2"]
                a_ok = fail_a is None and native_l2 is not None and native_l2 <= REL_L2_LIMIT
                row = {
                    "case": case_name, "tokens": t,
                    "A_native_vs_quantmatched": {
                        "status": "PASS" if a_ok else "FAIL",
                        "first_divergent_intermediate": fail_a,
                        "int32_products_exact_vs_numpy_int64": True,
                        "metrics_by_intermediate": a,
                    },
                    "B_quantmatched_vs_transformers_fp32": {
                        "quality_gate": "NOT_ESTABLISHED",
                        "first_divergent_intermediate_at_numeric_tolerances": fail_b,
                        "metrics_by_intermediate": b,
                    },
                    "immutable_input": arr_sha(x) == x_hash_before,
                    "immutable_rope": arr_sha(cos) == cos_hash and arr_sha(sin) == sin_hash,
                    "native_canary_and_finite": True,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
                if t == 16:
                    # Causal independence: changing the last token must not affect earlier ones.
                    perturbed = x.copy()
                    perturbed[-1] += np.float32(0.05)
                    native_float.clear()
                    native_int.clear()
                    future_native = np.empty_like(x)
                    run_native(perturbed, future_native)
                    future_numpy, _ = matched_reference(perturbed, cos, sin, packed, norms, cfg.rms_norm_eps)
                    row["native_future_token_isolation"] = bool(np.array_equal(future_native[:-1], native_output[:-1]))
                    row["numpy_future_token_isolation"] = bool(np.array_equal(future_numpy["final_residual"][:-1], states["final_residual"][:-1]))
                    if not row["native_future_token_isolation"] or not row["numpy_future_token_isolation"]:
                        a_ok = False
                        row["A_native_vs_quantmatched"]["status"] = "FAIL"
                    if arr_sha(x) != x_hash_before:
                        raise AssertionError("Original input mutated during causal check")
                report["comparisons"].append(row)
                save_report(report)
                emit(f"case={case_name} tokens={t} A={row['A_native_vs_quantmatched']['status']} "
                     f"first_A={fail_a} first_B={fail_b} "
                     f"A_max={a['final_residual']['max_abs']:.9g} A_relL2={native_l2:.9g} "
                     f"B_max={b['final_residual']['max_abs']:.9g} "
                     f"B_relL2={b['final_residual']['relative_l2']:.9g}")
                if not row["immutable_input"] or not row["immutable_rope"]:
                    raise AssertionError("Inputs or RoPE mutated")
        hashes_after = [arr_sha(a) for pair in packed for a in pair] + [arr_sha(w) for w in norms]
        report["weight_and_scale_immutability"] = "PASS" if hashes_after == hashes_before else "FAIL"
        if hashes_after != hashes_before:
            raise AssertionError("Weights/scales modified")
        report["implementation_gate"] = (
            "PASS" if all(r["A_native_vs_quantmatched"]["status"] == "PASS"
                          for r in report["comparisons"]) else "FAIL"
        )
        report["status"] = "COMPLETED"
        emit(f"END: {report['implementation_gate']} / quantization quality NOT_ESTABLISHED")
    except BaseException as err:
        report["status"] = "FAILED"
        report["implementation_gate"] = "FAIL"
        report["error"] = f"{type(err).__name__}: {err}"
        report["traceback"] = traceback.format_exc()
        emit(report["error"])
        emit(report["traceback"])
    finally:
        if work is not None:
            work.cleanup()
        report["artifact_hashes"] = {
            "new_script_sha256": sha(Path(__file__)),
            "original_script_sha256_after": sha(ORIGINAL),
            "kernel_sha256_after": sha(KERNEL),
        }
        report["preservation_ok"] = (
            report["artifact_hashes"]["original_script_sha256_after"] ==
            report["inputs"]["original_script_sha256"] and
            report["artifact_hashes"]["kernel_sha256_after"] ==
            report["inputs"]["kernel_sha256"]
        )
        save_report(report)
    return 0 if report["status"] == "COMPLETED" and report["implementation_gate"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
