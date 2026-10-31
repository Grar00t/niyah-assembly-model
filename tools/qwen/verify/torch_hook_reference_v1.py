#!/usr/bin/env python3
"""Independent Torch fake-quantized layer0, with per-Linear forward_pre_hooks."""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import traceback
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

HOME = Path("/home/a")
OUT = HOME / "qwen-layer0-quant-verify-v1"
MODEL = HOME / "models/Qwen3-8B-Base"
SRC = HOME / "qwen_native_layer0_fixed_v2.py"
KERNEL = HOME / "niyah-assembly-model/src/niyah_i8_gemm.s"
DEST = OUT / "report_torch_hooks_v1.json"
LOG = OUT / "torch_hooks_v1.log"
if DEST.exists() or LOG.exists():
    raise SystemExit("STOP: evidence file exists; won't overwrite")

spec = importlib.util.spec_from_file_location("numpy_reference", OUT / "verify.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
original_diagnostic = base.diagnostic_source
def fixed_diagnostic(source):
    text = original_diagnostic(source)
    old = " for(size_t i=0;i<n;++i)res[i]=x[i]+p[i]; TRACE_F(11,res,t,4096);"
    new = " for(size_t i=0;i<n;++i){res[i]=x[i]+p[i];}\n TRACE_F(11,res,t,4096);"
    assert text.count(old) == 1
    return text.replace(old, new, 1)
base.diagnostic_source = fixed_diagnostic

spec2 = importlib.util.spec_from_file_location("original_layer0_source", SRC)
orig = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(orig)

def emit(msg):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg+"\n")

def rel_l2(a, b):
    norm = np.linalg.norm(b.astype(np.float64))
    return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/norm)

def sha(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()

def quantize_rowwise(w):
    q = np.empty(w.shape, np.int8)
    s = np.empty((w.shape[0], 1), np.float32)
    for begin in range(0, len(w), 128):
        end = min(len(w), begin + 128)
        q[begin:end], s[begin:end] = base.quant_rows(w[begin:end])
    return q, s

torch.set_num_threads(1)
report = {
    "scope": "Single Qwen3-8B-Base decoder layer, 1/7/16 tokens",
    "reference": "Actual torch.nn.Linear layers with weights per-output-row fake-quantized and 7 forward_pre_hooks quantizing each activation row with np.rint round-half-even, clamp to +/-127, stored FP32 scales",
    "original_script_sha256": base.sha(SRC),
    "kernel_sha256": base.sha(KERNEL),
    "cases": [],
    "quality_gate": "NOT_ESTABLISHED",
    "native_vs_fakequant_gate": "NOT_ESTABLISHED",
    "limit_relative_l2": 1e-4,
}
work = None
saved_weights = {}
handles = []
packed = []
norms = []
try:
    config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    config._attn_implementation = "eager"
    if (config.hidden_size, config.intermediate_size, config.num_hidden_layers,
        config.num_attention_heads, config.num_key_value_heads, config.head_dim) != (
            4096, 12288, 36, 32, 8, 128):
        raise RuntimeError("Unexpected architecture")
    index = json.loads((MODEL/"model.safetensors.index.json").read_text())["weight_map"]
    def load(name, selection=slice(None)):
        with safe_open(str(MODEL/index[name]), framework="pt", device="cpu") as f:
            return f.get_slice(name)[selection].float()

    with torch.device("meta"):
        layer = Qwen3DecoderLayer(config, layer_idx=0)
    state = {k: load("model.layers.0."+k) for k in layer.state_dict()}
    layer.load_state_dict(state, assign=True, strict=True)
    layer.eval()
    targets = {
        "q": "self_attn.q_proj",
        "k": "self_attn.k_proj",
        "v": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "gate": "mlp.gate_proj",
        "up": "mlp.up_proj",
        "down": "mlp.down_proj",
    }
    modules = dict(layer.named_modules())
    wkeys = [name + ".weight" for name in targets.values()]
    for key in wkeys:
        q, s = quantize_rowwise(state[key].numpy())
        orig_q = np.empty_like(q)
        orig_s = np.empty_like(s)
        w = state[key].numpy()
        for a in range(0, len(q), 128):
            z = min(a+128, len(q))
            orig_q[a:z], orig_s[a:z] = orig.quant(w[a:z])
        if not np.array_equal(q, orig_q) or not np.array_equal(s, orig_s):
            raise AssertionError(f"Quantization mismatch for {key}")
        packed.append((np.ascontiguousarray(q), np.ascontiguousarray(s)))
        emit(f"weight quant match: {key}, {q.shape}")
    original_weight_hashes = [sha(state[k].numpy()) for k in wkeys]
    norms = [np.ascontiguousarray(state[k].numpy()) for k in (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight")]
    matrices = (base.Matrix * 7)(*[base.Matrix(q.ctypes.data, s.ctypes.data, *q.shape) for q,s in packed])
    work, lib = base.compile_native(orig)
    fn = lib.niyah_layer0
    fn.argtypes = [ctypes.c_void_p]*9 + [ctypes.c_int, ctypes.c_float]
    fn.restype = ctypes.c_int
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True,
                                               trust_remote_code=False)
    prompt = "نريد تنفيذ حساب حقيقي من نموذج كوين باستخدام الأوزان الأصلية ومقارنة نتيجة الاسمبلي بمرجع مستقل قبل دمجها في التشغيل الكامل."
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:16]
    if len(ids) != 16: raise AssertionError("Insufficient tokens")
    inputs = np.ascontiguousarray(
        torch.cat([load("model.embed_tokens.weight", slice(i,i+1)) for i in ids]).numpy())
    rotary = Qwen3RotaryEmbedding(config=config)
    report["tokens_used"] = ids
    fp32 = {}
    positional = {}
    for t in (1,7,16):
        x = np.ascontiguousarray(inputs[:t])
        pos = torch.arange(t).unsqueeze(0)
        with torch.inference_mode():
            co, si = rotary(torch.from_numpy(x).unsqueeze(0),pos)
        cos = np.ascontiguousarray(co[0].numpy())
        sin = np.ascontiguousarray(si[0].numpy())
        positional[t] = (cos,sin)
        fp32[t] = base.fp32_transformers_trace(layer,x,cos,sin,config.rms_norm_eps)["final_residual"]
    # Swap ONLY in-memory torch parameters; original source and saved weights remain intact.
    hook_count = {n:0 for n in targets}
    for name, module_path in targets.items():
        obj = modules[module_path]
        saved_weights[name] = obj.weight
        q,s = packed[list(targets).index(name)]
        w_fake = (q.astype(np.float64)*s.astype(np.float64)).astype(np.float32)
        obj.weight = torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(w_fake)), requires_grad=False)
        def prehook(mod, args, name=name):
            inp = args[0]
            shape = inp.shape
            flat = np.ascontiguousarray(inp.detach().cpu().numpy().reshape(-1,shape[-1]),np.float32)
            a, scale = base.quant_rows(flat)
            # Dequantize in FP32 from the same INT8 and stored-FP32 scale contract.
            deq = (a.astype(np.float64) * scale.astype(np.float64)).astype(np.float32)
            hook_count[name] += 1
            return (torch.from_numpy(np.ascontiguousarray(deq).reshape(shape)),) + args[1:]
        handles.append(obj.register_forward_pre_hook(prehook))
    for t in (1,7,16):
        x = np.ascontiguousarray(inputs[:t])
        cos, sin = positional[t]
        after_quant = base.fp32_transformers_trace(layer,x,cos,sin,config.rms_norm_eps)["final_residual"]
        before = {k:hook_count[k] for k in targets}
        if not all(before[k]==[1,7,16].index(t)+1 for k in targets):
            raise AssertionError(f"Missing fakequant Linear pre-hooks for t={t}: {before}")
        out = np.empty_like(x)
        args = [x.ctypes.data,out.ctypes.data,ctypes.addressof(matrices)]
        args += [n.ctypes.data for n in norms]
        args += [cos.ctypes.data,sin.ctypes.data]
        rc = fn(*args,t,config.rms_norm_eps)
        if rc: raise RuntimeError(f"native returned {rc}")
        reference = fp32[t]
        diff = np.abs(out.astype(np.float64)-reference.astype(np.float64))
        index = tuple(int(i) for i in np.unravel_index(diff.argmax(),diff.shape))
        row = {
            "tokens":t,
            "native_vs_torch_hook_fakequant_rel_l2":rel_l2(out,after_quant),
            "torch_hook_fakequant_vs_fp32_rel_l2":rel_l2(after_quant,reference),
            "native_vs_fp32_rel_l2":rel_l2(out,reference),
            "worst_token_index":index[0],
            "worst_channel_index":index[1],
            "native_at_worst":float(out[index]),
            "torch_fakequant_at_worst":float(after_quant[index]),
            "fp32_at_worst":float(reference[index]),
            "max_abs_fp32":float(np.max(np.abs(reference))),
            "seven_pre_hooks_ran":all(before.values()),
            "max_abs_native_vs_torch_fakequant":float(np.max(np.abs(out.astype(np.float64)-after_quant.astype(np.float64)))),
        }
        report["cases"].append(row)
        emit(json.dumps(row))
    # Check in-memory original weights still identical, plus on-disk source hashes.
    report["original_weights_unchanged"] = all(sha(state[k].numpy())==h
        for k,h in zip(wkeys,original_weight_hashes))
    report["original_source_unchanged"] = base.sha(SRC)==report["original_script_sha256"]
    report["original_kernel_unchanged"] = base.sha(KERNEL)==report["kernel_sha256"]
    report["native_vs_fakequant_gate"] = "PASS" if (
        all(row["native_vs_torch_hook_fakequant_rel_l2"]<=1e-4 for row in report["cases"])
        and report["original_weights_unchanged"]
        and report["original_source_unchanged"] and report["original_kernel_unchanged"]
    ) else "FAIL"
    report["status"] = "COMPLETED"
except BaseException as e:
    report["status"] = "FAILED"
    report["error"] = repr(e)
    report["traceback"] = traceback.format_exc()
    emit(report["traceback"])
finally:
    for h in handles:
        h.remove()
    for name, module_path in targets.items() if "targets" in globals() else []:
        if name in saved_weights:
            modules[module_path].weight = saved_weights[name]
    if work is not None: work.cleanup()
    report["script_sha256"] = base.sha(Path(__file__))
    with DEST.open("x", encoding="utf-8") as f:
        json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
emit(f"END status={report['status']} gate={report['native_vs_fakequant_gate']}")
raise SystemExit(0 if report["status"]=="COMPLETED" and
                 report["native_vs_fakequant_gate"]=="PASS" else 1)
