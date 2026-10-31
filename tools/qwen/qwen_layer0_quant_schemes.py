#!/usr/bin/env python3
# qwen_layer0_quant_schemes.py — Phase B measurement. Read-only: changes no repo file, no C, no weights.
# Fake-quant (quantize -> dequantize) each of the 7 Linear layers of Qwen3-8B layer 0 under several
# schemes and report rel L2 vs transformers FP32 on the same 16-token Arabic prompt used by
# qwen_native_layer0.py. Goal: find a scheme with rel L2 <= 0.02 BEFORE changing the native kernel.
import json, sys, hashlib
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

LINEARS = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
           'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
TEXT = 'نريد تنفيذ حساب حقيقي من نموذج كوين باستخدام الأوزان الأصلية ومقارنة نتيجة الاسمبلي بمرجع مستقل قبل دمجها في التشغيل الكامل.'

def fq(x, group):
    """Symmetric int8 fake-quant along the last dim. group=None -> one scale per row; else per group of `group`."""
    x = x.double()
    if group is None:
        m = x.abs().amax(-1, keepdim=True)
        s = torch.where(m > 0, m / 127.0, torch.ones_like(m))
        return (torch.clamp(torch.round(x / s), -127, 127) * s).float()
    *lead, k = x.shape
    if k % group: raise SystemExit(f'K={k} not divisible by group {group}')
    g = x.reshape(*lead, k // group, group)
    m = g.abs().amax(-1, keepdim=True)
    s = torch.where(m > 0, m / 127.0, torch.ones_like(m))
    return (torch.clamp(torch.round(g / s), -127, 127) * s).reshape(*lead, k).float()

# scheme: (weight_group, act_group, set of linears kept FP32). 'fp32' = not quantized.
SCHEMES = {
    'A_current_w_row__a_token':        ('row', 'row', set()),
    'B_w_row__a_fp32':                 ('row', 'fp32', set()),
    'C_w_fp32__a_token':               ('fp32', 'row', set()),
    'D_w_g128__a_token':               (128, 'row', set()),
    'E_w_g128__a_g128':                (128, 128, set()),
    'F_w_g64__a_g64':                  (64, 64, set()),
    'G_current__down_fp32':            ('row', 'row', {'mlp.down_proj'}),
    'H_current__q_k_fp32':             ('row', 'row', {'self_attn.q_proj', 'self_attn.k_proj'}),
    'I_current__down_a_g128':          ('row', 'row', set(), {'mlp.down_proj': 128}),
    'J_current__down_a_g64':           ('row', 'row', set(), {'mlp.down_proj': 64}),
    'K_current__down_a_g32':           ('row', 'row', set(), {'mlp.down_proj': 32}),
}

def main():
    torch.set_num_threads(1)
    model = Path.home() / 'models/Qwen3-8B-Base'
    cfg = AutoConfig.from_pretrained(str(model), local_files_only=True)
    cfg._attn_implementation = 'eager'
    mapping = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    def read(name, sel=slice(None)):
        with safe_open(str(model / mapping[name]), framework='pt', device='cpu') as f:
            return f.get_slice(name)[sel].float()
    with torch.device('meta'):
        layer = Qwen3DecoderLayer(cfg, layer_idx=0)
    state = {k: read('model.layers.0.' + k) for k in layer.state_dict()}
    layer.load_state_dict(state, strict=True, assign=True); layer.eval()
    orig_w = {n: state[n + '.weight'].clone() for n in LINEARS}
    tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    ids = tok(TEXT, add_special_tokens=False)['input_ids'][:16]
    assert len(ids) == 16
    emb = torch.cat([read('model.embed_tokens.weight', slice(i, i + 1)) for i in ids])
    rotary = Qwen3RotaryEmbedding(config=cfg)
    mods = {n: layer.get_submodule(n) for n in LINEARS}

    wcache = {}
    def run(t, scheme=None):
        x = emb[:t].unsqueeze(0)
        co, si = rotary(x, torch.arange(t).unsqueeze(0))
        mask = torch.full((1, 1, t, t), float('-inf')).triu(1)
        hooks = []
        if scheme:
            wg, ag, keep, *rest = scheme
            ov = rest[0] if rest else {}
            for n, m in mods.items():
                if n in keep:
                    m.weight.data = orig_w[n]; continue
                if (n, wg) not in wcache:
                    if any(k[1] != wg for k in wcache): wcache.clear()
                    wcache[(n, wg)] = orig_w[n] if wg == 'fp32' else fq(orig_w[n], None if wg == 'row' else wg)
                m.weight.data = wcache[(n, wg)]
                a = ov.get(n, ag)
                if a != 'fp32':
                    g = None if a == 'row' else a
                    hooks.append(m.register_forward_pre_hook(lambda mod, inp, g=g: (fq(inp[0], g),)))
        else:
            for n, m in mods.items(): m.weight.data = orig_w[n]
        try:
            with torch.inference_mode():
                out = layer(x, attention_mask=mask, position_ids=torch.arange(t).unsqueeze(0),
                            position_embeddings=(co, si), use_cache=False)
        finally:
            for h in hooks: h.remove()
        out = out[0] if isinstance(out, tuple) else out
        return out[0].double()

    report = {'scope': 'layer 0 fake-quant schemes vs transformers FP32', 'input_token_ids': ids,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'results': {}}
    refs = {t: run(t) for t in (1, 7, 16)}
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    for name, sch in SCHEMES.items():
        if only and name[0] not in only: continue
        rows = []
        for t in (1, 7, 16):
            d = run(t, sch) - refs[t]
            rel = float(d.norm() / refs[t].norm())
            idx = int(d.abs().argmax()); tk, ch = divmod(idx, d.shape[-1])
            rows.append({'tokens': t, 'rel_l2': rel, 'max_abs': float(d.abs().max()), 'worst_token': tk, 'worst_channel': ch})
        worst = max(r['rel_l2'] for r in rows)
        report['results'][name] = {'rows': rows, 'max_rel_l2': worst, 'gate_0.02': 'PASS' if worst <= 0.02 else 'FAIL'}
        print(f"{name:34s} rel_l2 t1={rows[0]['rel_l2']:.4f} t7={rows[1]['rel_l2']:.4f} t16={rows[2]['rel_l2']:.4f}  "
              f"worst=({rows[2]['worst_token']},{rows[2]['worst_channel']})  {report['results'][name]['gate_0.02']}", flush=True)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('quant_schemes_report.json')
    out.write_text(json.dumps(report, indent=1))
    print('report:', out, hashlib.sha256(out.read_bytes()).hexdigest())

if __name__ == '__main__':
    main()
