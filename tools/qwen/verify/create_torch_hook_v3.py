#!/usr/bin/env python3
from pathlib import Path
root = Path("/home/a/qwen-layer0-quant-verify-v1")
source = (root/"torch_hook_reference_v2.py").read_text()
dest = root/"torch_hook_reference_v3.py"
source = source.replace("report_torch_hooks_v2.json","report_torch_hooks_v3.json").replace("torch_hooks_v2.log","torch_hooks_v3.log")
def change(old, new):
    global source
    if source.count(old) != 1: raise RuntimeError("Not unique: "+old[:90])
    source = source.replace(old, new, 1)
change('    hook_count = {n:0 for n in targets}', '''    hook_count = {n:0 for n in targets}
    pre_hook_states = {}''')
change('            a, scale = base.quant_rows(flat)\n', '''            a, scale = base.quant_rows(flat)
            pre_hook_states[name] = (flat.copy(), a.copy(), scale.copy())
''')
change('        divergent = next((name for name in base.STAGES if stage_metrics[name]["failed_elements"]), None)',
'''        divergent = next((name for name in base.STAGES if stage_metrics[name]["failed_elements"]), None)
        true_divergent = next((name for name in base.STAGES if name not in
            ("attention","silu_product") and stage_metrics[name]["failed_elements"]), None)
        stage_for_input = {
            "q":"input_norm", "k":"input_norm", "v":"input_norm",
            "o_proj":"attention", "gate":"post_norm", "up":"post_norm",
            "down":"silu_product",
        }
        pre_hook_quant_comparisons = {}
        for name,stage in stage_for_input.items():
            raw_torch, q_torch, scale_torch = pre_hook_states[name]
            raw_numpy = matched_states[stage]
            q_numpy, scale_numpy = base.quant_rows(raw_numpy)
            if raw_numpy.shape != raw_torch.shape: raise AssertionError(name)
            positions = np.argwhere(q_torch != q_numpy)
            first = [
                {"token":int(a), "channel":int(b),
                 "torch_pre_hook_input":float(raw_torch[a,b]),
                 "numpy_matched_input":float(raw_numpy[a,b]),
                 "torch_quantized":int(q_torch[a,b]),
                 "numpy_quantized":int(q_numpy[a,b])}
                for a,b in positions[:4]
            ]
            pre_hook_quant_comparisons[name] = {
                "input_stage":stage,
                "quantized_int8_mismatches":int(len(positions)),
                "float_input_max_abs_difference":float(np.max(np.abs(
                    raw_torch.astype(np.float64)-raw_numpy.astype(np.float64)))),
                "scale_max_abs_difference":float(np.max(np.abs(
                    scale_torch.astype(np.float64)-scale_numpy.astype(np.float64)))),
                "first_int8_mismatches":first,
            }''')
change('            "first_torch_fakequant_vs_numpy_matched_divergent_stage":divergent,',
'''            "first_torch_fakequant_vs_numpy_matched_divergent_stage":divergent,
            "first_mathematical_output_divergence_excluding_quantized_hook_inputs":true_divergent,
            "pre_hook_int8_comparisons":pre_hook_quant_comparisons,''')
with dest.open("x", encoding="utf-8") as f: f.write(source)
print(dest)
