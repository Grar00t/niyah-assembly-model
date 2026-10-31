#!/usr/bin/env python3
from pathlib import Path
root=Path("/home/a/qwen-layer0-quant-verify-v1")
source=root/"torch_hook_reference_v1.py"
target=root/"torch_hook_reference_v2.py"
text=source.read_text()
text=text.replace("report_torch_hooks_v1.json","report_torch_hooks_v2.json").replace("torch_hooks_v1.log","torch_hooks_v2.log")
old='        after_quant = base.fp32_transformers_trace(layer,x,cos,sin,config.rms_norm_eps)["final_residual"]'
new='''        fake_states = base.fp32_transformers_trace(layer,x,cos,sin,config.rms_norm_eps)
        matched_states, exact_acc = base.matched_reference(x, cos, sin, packed, norms, config.rms_norm_eps)
        stage_metrics = {name:base.metrics(fake_states[name],matched_states[name]) for name in base.STAGES}
        divergent = next((name for name in base.STAGES if stage_metrics[name]["failed_elements"]), None)
        after_quant = fake_states["final_residual"]'''
if text.count(old)!=1: raise SystemExit("Patch pattern 1 not unique")
text=text.replace(old,new)
old2='            "max_abs_native_vs_torch_fakequant":float(np.max(np.abs(out.astype(np.float64)-after_quant.astype(np.float64)))),'
new2='''            "first_torch_fakequant_vs_numpy_matched_divergent_stage":divergent,
            "torch_fakequant_vs_numpy_matched_by_stage":stage_metrics,
            "max_abs_native_vs_torch_fakequant":float(np.max(np.abs(out.astype(np.float64)-after_quant.astype(np.float64)))),'''
if text.count(old2)!=1: raise SystemExit("Patch pattern 2 not unique")
text=text.replace(old2,new2)
with target.open("x",encoding="utf-8") as output:
    output.write(text)
print(target)
