#!/usr/bin/env python3
"""Version 3: preserve v1/v2, collect exact worst token/channel and all three outputs."""
import importlib.util
import json
import numpy as np
from pathlib import Path

root = Path("/home/a/qwen-layer0-quant-verify-v1")
spec = importlib.util.spec_from_file_location("layer0_verification_v1", root / "verify.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original_diagnostic = module.diagnostic_source

def corrected_diagnostic_source(source):
    result = original_diagnostic(source)
    old = " for(size_t i=0;i<n;++i)res[i]=x[i]+p[i]; TRACE_F(11,res,t,4096);"
    new = " for(size_t i=0;i<n;++i){res[i]=x[i]+p[i];}\n TRACE_F(11,res,t,4096);"
    if result.count(old) != 1:
        raise RuntimeError("Unexpected C trace source; refusing to patch")
    return result.replace(old, new, 1)

module.diagnostic_source = corrected_diagnostic_source
module.REPORT = root / "report_v3.json"
module.LOG = root / "run_v3.log"
module.__file__ = str(Path(__file__))
details_path = root / "worst_values_v3.json"
if module.REPORT.exists() or module.LOG.exists() or details_path.exists():
    raise SystemExit("STOP: v3 evidence path already exists")

original_metrics = module.metrics
captured = {}
worst_rows = []
counter = [0]
per = len(module.STAGES) * 2

def record_metrics(native_or_fake, fake_or_fp32):
    i = counter[0] % per
    if i == len(module.STAGES) - 1:
        captured["native"] = native_or_fake.copy()
        captured["fake"] = fake_or_fp32.copy()
    elif i == per - 1:
        captured["fp32"] = fake_or_fp32.copy()
        native, fake, fp32 = [captured[name] for name in ("native", "fake", "fp32")]
        if native.shape != fake.shape or native.shape != fp32.shape:
            raise AssertionError("Final tensor shape mismatch")
        delta = np.abs(native.astype(np.float64) - fp32.astype(np.float64))
        token, channel = [int(z) for z in np.unravel_index(np.argmax(delta), delta.shape)]
        vals = dict(
            worst_token_index=token,
            worst_channel_index=channel,
            native_value=float(native[token, channel]),
            fakequant_value=float(fake[token, channel]),
            fp32_reference_value=float(fp32[token, channel]),
            abs_error_native_vs_fp32=float(delta[token, channel]),
            max_abs_reference=float(np.max(np.abs(fp32.astype(np.float64)))),
            native_vs_fakequant_rel_l2=original_metrics(native, fake)["relative_l2"],
            fakequant_vs_fp32_rel_l2=original_metrics(fake, fp32)["relative_l2"],
            native_vs_fp32_rel_l2=original_metrics(native, fp32)["relative_l2"],
        )
        worst_rows.append(vals)
        captured.clear()
    counter[0] += 1
    return original_metrics(native_or_fake, fake_or_fp32)

module.metrics = record_metrics
exit_code = module.main()
report = json.loads(module.REPORT.read_text())
for i, vals in enumerate(worst_rows):
    if i < len(report["comparisons"]):
        vals["case"] = report["comparisons"][i]["case"]
        vals["tokens"] = report["comparisons"][i]["tokens"]
payload = {
    "reference": str(module.REPORT),
    "cases_expected": 6,
    "cases_recorded": len(worst_rows),
    "metrics": worst_rows,
    "verification_script_sha256": module.sha(Path(__file__)),
    "source_sha256": module.sha(module.ORIGINAL),
    "kernel_sha256": module.sha(module.KERNEL),
}
with details_path.open("x", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
for row in worst_rows:
    print(json.dumps(row, ensure_ascii=False), flush=True)
if len(worst_rows) != 6 or exit_code != 0:
    raise SystemExit(1)
raise SystemExit(0)
