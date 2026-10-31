#!/usr/bin/env python3
"""Version 2: observation-only brace fix; original v1 remains immutable."""
import importlib.util
from pathlib import Path
root = Path("/home/a/qwen-layer0-quant-verify-v1")
spec = importlib.util.spec_from_file_location("layer0_verification_v1", root / "verify.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = module.diagnostic_source

def corrected_diagnostic_source(source):
    result = original(source)
    bad = " for(size_t i=0;i<n;++i)res[i]=x[i]+p[i]; TRACE_F(11,res,t,4096);"
    good = " for(size_t i=0;i<n;++i){res[i]=x[i]+p[i];}\n TRACE_F(11,res,t,4096);"
    if result.count(bad) != 1:
        raise RuntimeError("Expected exactly one trace indentation site")
    return result.replace(bad, good, 1)

module.diagnostic_source = corrected_diagnostic_source
module.REPORT = root / "report_v2.json"
module.LOG = root / "run_v2.log"
module.__file__ = str(Path(__file__))
if module.REPORT.exists() or module.LOG.exists():
    raise SystemExit("STOP: version 2 artifacts already exist")
raise SystemExit(module.main())
