"""GPU window 1 sanity: torch-ROCm sees gfx1100, eager works, inductor compiles.

Exits 0 with a JSON verdict on stdout; the caller records it verbatim.
"""
import json
import torch

r = {"torch": torch.__version__}
try:
    r["cuda_available"] = torch.cuda.is_available()
    if not r["cuda_available"]:
        print(json.dumps(r, indent=1))
        sys.exit(1)
    r["device"] = torch.cuda.get_device_name(0)
    r["hip"] = torch.version.hip
    r["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)

    # eager sanity
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    r["eager_mm_ok"] = torch.isfinite(y).all().item()

    # inductor sanity, three shapes of op
    def run_compiled(fn, name, *args):
        try:
            cfn = torch.compile(fn, backend="inductor", mode="default")
            out = cfn(*args)
            torch.cuda.synchronize()
            r[name] = True
        except Exception as e:
            r[name] = False
            r[name + "_err"] = repr(e)[:1500]

    run_compiled(lambda a: torch.nn.functional.gelu(a), "inductor_gelu", x)
    run_compiled(lambda a: torch.softmax(a, dim=-1), "inductor_softmax", x)
    run_compiled(lambda a, b: a @ b + a.sum(), "inductor_mm_add", x, torch.randn(1024, 1024, device="cuda"))

    try:
        import triton
        r["triton"] = triton.__version__
    except Exception as e:
        r["triton"] = "ERR " + repr(e)[:300]

    r["vram_free_gb_after"] = round(torch.cuda.mem_get_info()[0] / 2**30, 2)
except Exception as e:
    r["fatal"] = repr(e)[:2000]

out = json.dumps(r, indent=1)
with open(str(__import__("pathlib").Path(__file__).resolve().parent.parent / "results" / "sanity-gpu-w1.json"), "w") as f:
    f.write(out + "\n")
print(out)
import sys
sys.exit(0 if r.get("eager_mm_ok") else 1)
