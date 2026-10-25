"""Baselines for the pilot subset: eager + torch.compile(inductor) reference times.

One process, one GPU window. Writes results/baselines.json incrementally
(crash-safe). Protocol identical to eval_protected (same constants, imported).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
import problems  # noqa: E402
from eval_protected import time_fn, set_seed  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "baselines.json"


def main(only: list[str] | None = None):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    done = {}
    if OUT.exists():
        done = json.loads(OUT.read_text())

    for name in problems.subset_names(only):
        if name in done:
            continue
        Model, get_inputs, get_init_inputs, _ = problems.load_problem(name)
        init_inputs = get_init_inputs()
        row = {"problem": name}

        try:
            model = Model(*init_inputs).to(device=device, dtype=torch.float32)
            inputs = [x.to(device=device, dtype=torch.float32) if isinstance(x, torch.Tensor) else x
                      for x in get_inputs()]
            with torch.no_grad():
                row["eager"] = time_fn(model, tuple(inputs), device)
        except torch.OutOfMemoryError as e:
            row["eager"] = None
            row["eager_oom"] = str(e)[:150]
        except Exception as e:
            row["eager"] = None
            row["eager_error"] = repr(e)[:200]

        try:
            torch._dynamo.reset()
            cmodel = torch.compile(model, backend="inductor", mode="default")
            t0 = time.time()
            with torch.no_grad():
                row["compiled"] = time_fn(cmodel, tuple(inputs), device)
            row["compile_s"] = round(time.time() - t0, 1)
        except torch.OutOfMemoryError as e:
            row["compiled"] = None
            row["compiled_oom"] = str(e)[:150]
        except Exception as e:
            row["compiled"] = None
            row["compiled_error"] = repr(e)[:200]

        done[name] = row
        OUT.write_text(json.dumps(done, indent=1))
        e = row["eager"]["median_ms"] if row.get("eager") else None
        c = row["compiled"]["median_ms"] if row.get("compiled") else None
        print(f"{name}: eager {e} compiled {c}", flush=True)
        try:
            del model, cmodel, inputs
        except Exception:
            pass
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or None)
