"""Prerequisite only: real CUDA work, backward and AdamW, not a pi0 smoke."""
import json
import pathlib
import platform
import time
import traceback

out = pathlib.Path(__file__).parent
result = {"gate": "torch_cuda_primitives", "status": "running", "cpu_arch": platform.machine()}
started = time.monotonic()
try:
    import torch
    import torchvision
    result.update(torch=torch.__version__, torchvision=torchvision.__version__,
                  cuda_runtime=torch.version.cuda, device=torch.cuda.get_device_name(0),
                  capability=list(torch.cuda.get_device_capability(0)),
                  compiled_arches=torch.cuda.get_arch_list())
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    model = torch.nn.Sequential(torch.nn.Linear(1024, 1024), torch.nn.LayerNorm(1024),
                                torch.nn.GELU(), torch.nn.Linear(1024, 1024)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    initial = model[0].weight.detach().clone()
    x = torch.randn(2, 256, 1024, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(x)
        attention = torch.nn.functional.scaled_dot_product_attention(
            y.reshape(2, 256, 8, 128).transpose(1, 2),
            y.reshape(2, 256, 8, 128).transpose(1, 2),
            y.reshape(2, 256, 8, 128).transpose(1, 2))
        grid = y[:, :196].transpose(1, 2).reshape(2, 1024, 14, 14)
        grid = torch.nn.functional.interpolate(grid, (16, 16), mode="bilinear", align_corners=False)
        loss = y.float().square().mean() + attention.float().square().mean() + grid.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    torch.cuda.synchronize()
    assert not torch.equal(initial, model[0].weight)
    result.update(status="passed", loss=float(loss.detach()), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), optimizer_changed_weights=True)
except Exception as exc:
    result.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
finally:
    result["seconds"] = time.monotonic() - started
    (out / "torch_compute_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
if result["status"] != "passed":
    raise SystemExit(1)
