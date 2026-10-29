"""JAX CUDA JIT/gradient/optimizer prerequisite; does not replace pi0 tests."""
import json
import os
import pathlib
import time
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
out = pathlib.Path(__file__).parent
result = {"gate": "jax_cuda_primitives", "status": "running"}
started = time.monotonic()
try:
    import jax
    import jax.numpy as jnp
    import optax
    import numpy as np
    result.update(jax=jax.__version__, devices=[str(d) for d in jax.devices()])
    assert jax.devices()[0].platform == "gpu"
    tx = optax.adamw(1e-4)
    params = {"w": jnp.ones((1024, 1024), jnp.float32) * .01,
              "b": jnp.zeros((1024,), jnp.float32)}
    state = tx.init(params)
    x = jnp.ones((256, 1024), jnp.bfloat16)
    initial = np.asarray(params["w"])

    @jax.jit
    def step(params, state, x):
        def objective(p):
            y = x @ p["w"].astype(jnp.bfloat16) + p["b"]
            return jnp.mean(jnp.square(jax.nn.gelu(y).astype(jnp.float32)))
        loss, grad = jax.value_and_grad(objective)(params)
        updates, state = tx.update(grad, state, params)
        return optax.apply_updates(params, updates), state, loss, optax.global_norm(grad)

    params, state, loss, grad_norm = step(params, state, x)
    jax.block_until_ready((params, state, loss, grad_norm))
    assert np.isfinite(float(loss)) and np.isfinite(float(grad_norm)) and float(grad_norm) > 0
    assert not np.array_equal(initial, np.asarray(params["w"]))
    result.update(status="passed", loss=float(loss), gradient_norm=float(grad_norm),
                  optimizer_changed_weights=True, memory_stats=jax.devices()[0].memory_stats())
except Exception as exc:
    result.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
finally:
    result["seconds"] = time.monotonic() - started
    (out / "jax_compute_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
if result["status"] != "passed":
    raise SystemExit(1)
