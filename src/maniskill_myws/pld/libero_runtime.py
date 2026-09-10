"""Versioned numerical execution of the frozen JAX base (not SFT settings)."""
import importlib.metadata
import os


def _requirements(config):
    contract=config.get('base_numerical_contract')
    if contract is None:return None
    if contract!='jax_cuda_autotune0_v1':
        raise ValueError(f'Unknown base numerical contract: {contract}')
    return {'JAX_PLATFORMS':'cuda','XLA_FLAGS':'--xla_gpu_autotune_level=0'}


def configure_base_inference(config,environ=None):
    """CLI must call before initializing any JAX backend; reject external drift."""
    env=os.environ if environ is None else environ
    required=_requirements(config)
    if required is None:return
    for key,value in required.items():
        if env.get(key) not in (None,'',value):
            raise ValueError(f'{key} conflicts with frozen-base numerical contract')
    env.update(required)


def require_base_inference_runtime(config,environ=None):
    """Library calls fail closed unless the launcher configured the runtime."""
    env=os.environ if environ is None else environ
    required=_requirements(config)
    if required is None:return
    for key,value in required.items():
        if env.get(key)!=value:
            raise ValueError(f'{key} must be configured before JAX backend initialization')
    for package in ('jax','jaxlib'):
        if importlib.metadata.version(package)!='0.5.3':
            raise ValueError(f'{package} version differs from verified numerical contract0.5.3')
