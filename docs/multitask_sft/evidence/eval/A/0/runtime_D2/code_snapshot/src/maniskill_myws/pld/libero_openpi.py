"""Official OpenPI model/transforms with Transformers' safe fast-loading utility."""
import dataclasses
from openpi.models.pi0_config import Pi0Config


def load_pi0_pytorch(config,weight_path):
    import safetensors.torch
    from transformers.modeling_utils import no_init_weights
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    # CPU buffers still initialize normally (unlike meta/to_empty). Every model
    # parameter must subsequently be supplied by strict checkpoint loading.
    with no_init_weights():
        model=PI0Pytorch(config)
    model.paligemma_with_expert.paligemma.tie_weights()
    safetensors.torch.load_model(model,str(weight_path),strict=True)
    return model


class FastLoadPi0Config(Pi0Config):
    def load_pytorch(self,train_config,weight_path):
        return load_pi0_pytorch(train_config.model,weight_path)


def fast_loading_config(config):
    return FastLoadPi0Config(**{f.name:getattr(config,f.name) for f in dataclasses.fields(config) if f.init})
