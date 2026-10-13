import unittest
import torch
from convert_dinov3 import convert_state_dict


class ConversionContracts(unittest.TestCase):
    def test_qkv_split_and_key_bias_mask_preserve_weights(self):
        weight=torch.arange(12.).reshape(6,2)
        source={'blocks.0.attn.qkv.weight':weight,
                'blocks.0.attn.qkv.bias':torch.arange(6.),
                'blocks.0.attn.qkv.bias_mask':torch.tensor([1.,1.,0.,0.,1.,1.]),
                'mask_token':torch.zeros(1,2)}
        converted=convert_state_dict(source)
        torch.testing.assert_close(converted['layer.0.attention.q_proj.weight'],weight[:2])
        torch.testing.assert_close(converted['layer.0.attention.k_proj.weight'],weight[2:4])
        torch.testing.assert_close(converted['layer.0.attention.v_proj.bias'],torch.tensor([4.,5.]))
        self.assertNotIn('layer.0.attention.k_proj.bias',converted)
        self.assertEqual(converted['embeddings.mask_token'].shape,(1,1,2))

    def test_unmasked_nonzero_key_bias_cannot_be_silently_discarded(self):
        with self.assertRaisesRegex(ValueError,'key bias'):
            convert_state_dict({'blocks.0.attn.qkv.bias':torch.ones(6)})

    def test_unknown_tensor_cannot_be_silently_dropped(self):
        with self.assertRaisesRegex(ValueError,'Unknown'):
            convert_state_dict({'unexpected.weight':torch.zeros(2)})


if __name__=='__main__': unittest.main()
