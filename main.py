import os
import sys
import time
import multiprocessing as mp

from helper.argvs import parse_args
from utils.trainer import Imgtrainer

import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, _create_vision_transformer, default_cfgs

os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

@register_model
def vit_base_patch16_224_l2p(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_l2p', pretrained=pretrained, **model_kwargs)
    return model

default_cfgs['vit_base_patch16_224_l2p'] = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        num_classes=21843)

def main(kwargs):
    print(kwargs)
    mp.set_start_method('spawn')
    trainer = Imgtrainer(**kwargs)
    trainer.run()
    time.sleep(60)

if __name__ == "__main__":
    main(vars(parse_args(sys.argv)))
    print("Done")
