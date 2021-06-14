from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import haiku as hk
from haiku._src.data_structures import FlatMapping

from davis_utils.jax import adaptive_pool, maybe_hk_dropout

# ported from: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class Features(hk.Module):
    def __call__(self, x):
        module_num = 0
        for layer_type in CFG:
            if layer_type == 'M':
                x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID', channel_axis=0)
                module_num += 1
            else:
                conv = hk.Conv2D(layer_type, kernel_shape=3, padding=(1,1), data_format='NCHW', name=f'conv_{module_num}')
                x = conv(x)
                module_num += 1
                x = jax.nn.relu(x)
                module_num += 1
        return x

class Classifier(hk.Module):
    def __call__(self, x):
        x = hk.Linear(4096, name='l0')(x)
        x = jax.nn.relu(x)

        x = maybe_hk_dropout(0.5, x)

        x = hk.Linear(4096, name='l3')(x)
        x = jax.nn.relu(x)
        x = maybe_hk_dropout(0.5, x)

        x = hk.Linear(1000, name='l6')(x)
        return x

class VGG16(hk.Module):
    def __call__(self, x):
        """x should be in CHW format"""
        x = Features(name='features')(x)
        x = adaptive_pool(hk.avg_pool, x, (7, 7))

        x = x.reshape(-1)

        x = Classifier(name='classifier')(x)

        return x


def _dicts_to_flatmappings(tree):
    if isinstance(tree, dict):
        return FlatMapping(**{k: _dicts_to_flatmappings(v) for k, v in tree.items()})
    return tree

def get_model(with_dropout=False, weights_location='weights.pkl'):
    weights_location = Path(weights_location)
    with weights_location.open('rb') as f:
        weights = pickle.load(f)

    weights = jax.tree_map(jnp.array, weights)
    weights = _dicts_to_flatmappings(weights)

    def model_fn(x):
        return VGG16(name='model')(x)

    model = hk.transform(model_fn)
    if not with_dropout:
        model = hk.without_apply_rng(model)

    return model, weights

if __name__ == '__main__':
    model, weights = get_model()
    model.apply(weights, jnp.zeros((3, 224, 224)))
