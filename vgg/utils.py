import haiku as hk
from haiku._src.data_structures import FlatMapping

def dicts_to_flatmappings(tree):
    if isinstance(tree, dict):
        return FlatMapping(**{k: dicts_to_flatmappings(v) for k, v in tree.items()})
    return tree

def adaptive_pool(pool_func, value, output_shape, padding='VALID'):
    """Expects CHW format"""
    strides, shape = zip(*(
        ((stride := inp_size // out_size), inp_size - (out_size - 1) * stride)
        for inp_size, out_size in zip(value.shape[1:], output_shape)))

    return pool_func(value, window_shape=shape, strides=strides, padding=padding, channel_axis=0)

def maybe_hk_dropout(rate, value):
    key = hk.maybe_next_rng_key()
    if key is not None:
        value = hk.dropout(key, rate, value)
    return value
