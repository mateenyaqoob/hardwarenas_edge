import numpy as np
import pytest

from hwnas import OPS, SearchSpace, arch_metrics
from hwnas.cost import block_parts


@pytest.fixture(scope="module")
def space():
    return SearchSpace()


def test_space_shape(space):
    assert space.n_layers == 11
    assert space.size() == int(np.prod([len(v) for v in space.valid_ops]))
    # skip is only legal where nothing changes shape
    for L, v in zip(space.layers, space.valid_ops):
        assert (0 in v) == L.can_skip


def test_random_mutate_crossover_valid(space):
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = space.random_arch(rng), space.random_arch(rng)
        assert space.is_valid(a)
        m = space.mutate(a, rng)
        assert space.is_valid(m) and m != a
        assert space.is_valid(space.crossover(a, b, rng))


def test_encode_one_hot(space):
    a = space.random_arch(np.random.default_rng(1))
    z = space.encode(a)
    assert z.sum() == space.n_layers and set(np.unique(z)) <= {0.0, 1.0}


def test_depthwise_separable_is_cheaper(space):
    L = space.layers[3]  # stride-2 layer, 24 -> 32 channels
    std = L.res_out ** 2 * 9 * L.cin * L.cout
    sep = sum(p.macs for p in block_parts(L, OPS[1]))  # k3, e1: dw + pw
    assert sep < std / 3


def test_bigger_ops_cost_more(space):
    base = tuple(1 for _ in space.layers)
    big = tuple(5 for _ in space.layers)
    mb, mg = arch_metrics(space, base), arch_metrics(space, big)
    assert mg["macs"] > mb["macs"] and mg["params"] > mb["params"]
    assert mg["peak_sram_bytes"] > mb["peak_sram_bytes"]


def test_skip_reduces_cost(space):
    a = list(5 for _ in space.layers)
    i = next(i for i, L in enumerate(space.layers) if L.can_skip)
    b = list(a); b[i] = 0
    assert arch_metrics(space, tuple(b))["macs"] < arch_metrics(space, tuple(a))["macs"]


def test_int8_vs_fp32_memory(space):
    a = space.random_arch(np.random.default_rng(2))
    m1, m4 = arch_metrics(space, a, 1), arch_metrics(space, a, 4)
    assert m4["flash_bytes"] == 4 * m1["flash_bytes"] and m4["peak_sram_bytes"] == 4 * m1["peak_sram_bytes"]
