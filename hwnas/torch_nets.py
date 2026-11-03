"""PyTorch realisation of the search space (inverted-residual blocks, ReLU6, BN)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .space import OPS, SKIP, Arch, Layer, SearchSpace


def conv_bn(cin, cout, k, stride=1, groups=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class MBConv(nn.Module):
    def __init__(self, cin, cout, k, e, stride):
        super().__init__()
        m = cin * e
        layers = [conv_bn(cin, m, 1)] if e > 1 else []
        layers += [conv_bn(m, m, k, stride, groups=m), conv_bn(m, cout, 1, act=False)]
        self.body = nn.Sequential(*layers)
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.body(x)
        return x + y if self.res else y


def make_op(L: Layer, op_idx: int) -> nn.Module:
    if op_idx == SKIP:
        assert L.can_skip, "skip is only valid when shape is preserved"
        return nn.Identity()
    op = OPS[op_idx]
    return MBConv(L.cin, L.cout, op.kernel, op.expand, L.stride)


def make_stem(space: SearchSpace) -> nn.Module:
    c = space.cfg
    return conv_bn(c.in_ch, c.stem_ch, 3, c.stem_stride)


def make_head(space: SearchSpace) -> nn.Module:
    c = space.cfg
    return nn.Sequential(conv_bn(space.last_ch, c.head_ch, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(c.head_ch, c.num_classes))


class Net(nn.Module):
    def __init__(self, space: SearchSpace, arch: Arch):
        super().__init__()
        assert space.is_valid(arch), "invalid architecture for this space"
        self.stem = make_stem(space)
        self.blocks = nn.ModuleList(make_op(L, a) for L, a in zip(space.layers, arch))
        self.head = make_head(space)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def build_network(space: SearchSpace, arch: Arch) -> Net:
    return Net(space, arch)
