"""Activation helper inherited from D-FINE / RT-DETR.

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn


def get_activation(act: str, inpace: bool = True):
    """get activation"""
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act

    act = act.lower()

    if act == "silu" or act == "swish":
        m = nn.SiLU()

    elif act == "relu":
        m = nn.ReLU()

    elif act == "leaky_relu":
        m = nn.LeakyReLU()

    elif act == "silu":
        m = nn.SiLU()

    elif act == "gelu":
        m = nn.GELU()

    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError("")

    if hasattr(m, "inplace"):
        m.inplace = inpace

    return m
