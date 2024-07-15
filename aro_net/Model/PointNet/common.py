import torch

def maxpool(x: torch.Tensor, dim: int=-1, keepdim: bool=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out
