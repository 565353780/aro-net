import torch


def np2th(array, device: str='cpu'):
    tensor = array
    if type(array) is not torch.Tensor:
        tensor = torch.tensor(array).float()
    if type(tensor) is torch.Tensor:
        return tensor.to(device)
    else:
        return array
