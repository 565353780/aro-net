import torch


def np2th(array, device="cuda"):
    tensor = array
    if type(array) is not torch.Tensor:
        tensor = torch.tensor(array).float()
    if type(tensor) is torch.Tensor:
        if device == "cuda":
            return tensor.cuda()
        return tensor.cpu()
    else:
        return array
