import torch

t = torch.ones([3, 2])
t2 = torch.Tensor([
    [0.5, 2.0],
    [1.0, 3.0],
    [4.0, 5.0]
])
print((t * t2).shape, t * t2)