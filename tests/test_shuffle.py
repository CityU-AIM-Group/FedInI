import torch
import numpy as np

t = torch.rand(2,3,2,2)
print('Original Tensor:', t)

k = torch.stack((t[1], t[0]))
print(k)