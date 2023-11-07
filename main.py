import torch
import torchvision
print(torchvision.__version__)
x = torch.rand(5, 3)
print(torch.cuda.is_available() )
print(x)


print(torch.version.cuda)
print(torch.cuda.is_available())