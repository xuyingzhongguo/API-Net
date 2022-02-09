import torch

path = 'model_save/res101/2619307/model_best.pth.tar'
checkpoint = torch.load(path)
print('loaded checkpoint {}(epoch {})'.format(path, checkpoint['epoch']))
