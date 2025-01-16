import torch

def create_zero_list(model, cpu=False):
    l = []
    param_list = list(model.parameters())
    for i in range(0, len(param_list)):
        if cpu:
            l.append(torch.zeros_like(param_list[i]).to('cpu'))
        else:
            l.append(torch.zeros_like(param_list[i]))
    return l

def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)