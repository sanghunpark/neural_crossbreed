# PyTorch
import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F

def match_size(w, x):
    if w is not None:
        while len(w.size()) < len(x.size()): 
            w = w.unsqueeze(1)
    else:
        w = 1
    return w

# hinge loss version
def D_adv_loss(pred, real = False, w=None): # hinge loss version
    w = match_size(w, pred)

    if real:
        return (w*F.relu(1 - pred)).mean()
    else:       
        return (w*F.relu(1 + pred)).mean() # PatchGAN


def D_gp_loss(dis_input, dis_out): # gradient penalty on real data
    batch_size = dis_input.size(0)
    # real = autograd.Variable(real, requires_grad=True)
    grad_penalty = autograd.grad(outputs=dis_out.mean(),
                                inputs=dis_input,
                                # grad_outputs=torch.ones(dis_out.size()).cuda(),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True)[0]
    
    grad_penalty = grad_penalty.pow(2)
    assert (grad_penalty.size() == dis_input.size())
    real_grad = grad_penalty.view(batch_size, -1).sum(1)
    return real_grad.mean()

def G_adv_loss(pred_fake, w=None): # modified minimax Loss: min log(1-D(G(z))) > max log(G(z))
    w = match_size(w, pred_fake)
    return (w*(-pred_fake)).mean()

def G_rec_loss(real, fake, w=None):
    w = match_size(w, real)
    return (w*(nn.L1Loss(reduction='none')(real, fake, ))).mean()

def G_rec_loss(real, fake, w=None):
    w = match_size(w, real)
    return (w*(nn.L1Loss(reduction='none')(real, fake, ))).mean()