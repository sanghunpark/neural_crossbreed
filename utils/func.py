import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.blocks import ResBlocks, Conv2dBlock 

# import kornia

class Interp(nn.Module):
    def __init__(self, n_res, z_dim, norm, activ, pad_type):
        super(Interp, self).__init__()
        # self.model = []
        # dim = 2*z_dim + 1 # z_A + z_B + t
        # self.model += [ResBlocks(n_res, dim, norm,
        #                          activ, pad_type=pad_type)]
        # self.model += [Conv2dBlock(dim, z_dim, 7, 1, 3,
        #                            norm=norm,
        #                            activation=activ,
        #                            pad_type=pad_type)]        
        # self.model = nn.Sequential(*self.model)
    
    @staticmethod
    def lerp(z_A, z_B, t):
        while len(t.size()) < len(z_A.size()): 
            t = t.unsqueeze(1)
        z_t = (1-t)*z_A + t*z_B
        return z_t

    @staticmethod
    def slerp(z_A, z_B, t):
        while len(t.size()) < len(z_A.size()): 
            t = t.unsqueeze(1)
        cos_val = (z_A * z_B).sum(dim=1, keepdim=True)
        cos_val = cos_val / z_A.pow(2).sum(dim=1, keepdim=True).sqrt()
        cos_val = cos_val / z_B.pow(2).sum(dim=1, keepdim=True).sqrt()
        theta = torch.acos(cos_val)
        s1 = torch.sin((1-t)*theta)/torch.sin(theta)
        s2 = torch.sin(t*theta)/torch.sin(theta)
        z_t = s1*z_A + s2*z_B
        return z_t

    def forward(self, z_A, z_B, t): # learning based interpolation
        t = t.unsqueeze(1).unsqueeze(1).unsqueeze(1) # Bx1x1x1        
        z_ABt = torch.cat((z_A, z_B, t.expand(-1,-1,z_A.size(2),z_A.size(3))), 1)
        return self.model(z_ABt) # z_t

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

# def AdaIN(A_feat, B_feat):
#     assert (A_feat.size()[:2] == B_feat.size()[:2]), 'Batch size & channel size must be matched.'
#     A_size = A_feat.size()
#     B_mean, B_std = calc_mean_std(B_feat)
#     A_mean, A_std = calc_mean_std(A_feat)

#     # modulation
#     normalized_feat = (A_feat - A_mean.expand(A_size)) / A_std.expand(A_size)
#     return normalized_feat * B_std.expand(A_size) + B_mean.expand(A_size)

# def warp(x, flow):
#     H = x.size(2)
#     W = x.size(3)
#     grid = kornia.create_meshgrid(H, W).to(x.device)
#     flow_grid = grid + flow.permute(0,2,3,1)
#     return F.grid_sample(x, flow_grid)

def normalize_tensor(tensor):
    tensor = (tensor + 1.0) / 2.0 # [-1, 1] -> [0, 1]
    return torch.clamp(tensor, min=0, max=1) # fix a saturation artifcat cause of a 'tanh' activation in Decoder
