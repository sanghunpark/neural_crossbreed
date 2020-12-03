#
import numpy as np

# torch
import torch
from torch import nn
from torch.nn import functional as F

# Blocks
from model.blocks import LinearBlock, Conv2dBlock, ActFirstResBlocks, ActFirstResBlock, UpdownResBlock

class GPPatchMcResDis(nn.Module):
    def __init__(self, input_dim, dim, n_class, n_res):
        super(GPPatchMcResDis, self).__init__()
        assert n_res % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = n_res // 2
        cnn_f = [Conv2dBlock(input_dim, dim, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            output_dim = np.min([dim * 2, 1024])
            cnn_f += [ActFirstResBlock(dim, dim, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(dim, output_dim, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            dim = np.min([dim * 2, 1024])
        output_dim = np.min([dim * 2, 1024])
        cnn_f += [ActFirstResBlock(dim, dim, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(dim, output_dim, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(output_dim, n_class, 1, 1,
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)


    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        feat = self.cnn_f(x)
        out = self.cnn_c(feat)
        index = torch.LongTensor(range(out.size(0))).cuda()
        out1 = out[index, y, :, :]
        return out1, out

class Encoder(nn.Module):
    def __init__(self, downs, input_dim, dim, n_res_blks, norm, activ, pad_type, global_pool=False, keepdim=False):
        super(Encoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3,
                                   norm='none',
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(downs):
            self.model += [UpdownResBlock(dim, dim*2, norm=norm, activation=activ, updown='down')]
            dim *= 2
        # resblks
        self.model += [ActFirstResBlocks(n_res_blks, fin=dim, fout=dim,
                                norm=norm,
                                activation=activ,
                                pad_type=pad_type)]
        # global pooling
        self.global_pool = global_pool
        self.keepdim = keepdim

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        x = self.model(x)
        if self.global_pool:
           x = torch.sum(x, dim=(2,3), keepdim=self.keepdim)
        return x

class Decoder(nn.Module):
    def __init__(self, ups, dim, output_dim, n_res_blks, norm, activ, pad_type, upsample=False):
        super(Decoder, self).__init__()
        self.model = []
        # resblks
        self.model += [ActFirstResBlocks(n_res_blks, fin=dim, fout=dim,
                                norm=norm,
                                activation=activ,
                                pad_type=pad_type)]
        for i in range(ups): # AdaIN used only upsample layer
            self.model += [UpdownResBlock(dim, dim//2, norm=norm, activation=activ, updown='up')]
            dim //= 2
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, 
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

        self.upsample = upsample

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, dim, output_dim, n_blk, norm, activ):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))