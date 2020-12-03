#
import math
import numpy as np

# torch
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

# My library
from utils.utils import get_device
from model.blocks import assign_adain_params, get_num_adain_params
from model.nets import Encoder, Decoder, MLP, GPPatchMcResDis
from model.loss import D_gp_loss, D_adv_loss, G_adv_loss, G_rec_loss
from utils.func import Interp #, AdaIN

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        config_G = config['gen']
        n_cont_down = config_G['n_cont_down']
        n_f = config_G['n_f']
        n_cont_res_blks = config_G['n_cont_res_blks']
        self.cont_enc = Encoder(
                            downs=n_cont_down,
                            input_dim=3,
                            dim=n_f,
                            n_res_blks=n_cont_res_blks,
                            norm='in',
                            activ='relu',
                            pad_type='reflect',
                            global_pool=False,
                            keepdim=False)

        self.dec = Decoder(
                            ups=n_cont_down,
                            dim=self.cont_enc.output_dim,
                            output_dim=3,
                            n_res_blks=n_cont_res_blks,
                            norm='adain', # Decoder use 'adain' in upsampling layer
                            activ='relu',
                            pad_type='reflect',
                            upsample=False)

        n_style_down = config_G['n_style_down']
        n_style_res_blks = config_G['n_style_res_blks']
        self.style_enc = Encoder(
                            downs=n_style_down,
                            input_dim=3,
                            dim=n_f,
                            n_res_blks=n_style_res_blks, # no resisual block for Style Encoder
                            norm='in',
                            activ='relu',
                            pad_type='reflect',
                            global_pool=True)

        n_mlp_f = config_G['n_mlp_f']
        mlp_dim = self.style_enc.output_dim
        if n_style_res_blks is not 0:
            size = config['img_size'] / (2**n_style_down)
            mlp_dim = int(mlp_dim * size * size)
        # mlp_dim = n_f * (2**n_style_down)
        n_mlp_blks = config_G['n_mlp_blks']
        self.mlp = MLP(     input_dim=mlp_dim,
                            dim=n_mlp_f,
                            output_dim=get_num_adain_params(self.dec),
                            n_blk=n_mlp_blks,
                            norm='none',
                            activ='relu')

    def forward(self, x_A, x_B=None, a=None, x_a=None, train=True, a_c=None, a_s=None):       
        B, C, H, W = x_A.size()
        if train is True:
            ''' encoding '''
            # forward input frames to content encoder
            z_A = self.cont_enc(x_A)
            z_B = self.cont_enc(x_B)
            
            # forward input frames to apperance encoder & mlp
            s_A = self.style_enc(x_A)
            s_B = self.style_enc(x_B)           

            ''' decoding '''
            assign_adain_params(self.mlp(s_A), self.dec)
            y_AA = self.dec(z_A)
            y_BA = self.dec(z_B)
            assign_adain_params(self.mlp(s_B), self.dec)
            y_BB = self.dec(z_B)
            y_AB = self.dec(z_A)

            ''' interpolation '''
            z_a = Interp.lerp(z_A, z_B, a)
            s_a = Interp.lerp(s_A, s_B, a)
            assign_adain_params(self.mlp(s_a), self.dec)
            y_aa = self.dec(z_a)

            ''' cycle reconstruction '''
            # content preserving
            z_A = self.cont_enc(y_AB)
            z_B = self.cont_enc(y_BA)
            
            assign_adain_params(self.mlp(s_A), self.dec)
            y_rAA = self.dec(z_A)

            assign_adain_params(self.mlp(s_B), self.dec)
            y_rBB = self.dec(z_B)         
            return y_AA, y_BB, y_aa, y_AB, y_BA, y_rAA, y_rBB

        else: # inference
            z_A = self.cont_enc(x_A)
            z_B = self.cont_enc(x_B)
            z_a = Interp.lerp(z_A, z_B, a_c)

            s_A = self.style_enc(x_A)
            s_B = self.style_enc(x_B)
            s_a = Interp.lerp(s_A, s_B, a_s)

            assign_adain_params(self.mlp(s_a), self.dec)
            y_a = self.dec(z_a)
            return y_a


class Discriminator(nn.Module):
    def __init__(self, config, n_class):
        super(Discriminator, self).__init__()
        self.disc = GPPatchMcResDis(
                                    input_dim=3,
                                    dim=config['n_f'],
                                    n_class=n_class,
                                    n_res=config['n_res'])
    
    def forward(self, x, idx):
        return self.disc(x, idx)


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun

class NeuralCrossbreed(nn.Module):
    def __init__(self, args, config, n_class, fine_tune=False):
        super(NeuralCrossbreed, self).__init__()

        ## set networks
        self.G = Generator(config)
        self.D = Discriminator(config['dis'], n_class)

        # for multi-gpu
        self.fine_tune = fine_tune
        self.device, self.multi_gpu = get_device(args)
        if self.multi_gpu: 
            self.G = nn.DataParallel(self.G, device_ids=list(range(args.gpu_1st, args.gpu_1st + args.ngpu)),
                               output_device=args.gpu_1st)
            self.D = nn.DataParallel(self.D, device_ids=list(range(args.gpu_1st, args.gpu_1st + args.ngpu)),
                               output_device=args.gpu_1st)
            print('multi-gpu')
        
        # set gpu config        
        self.G.to(self.device)
        self.D.to(self.device)

        # set loss weights
        self.gp_w = config['gp_w']
        self.adv_w = config['adv_w']
        self.rec_w = config['rec_w']

        # set optimizer 
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=config['G_lr'], betas=(config['beta1'], config['beta2']), eps=config['eps'])
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=config['D_lr'], betas=(config['beta1'], config['beta2']), eps=config['eps'])
        
        # init networks
        self.G.apply(weights_init('gaussian'))
        self.D.apply(weights_init('gaussian'))

        # indices
        global B
        B = config['batch_size']
        self.set_current_batch_size()
 

    def set_current_batch_size(self):        
        global AA
        global BB
        global aa
        global AB
        global BA
        global rAA
        global rBB

        global B
        print('batch size is changed, B: %d' %(B))
        AA = list(range(0,B))
        BB = list(range(B,2*B))
        aa = list(range(2*B,3*B))
        AB = list(range(3*B,4*B))
        BA = list(range(4*B,5*B))
        rAA = list(range(5*B,6*B))
        rBB = list(range(6*B,7*B))
        # ArA = list(range(7*B,8*B))
        # BrB = list(range(8*B,9*B))

    def forward(self, data):
        pass

    def set_finetune(self, fine_tune):
        self.fine_tune = fine_tune

    def get_device(self):
        return self.device

    def get_data(self, data):
        # get data
        x_A = data['x_A'].to(self.device)
        x_B = data['x_B'].to(self.device)
        
        A_idx = data['A_idx'].to(self.device)
        B_idx = data['B_idx'].to(self.device)
        global B
        if (B != x_A.size(0)):
            B = x_A.size(0)
            self.set_current_batch_size()

        if self.fine_tune == False:
            x_a = data['x_a'].to(self.device)
            a = data['a'].to(self.device)
            return x_A, x_B, x_a, a, A_idx, B_idx  
        else:
            a = 0.5 * torch.ones(B, 1).to(self.device)
            return x_A, x_B, a, A_idx, B_idx            

    def train_model(self, data, train=True):
        # set data
        if self.fine_tune == False:
            x_AA, x_BB, x_a, a, A_idx, B_idx = self.get_data(data)
            x = torch.cat([x_AA, x_BB, x_a])
        else:
            x_AA, x_BB, a, A_idx, B_idx = self.get_data(data)
            x = torch.cat([x_AA, x_BB])
        cls_idx = torch.cat([A_idx, B_idx], dim=0)

        ### forward G
        y_AA, y_BB, y_aa, y_AB, y_BA, y_rAA, y_rBB = self.G(x_AA, x_BB, a=a)
        y = torch.cat([y_AA, y_BB, y_aa, y_AB, y_BA, y_rAA, y_rBB])

        ### train D
        # forward D with all-real batch        
        with torch.enable_grad():
            x.requires_grad_()
            pred_real, feat_real = self.D(x[AA+BB], cls_idx[AA+BB])
            loss_D_gp = D_gp_loss(x, pred_real)
        
        # forward D with all-fake batch        
        pred_fake, _ = self.D(y[AA+BB+BA+AB+rAA+rBB].detach(), cls_idx[AA+BB+AA+BB+AA+BB]) # to stop updating G
        loss_D_adv = D_adv_loss(pred_real, real=True) + D_adv_loss(pred_fake, real=False)

        # forward D with interp batch        
        pred_real_A,  _ = self.D(x[aa].detach(), cls_idx[AA])
        pred_real_B,  _ = self.D(x[aa].detach(), cls_idx[BB])
        pred_fake_A,  _ = self.D(y[aa].detach(), cls_idx[AA])
        pred_fake_B,  _ = self.D(y[aa].detach(), cls_idx[BB])

        loss_D_adv = D_adv_loss(pred_real, real=True) + D_adv_loss(pred_fake, real=False)       
        loss_D_adv_itp = D_adv_loss(pred_real_A, real=True, w=1-a) + D_adv_loss(pred_real_B, real=True, w=a) + \
                         D_adv_loss(pred_fake_A, real=False, w=1-a) + D_adv_loss(pred_fake_B, real=False, w=a)

        loss_D = self.adv_w * loss_D_adv + \
                 self.gp_w * loss_D_gp + \
                 self.adv_w * loss_D_adv_itp

        if train == True:
            self.D_optimizer.zero_grad()
            loss_D.requres_grad = True
            loss_D.backward()
            self.D_optimizer.step()

        ## train G
        pred_fake, _ = self.D(y[AA+BB+BA+AB+rAA+rBB], cls_idx[AA+BB+AA+BB+AA+BB])
        pred_fake_A, _ = self.D(y[aa], cls_idx[AA])
        pred_fake_B, _ = self.D(y[aa], cls_idx[BB])        
       
        loss_G_adv = G_adv_loss(pred_fake)
        loss_G_adv_itp = G_adv_loss(pred_fake_A, w=1-a) + G_adv_loss(pred_fake_B, w=a)
        loss_G_rec = G_rec_loss(x[AA+BB+AA+BB], y[AA+BB+rAA+rBB])
        loss_G_rec_itp = G_rec_loss(x[aa], y[aa])

        loss_G = self.adv_w * loss_G_adv + \
                 self.rec_w * loss_G_rec + \
                 self.adv_w * loss_G_adv_itp + \
                 self.rec_w * loss_G_rec_itp                 

        if train == True:
            self.G_optimizer.zero_grad()
            loss_G.backward()
            self.G_optimizer.step()

        # loss
        losses = \
        loss_G, loss_D, \
        loss_G_adv, loss_G_adv_itp, \
        loss_G_rec, loss_G_rec_itp, \
        loss_D_adv, loss_D_adv_itp, loss_D_gp

        if train == False:
            losses = (y.clone(), ) + losses
        return losses

    def test_model(self, data):
        self.eval()
        out = self.train_model(data, train=False)
        self.train()
        return out