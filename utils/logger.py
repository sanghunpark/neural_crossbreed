# system
import os
import functools
import glob

# PyTorch
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter # for SummaryWriter.add_hparams()
import torchvision.utils as vutils

# My library
from utils.utils import get_device

# import flowiz as fz
from tqdm import tqdm

def sample_fn(G, sample_iter, device):
    with torch.no_grad():
        x_A = next(sample_iter)[0]
        x_B = next(sample_iter)[0]
        a = torch.Tensor([0.5]).to(device).expand(x_A.size(0))
        x_A = x_A.to(device)
        x_B = x_B.to(device)
        out = G(x_A, x_B, a_c =a, a_s= a, train=False)
    return out, None, None

class Logger:
    def __init__(self, args, config, model, log_path, checkpoint_path, metric_fn=None, sample_loader=None, fine_tune=False):
        # path
        self.device, self.multi_gpu = get_device(args)
        self.checkpoint_dir = os.path.dirname(checkpoint_path)
        self.checkpoint_path = checkpoint_path      
        self.fine_tune = fine_tune
        self.metric_fn = metric_fn
        self.sample_loader = sample_loader
        self.model = model
        self.metric = config['which_best']
        # self.checkpoint_file_name = 'checkpoint'
        # self.checkpoint_ext = '.pt'
        
        if self.metric_fn is not None:
            self.best_IS = 0
            self.best_IS_std = -1
            self.best_FID = 99999

        checkpoint_dir_name = os.path.basename(self.checkpoint_dir)
        self.logger = SummaryWriter(log_path + checkpoint_dir_name)

    def _model(self, model):
        return model.module if self.multi_gpu else model # convension between multi-GPU and single-GPU(or CPU)

    def save(self, it, epoch, mode='model'):
        # create a directory for checkpoints
        if not os.path.exists(self.checkpoint_dir):
            try:
                original_umask = os.umask(0)
                os.makedirs(self.checkpoint_dir, mode=0o777)
            finally:
                os.umask(original_umask)

        # # save optimizers
        # if epoch != -1:
        #     path = os.path.join(self.checkpoint_dir, 'optimizer.pt')
        #     torch.save({'G_optimizer' : self.model.G_optimizer.state_dict(),
        #                 'D_optimizer' : self.model.D_optimizer.state_dict()}, path)

        # save model
        path = os.path.join(self.checkpoint_dir, mode + '_%08d.pt' % it)
            
        torch.save({'G' : self._model(self.model.G).state_dict(),
                    'D' : self._model(self.model.D).state_dict(),
                    'iter' : it,
                    'epoch' : epoch}, path)
        tqdm.write('saved model at iteration %d , path: %s' % (it, path))

        # remove previous models
        model_list = glob.glob(self.checkpoint_dir + '/*.pt')
        for file_path in model_list:
            if path != file_path and mode in file_path:
                os.remove(file_path)

        # save best IS, FID
        if mode == 'best' and self.metric_fn is not None:            
            with open(os.path.join(self.checkpoint_dir, 'metric.txt'), 'w+') as f:
                f.write('IS:\n')
                f.write(str(self.best_IS) + '\n')
                f.write('IS_std:\n')
                f.write(str(self.best_IS_std) + '\n')
                f.write('FID:\n')
                f.write(str(self.best_FID) + '\n')
    
    def load(self):
        if not os.path.isfile(self.checkpoint_path):
            print('[**Warning**] There is no checkpoint to load')
            print(self.checkpoint_path)
            return 0, 0

        # # load optimizers
        # path = os.path.join(self.checkpoint_dir, 'optimizer.pt')
        # checkpoint = torch.load(path, map_location=self.model.device)
        # self.model.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
        # self.model.D_optimizer.load_state_dict(checkpoint['D_optimizer'])

        # load models
        checkpoint = torch.load(self.checkpoint_path, map_location=self.model.device)
        self._model(self.model.G).load_state_dict(checkpoint['G'])
        self._model(self.model.D).load_state_dict(checkpoint['D'])

        print('[**Notice**] Succeeded to load')
        print(self.checkpoint_path)
        if self.metric_fn is not None:
            self.compute_metric(checkpoint['iter'], checkpoint['epoch'], save=True)
        return checkpoint['epoch'], checkpoint['iter']

    def log(self, mode, losses, it):
        loss_G, loss_D, \
        loss_G_adv, loss_G_adv_itp, \
        loss_G_rec, loss_G_rec_itp, \
        loss_D_adv, loss_D_adv_itp, loss_D_gp = losses
        self.logger.add_scalars(mode+'/loss', {'loss_G' : loss_G,
                                            'loss_D' : loss_D}, it+1)
    
        self.logger.add_scalars(mode+'/loss_G', {'loss_G_adv' : loss_G_adv,
                                                'loss_G_adv_itp' : loss_G_adv_itp,                                                    
                                                'loss_G_rec' : loss_G_rec,
                                                'loss_G_rec_itp' : loss_G_rec_itp,
                                                }, it+1)
        self.logger.add_scalars(mode+'/loss_D', {'loss_D_adv' : loss_D_adv,
                                                'loss_D_adv_itp' : loss_D_adv_itp,
                                                'loss_D_gp' : loss_D_gp,
                                                }, it+1)

            

    def iterate(self, data, test_data, i, it, epoch, config, no_log=False):
        # train
        losses = self.model.train_model(data)
        if no_log:
            return
        # print status            
        tqdm.write(' [%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch+1, config['max_epochs'], i+1, config['n_inter'], losses[0].item(), losses[1].item()) )          

        # log losses
        if (it+1) % config['log_freq'] == 0:
            self.log('train', losses, it)
            tqdm.write('logged losses at iteration %d' % (it+1))

        # log test losses & save model
        if (it+1) % config['checkpoint_freq'] == 0:
            with torch.no_grad():
                out = self.model.test_model(test_data)
                y_a = out[0]
                losses = out[1:]
                self.log('test', losses, it)
                y_a_grid = vutils.make_grid(y_a, normalize=True)
                self.logger.add_image('y_a', y_a_grid, it+1)

            # get metric
            if self.metric_fn is not None:
                IS_mean, IS_std, FID, \
                LPIPS_mean, LPIPS_std, \
                MSE_mean, MSE_std, \
                PSNR_mean, PSNR_std, \
                SSIM_mean, SSIM_std= self.compute_metric(it, epoch, save=True)
                self.logger.add_scalars('test/metric', {'IS' : IS_mean,
                                                        'FID' : FID}, it+1)                
            self.save(it+1, epoch, 'model')
            
    def compute_metric(self, it, epoch, save=True):
        sample_iter = iter(self.sample_loader)
        self.n_sample = len(sample_iter)    
        self.sample = functools.partial(sample_fn,
                                G=self.model.G,
                                sample_iter=sample_iter,
                                device=self.device)

        IS_mean, IS_std, FID, \
        LPIPS_mean, LPIPS_std, \
        MSE_mean, MSE_std, \
        PSNR_mean, PSNR_std, \
        SSIM_mean, SSIM_std = self.metric_fn(self.sample, 
                                            int(self.n_sample / 2),
                                            num_splits=10)
        if ((self.metric == 'IS' and IS_mean > self.best_IS) or (self.metric == 'FID' and FID < self.best_FID)):                            
            self.best_IS = IS_mean
            self.best_IS_std = IS_std
            self.best_FID = FID
            if save: self.save(it+1, epoch, mode='best')
            tqdm.write('Best IS: %.4f' % self.best_IS )
            tqdm.write('Best FID: %.4f' % self.best_FID )
        return IS_mean, IS_std, FID, LPIPS_mean, LPIPS_std

    def log_test_data(self, data):
        if self.fine_tune:
            x_A, x_B, _, _, _ = self.model.get_data(data)
        else:
            x_A, x_B, x_a, _, _, _ = self.model.get_data(data)
            x_a_grid = vutils.make_grid(x_a, normalize=True)
            self.logger.add_image('x_a', x_a_grid, 0)

        x_A_grid = vutils.make_grid(x_A, normalize=True)
        x_B_grid = vutils.make_grid(x_B, normalize=True)
       
        self.logger.add_image('x_A', x_A_grid, 0)
        self.logger.add_image('x_B', x_B_grid, 0)

        # for i in range(t.size(0)):
        #     self.logger.add_hparams({'i':i}, {'t':t[i]})