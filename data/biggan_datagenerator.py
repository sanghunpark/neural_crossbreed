# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# My library
from utils.utils import get_device

# BigGAN
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
import nltk
nltk.download('wordnet')

import numpy as np
import random

def lerp(z_A, z_B, t):
    z_t = (1-t)*z_A + t*z_B
    return z_t

def slerp(z_A, z_B, t, eps=1e-20):
    cos_val = (z_A * z_B).sum(dim=1, keepdim=True)
    temp_z_A = z_A.pow(2).sum(dim=1, keepdim=True).sqrt()
    temp_z_B = z_B.pow(2).sum(dim=1, keepdim=True).sqrt()    
    cos_val = cos_val / z_A.pow(2).sum(dim=1, keepdim=True).sqrt()
    cos_val = cos_val / z_B.pow(2).sum(dim=1, keepdim=True).sqrt()
    cos_val = torch.clamp(cos_val, min=-1, max=1)
    theta = torch.acos(cos_val)
    s1 = torch.sin((1-t)*(theta+eps))/(torch.sin(theta)+eps)
    s2 = torch.sin(t*(theta+eps))/(torch.sin(theta)+eps)
    z_t = s1*z_A + s2*z_B
    return z_t

class BigGanData(Dataset):
    def __init__(self, args, batch_size, n_inter=100, transform=None, interp_type='lerp'):
        self.args = args     
        self.n_inter = n_inter
        self.transform = transform
        self.set_interp_func(interp_type)        
        self.batch_size = batch_size
        
        # indices
        global A
        global B
        global N    
        A = list(range(0,batch_size))
        B = list(range(batch_size,2*batch_size))
        N = 3 # frame number in a dataset

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
         if self.n < self.n_inter:
             item = self.__getitem__(self.n)
             self.n += 1
             return item
         else:
             raise StopIteration()
        
    def __len__(self):
        return self.n_inter

    def __getitem__(self, index):
        output, a, class_ab = self.generate_batch_morphing_dataset(
            batch_size=self.batch_size,
            interp_noise=True, interp_class=True,
            truncation=self.truncation)
        
        # transform
        if self.transform is not None:
            output = output.view(N * self.batch_size,  3, output.size(-2), output.size(-1)).cpu()
            output = [self.transform(x) for x in output]
            output = torch.stack(output)
            output = output.view(N, self.batch_size, 3, output.size(-2), output.size(-1)).to(self.device)
        
        return {'x_A':output[0], 'x_a':output[1], 'x_B':output[2], 'a':a, 'A_idx':class_ab[0], 'B_idx':class_ab[1]}
    
    def get_sample_image(self):
        n = truncated_noise_sample(batch_size=self.batch_size, truncation=self.truncation)
        c_idx = np.random.choice(self.class_list, size=1)
        c = one_hot_from_int(c_idx, batch_size=self.batch_size)
        # np to torch
        n = torch.from_numpy(n).to(self.device)
        c = torch.from_numpy(c).to(self.device)

        with torch.no_grad():
            output = self.big_gan(n, c, self.truncation)
        return output

    def set_class(self, class_list):
        self.class_list = class_list
    
    def set_truncation(self, truncation=0.4):
        self.truncation = truncation
        
    def set_model(self, resolution):
        model_name = 'biggan-deep-' + str(resolution)
        self.big_gan = BigGAN.from_pretrained(model_name)

        args = self.args
        self.device, self.multi_gpu = get_device(args)        
        if self.multi_gpu:
            self.big_gan = nn.DataParallel(self.big_gan, device_ids=list(range(args.gpu_1st, args.gpu_1st + args.ngpu)),
                               output_device=args.gpu_1st)
        self.big_gan.to(self.device)

    def set_interp_func(self, interp='slerp'):
        if interp == 'lerp':
            self.interp_func = lerp
        elif interp == 'slerp':
            self.interp_func = slerp

    def generate_class_AB(self, class_A, class_B, batch_size=0):
        if batch_size == 0:
            self.class_A = one_hot_from_int(class_A)
            self.class_B = one_hot_from_int(class_B)
        else: # for translation dataset
            class_A = one_hot_from_int(class_A, batch_size=batch_size)
            class_B = one_hot_from_int(class_B, batch_size=batch_size)

            # generate class -> BxN(1)x1000
            c_A = torch.from_numpy(class_A).unsqueeze(0)
            c_B = torch.from_numpy(class_B).unsqueeze(0)
            c_AB = torch.cat([c_A, c_B], dim=0).to(self.device)
            self.c_AB = c_AB.view(2 * batch_size, -1) # (B*2)x1000

       # for supervised translation dataset
    def generate_batch_translation_dataset(self, batch_size, truncation=0.4):        
        # generate noise -> BxN(2)x128
        noise_A = truncated_noise_sample(batch_size=batch_size, truncation=truncation)
        noise_B = truncated_noise_sample(batch_size=batch_size, truncation=truncation)
        n_A = torch.from_numpy(noise_A).unsqueeze(0)
        n_B = torch.from_numpy(noise_B).unsqueeze(0)
        n_AB = torch.cat([n_A, n_B], dim=0).to(self.device)
        n_AB = n_AB.view((N//2) * batch_size, -1) # (B*2)x128

        # generate class -> BxN(2)x1000
        idx_list = range(len(self.class_list))
        idx_ab = [random.sample(idx_list, 2) for i in range(batch_size)] # 2 pair idx
        class_ab =  [[self.class_list[idx_ab[i][0]], self.class_list[idx_ab[i][1]]] for i in range(len(idx_ab))]
        class_ab = np.array(class_ab).transpose()
        class_A = one_hot_from_int(class_ab[0], batch_size=batch_size) # Bx1000 numpy array
        class_B = one_hot_from_int(class_ab[1], batch_size=batch_size)        
        c_A = torch.from_numpy(class_A).unsqueeze(0)
        c_B = torch.from_numpy(class_B).unsqueeze(0)
        c_AB = torch.cat([c_A, c_B], dim=0).to(self.device)
        c_AB = c_AB.view((N//2) * batch_size, -1) # (B*2)x1000

        with torch.no_grad():
            output = self.big_gan(n_AB[A+B+A+B], c_AB[A+B+B+A], truncation)

        # generate class idx
        idx_ab = torch.from_numpy(np.array(idx_ab).transpose()) # 2xB
        return output.view(N, batch_size, 3, output.size(2), output.size(3)), idx_ab.long()

    # for supervised morphing dataset
    def generate_batch_morphing_dataset(
        self, batch_size,
        interp_noise=True, interp_class=True,
        truncation=0.4):

        # generate noise -> BxN(3)x128
        noise_A = truncated_noise_sample(batch_size=batch_size, truncation=truncation)
        noise_B = truncated_noise_sample(batch_size=batch_size, truncation=truncation) if interp_noise else 0
        n_A = torch.from_numpy(noise_A).expand(N, batch_size, -1)
        n_B = torch.from_numpy(noise_B).expand(N, batch_size, -1) if interp_noise else  n_A.clone()

        # generate class -> BxN(3)x1000
        idx_list = range(len(self.class_list))
        idx_ab = [random.sample(idx_list, 2) for i in range(batch_size)] # 2 pair idx
        class_ab =  [[self.class_list[idx_ab[i][0]], self.class_list[idx_ab[i][1]]] for i in range(len(idx_ab))]
        class_ab = np.array(class_ab).transpose()
        class_A = one_hot_from_int(class_ab[0], batch_size=batch_size) # Bx1000 numpy array
        class_B = one_hot_from_int(class_ab[1], batch_size=batch_size)

        c_A = torch.from_numpy(class_A).expand(N, batch_size, -1)
        c_B = torch.from_numpy(class_B).expand(N, batch_size, -1) if interp_class else c_A.clone()
        
        # # sample index for intermediate image
        # idx_0 = random.randint(0, n_frames-i_frames)
        # idx_1 = idx_0 + i_frames - 1
        # idx_i = random.randint(idx_0, idx_1) # sample index for intermediate image, including end points

        # interpolation -> BxN(3)xZ
        # t = torch.linspace(start=0, end=1.0, steps=n_frames)
        # t = t[[idx_0, idx_i, idx_1]].unsqueeze(1).unsqueeze(1).expand(3, batch_size, 1)
        a = torch.cat( [torch.zeros(1, batch_size, 1), torch.rand(1, batch_size, 1), torch.ones(1, batch_size, 1)])         
        n = self.interp_func(n_A, n_B, a).to(self.device)
        c = self.interp_func(c_A, c_B, a).to(self.device)

        n = n.view(N * batch_size, -1) # (B*N(3))xZ
        c = c.view(N * batch_size, -1)
        with torch.no_grad():
            output = self.big_gan(n, c, truncation)        
        
        # generate class idx
        idx_ab = torch.from_numpy(np.array(idx_ab).transpose()) # 2xB
        return output.view(N, batch_size, 3, output.size(2), output.size(3)), a[1].squeeze(dim=1), idx_ab.long()
                
    def generate_noise_AB(self, truncation):
        self.noise_A = truncated_noise_sample(truncation=truncation)
        self.noise_B = truncated_noise_sample(truncation=truncation)
    
    def generate_dataset(self, n_frames=5, interp_noise=True, interp_class=True, truncation=0.4):
        # generate 2 noises & 2 classes
        n_A = torch.from_numpy(self.noise_A).expand(n_frames, -1)
        n_B = torch.from_numpy(self.noise_B).expand(n_frames, -1) if interp_noise else  n_A.clone()
        
        c_A = torch.from_numpy(self.class_A).expand(n_frames, -1)
        c_B = torch.from_numpy(self.class_B).expand(n_frames, -1) if interp_class else c_A.clone()

        # interpolation
        t = torch.linspace(start=0, end=1.0, steps=n_frames).unsqueeze(1)
        n = self.interp_func(n_A, n_B, t).to(self.device)
        c = self.interp_func(c_A, c_B, t).to(self.device)

        with torch.no_grad():
            output = self.big_gan(n, c, truncation)
        return output
        
    def generate_noise(self, truncation=0.4):
        self.noise_vector = truncated_noise_sample(truncation=truncation)
        self.noise_vector = torch.from_numpy(self.noise_vector)
        self.noise_vector = self.noise_vector.to(self.device)
    
    def generate_t(self, class_A=0, class_B=1, class_t=0.5, truncation=0.4):
        self.class_vector_A = torch.from_numpy(one_hot_from_int(class_A))
        self.class_vector_B = torch.from_numpy(one_hot_from_int(class_B))

        # Interpolation
        class_vector_t = lerp(self.class_vector_A, self.class_vector_B, torch.tensor(class_t, dtype=torch.float32))
        
        # GPU.
        class_vector_t = class_vector_t.to(self.device)        

        # Generate images
        with torch.no_grad():
            output = self.big_gan(self.noise_vector, class_vector_t, truncation=0.4)
        # save_as_images(output)        
        return output