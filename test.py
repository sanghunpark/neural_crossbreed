from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# System
import os
import functools

# PyTorch
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.utils as vutils


# My library
from utils.utils import get_config
from utils.utils import get_device
from utils.logger import Logger
from model.neural_crossbreed import NeuralCrossbreed
from utils.func import normalize_tensor

# Etc
from PIL import Image


def options():
    import argparse
    parser = argparse.ArgumentParser(description='NeuralCrossbreed Arguments')    
    parser.add_argument('-c','--config', type=str, default='./config.yaml', help='e.g.> --config=\'./config.yaml\'')
    parser.add_argument('-n','--ngpu', type=int, help='e.g.> --ngpu=1', default=None)
    parser.add_argument('-g','--gpu_1st', type=int, help='e.g.> --gpu_1st=0', default=0)

    parser.add_argument('-xa','--input_a', type=str, default='./sample_images/dog5.png', help='e.g.> --input_a=\'./sample_images/dog2.png\'')
    parser.add_argument('-xb','--input_b', type=str, default='./sample_images/dog3.png', help='e.g.> --input_b=\'./sample_images/dog2.png\'')
    parser.add_argument('-i','--niter', type=int, help='e.g.> --niter=5', default=5)
    parser.add_argument('-d','--disentangled', action='store_true')
    parser.add_argument('-t','--tau', type=float, help='e.g.> --tau=0.3', default=0.3)

    args = parser.parse_args()
    print(args)
    config = get_config(args.config)
    global trans
    trans = transforms.Compose([
        transforms.Resize(config['img_size']),
        transforms.CenterCrop(config['img_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    return args, config

def load_image(path):
    global trans
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return trans(img).unsqueeze(0).to(device)

def truncation(a_c, a_s, tau):
    tau = torch.tensor([tau])
    new_a_c = a_c + tau * (a_s - a_c) / 2
    new_a_s = a_s + tau * (a_c - a_s) / 2
    return new_a_c, new_a_s

def test(args, config):
    root_dir = config['root_dir']    
    out_path = config['out_path']
    out_dir = os.path.dirname(out_path)
    path, ext = os.path.splitext(out_path)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    global device
    n_images = args.niter + 2

    x_A = load_image(args.input_a)
    x_B = load_image(args.input_b)

    class_list = range(151, 268+1) # dog classes, (https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)
    n_class = len(class_list)

    # set networks
    NC = NeuralCrossbreed(args, config, n_class)

    # load model
    log_dir = root_dir + 'logs/'
    checkpoint_dir = os.path.join(root_dir, config['checkpoints'])
    logger = Logger(args=args,
                config=config,
                model=NC,
                log_path = log_dir,
                checkpoint_path = checkpoint_dir)
    load_epoch, it = logger.load()
    
    if args.disentangled:
        a_c = torch.linspace(start=0, end=1.0, steps=n_images) # N
        a_c = a_c.unsqueeze(0).expand(n_images, -1) # NxN
        a_s = a_c.transpose(0,1)
        a_c = a_c.contiguous().view(n_images*n_images)
        a_s = a_s.contiguous().view(n_images*n_images)
        a_c_, a_s_ = truncation(a_c, a_s, args.tau)
        # y_a = torch.zeros(n_images*n_images, 3, config['img_size'], config['img_size'])
        for i in range(n_images*n_images):
            y_a = NC.G(x_A, x_B, train=False,
                            a_c=a_c_[i].to(device).unsqueeze(0),
                            a_s=a_s_[i].to(device).unsqueeze(0))
            file_path = path + '_' + str(a_s[i]) + '_' + str(a_c[i]) + ext
            save_image(normalize_tensor(y_a), file_path)
    else: # basic transition
        a = torch.linspace(start=0, end=1.0, steps=n_images)
        # y_a = torch.zeros(n_images, 3, config['img_size'], config['img_size'])
        for i in range(n_images):
            y_a = NC.G(x_A, x_B, train=False,
                        a_c=a[i].to(device).unsqueeze(0),
                        a_s=a[i].to(device).unsqueeze(0))
            file_path = path + '_' + str(a[i]) + ext
            save_image(normalize_tensor(y_a), file_path)
    
    save_image(normalize_tensor(x_A), out_dir + '/x_A' + ext)
    save_image(normalize_tensor(x_B), out_dir + '/x_B' + ext)
    print('Generated images are saved at %s' % out_dir)

    # y_a = vutils.make_grid(y_a,
    #                 nrow=n_images,
    #                 normalize=True,
    #                 scale_each=True)
    # save_image(y_a, out_path)


if __name__ == '__main__':
    args, config = options()
    device, _ = get_device(args) 
    test(args, config)