from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# System
import os

# PyTorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# My library
from utils.utils import get_config
from utils.logger import Logger
from data.biggan_datagenerator import BigGanData
from model.neural_crossbreed import NeuralCrossbreed

# Etc.
from tqdm import tqdm, trange

def options():
    import argparse
    parser = argparse.ArgumentParser(description='NeuralCrossbreed Arguments')    
    parser.add_argument('-c','--config', type=str, default='./config.yaml', help='e.g.> --config=\'./config.yaml\'')
    parser.add_argument('-n','--ngpu', type=int, help='e.g.> --ngpu=1', default=None)
    parser.add_argument('-g','--gpu_1st', type=int, help='e.g.> --gpu_1st=0', default=0)
    # parser.add_argument('-m','--metric', action='store_true', help='if specified, use AFHQ dataset for quantitative evaluation)  

    args = parser.parse_args()
    print(args)
    config = get_config(args.config)
    return args, config

def train(args, config):
    root_dir = config['root_dir']

    # set dataset
    trans = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)), # BigGAN produces a image in the range [-1, 1], tensor will be converted to [0, 1] rnage
        transforms.ToPILImage(),
        transforms.Resize(config['img_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # get back tensors back in the range [-1, 1]
    ]) if config['img_size'] != config['biggan_model'] else None

    gan_data = BigGanData(
        args=args,
        batch_size = config['batch_size'],
        n_inter=config['n_inter'],
        transform = trans,
        interp_type='lerp')

    class_list = range(151, 268+1) # dog classes
    n_class = len(class_list)

    gan_data.set_class(class_list)
    gan_data.set_truncation(config['truncation']) # truncation for test dataset
    gan_data.set_model(resolution=config['biggan_model'])

    test_data = gan_data[0]
    test_data['a'].fill_(0.5) # to see interpolated image at a=0.5

    # set networks
    NC = NeuralCrossbreed(args, config, n_class)    
    afhq_loader = [None]
    get_inception_metrics = None
        
    # set a logger and a checkpoint folder
    log_dir = root_dir + 'logs/'
    checkpoint_dir = root_dir + config['checkpoints']
    logger = Logger(args=args,
                    config=config,
                    model=NC,
                    log_path = log_dir,
                    checkpoint_path = checkpoint_dir,
                    metric_fn=get_inception_metrics,
                    sample_loader=afhq_loader[0])
    logger.log_test_data(test_data)

    # train networks
    load_epoch, it = logger.load()
    if it == 0:
        logger.save(-1, -1) # test a save path

    max_epochs = config['max_epochs']
    trunc_step = 1.0 / max_epochs # 0.001
    epoch_bar = tqdm(desc='epoch: ', initial=load_epoch, total=max_epochs, position=2, leave=True)
    iter_num = len(gan_data)
    iter_bar = tqdm(desc='iter: ', total=iter_num, position=3, leave=True)

    for epoch in range(load_epoch, max_epochs):
        iter_bar.reset()
        for i, data in enumerate(gan_data):
            logger.iterate(data, test_data, i, it, epoch, config)

            it += 1
            iter_bar.update()
        epoch_bar.update()

if __name__ == '__main__':
    args, config = options()
    train(args, config)