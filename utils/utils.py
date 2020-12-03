import yaml
import torch

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def get_device(args):
    args.ngpu = torch.cuda.device_count() if(args.ngpu==None) else args.ngpu
    cuda = 'cuda:' + str(args.gpu_1st)
    device = torch.device(cuda if (torch.cuda.is_available() and args.ngpu > 0) else 'cpu')
    multi_gpu = True if (args.ngpu > 1) else False
    
    return device, multi_gpu
