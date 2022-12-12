import numpy as np
import torch
import os
import glob
import argparse
import models.resnet as resnet
import models.googlenet as googlenet
import torchvision
import models.vgg as vgg





def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='CUB',        help='CUB/miniImagenet/cross/omniglot/cross_char/cifar10')
    parser.add_argument('--model'       , default='resnet18') 
    parser.add_argument('--method'      , default='baseline',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') 
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') 
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') 
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--save_dir',metavar='DIR', default='./pge_grad') 
    if script == 'train':
        parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ')
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=1, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
    else:
       raise ValueError('Unknown script')
        
    return parser.parse_args()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)
