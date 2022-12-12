import torch
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from methods.protonet import ProtoNet
from torchvision import models
from io_utils import parse_args
import os
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import torchvision
import pdb
import xlwt
import xlrd
from xlutils.copy import copy



def m_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grad1', type=str, default = '')
    parser.add_argument('--grad2', type=str, default = '')                                                
    args = parser.parse_args()
    return args


def get_dis(parameters1, parameters2):
    smi_count = 0
    smi_sum = 0
    for (k1,v1), (k2,v2) in zip(parameters1.items(),parameters2.items()):
        if 'fc' not in k1 and 'fc' not in k2 and 'running_mean'not in k1 and 'running_mean' not in k2 and 'running_var' not in k1 and 'running_var' not in k2 and 'num_batches_tracked' not in k1 and 'num_batches_tracked' not in k2 and 'mask' not in k1 and 'mask' not in k2 and 'mlp_head'not in k1 and 'mlp_head' not in k2 :
            assert k1 == k2
            w1 = v1.flatten()
            w2 = v2.flatten()
            if smi_count == 0:
                param1 = w1
                param2 = w2
            else:
                param1 = torch.cat([param1, w1], 0)
                param2 = torch.cat([param2, w2], 0)


            smi_count += 1

    x=torch.mul(param1,param2)
    a1 = torch.norm(param1)
    a2 = torch.norm(param2)
    similarity = torch.norm(param1-param2)/(a1*a2)
    
    return similarity.item()




def calculate(args):

    parameters1 = torch.load(args.grad1)
    parameters2 = torch.load(args.grad2)
    dis = get_dis(parameters1['state'], parameters2['state'])

    return dis
  
if __name__ == '__main__':
    params = m_parse()
    distance = calculate(params)
    print(distance)





