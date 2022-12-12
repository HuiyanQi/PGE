import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import json

# import sys
# sys.path.append("./")

import configs
import models.resnet as resnet
import models.vgg as vgg
import models.swin as swin
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from io_utils import parse_args
import pdb



def train_randinit(params):
    if 'resnet' in params.model:
        arch = getattr(resnet,params.model)
    if 'vgg' in params.model:
        arch = getattr(vgg,params.model)
    if 'swin' in params.model:
        arch = getattr(swin,params.model)
    if params.method == 'protonet':
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params   = dict(n_way = params.train_n_way, n_support = params.n_shot)
        base_datamgr            = SetDataManager(dataset_name = params.dataset, n_query = n_query,  **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        model = ProtoNet(arch, **train_few_shot_params)
        model = model.cuda()
        model.zero_grad()
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError('Only support protonet!')

    start_epoch = params.start_epoch
    stop_epoch  = params.stop_epoch

    count = 0
    for epoch in range(start_epoch,stop_epoch+1):
        #train
        print_freq = 20
        avg_loss=0
        for i, (x,_ ) in enumerate(base_loader):
            new_model = None           
            if params.method == 'protonet':
                new_model = ProtoNet(arch, **train_few_shot_params)
                new_model = new_model.cuda()

            new_model.n_query = x.size(1) - new_model.n_support   # x.size- [5,21,3,84,84]    
            if new_model.change_way:
                new_model.n_way  = x.size(0)
            
            new_model.zero_grad()
            scores = new_model.set_forward(x,get_features=False)

            y_query = torch.from_numpy(np.repeat(range(new_model.n_way ), new_model.n_query ))
            y_query = Variable(y_query.cuda())
            loss    = criterion(scores, y_query)

            loss.backward()

            # copy grads from new model to model
            for p,new_p in zip(model.parameters(),new_model.parameters()):
                p.data = (p.data*count + new_p.grad)/(count+1)
            count += 1

            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Count {:d}'.format(epoch, i, len(base_loader), count))
       
        if epoch==stop_epoch:
            outfile = os.path.join(params.save_dir, model, dataset + '.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    base_file = configs.data_dir[params.dataset] + 'train.json'

    if params.stop_epoch == -1: 
        if params.n_shot == 1:
            params.stop_epoch = 600
        elif params.n_shot == 5:
            params.stop_epoch = 400
        else:
            params.stop_epoch = 600 #default

    train_randinit(params)