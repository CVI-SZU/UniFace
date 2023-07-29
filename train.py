''' Implementation for the training process of Network '''

import os
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import warning
# warning.filterwarning('ignore')

from torch.optim.lr_scheduler import MultiStepLR
from dataset import MXFaceDataset
from backbone import LResNet50EIR

import random
def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Hyperparameters():
    def __init__(self):
        self.cuda = True
        self.cudnn = False
        self.visible_devices = "0"
        self.fp16 = True
        self.base_lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.gamma = 0.1
        self.resume = None
        # self.resume = './Models-LResNet50EIR/LResNet50EIR_1th_checkpoint.tar'
        self.finetune = None
        # self.finetune = './Models-LResNet50EIR/LResNet50EIR_2th_epoch.pth'
        self.data_path = '/home/jason/Datasets/InsightFace/faces_webface'
        self.img_size = [112, 112]
        self.train_batch_size = 128
        self.bs_mult = 4
        self.drop_last = True
        self.steps = [16, 24]
        self.start_epoch = 0
        self.epochs = 28
        self.warmups = 0
        self.display = 100.0
        self.workers = 2
        self.num_classes = 10572
        self.model_name = 'LResNet50EIR'
        self.model_dir = './Models-LResNet50EIR'
        self.log_dir = './log-LResNet50EIR'


def main():
    global params

    ''' optionally finetune from a pre-trained model '''
    if params.finetune is not None:
        model = torch.load(params.finetune)
        print("=> load pre-trained model '{}'\n".format(params.finetune))
    else:
        ''' create Network for face recognition '''
        model = eval(params.model_name)(num_classes=params.num_classes)
    print(model)
    print()

    model_params = []
    for name, value in model.named_parameters():
        model_params += [{'params': value}]

    ''' define loss function and optimizer '''
    optimizer = torch.optim.SGD(model_params, params.base_lr, momentum=params.momentum, weight_decay=params.weight_decay, nesterov=False)

    if params.cuda:
        # model = nn.DataParallel(model).cuda()
        model = model.cuda()
        # cudnn.benchmark = params.cudnn
        # net = model.module
        net = model
    else:
        net = model.cpu()

    ''' optionally resume from a checkpoint '''
    if params.resume is not None:
        checkpoint = torch.load(params.resume)
        params.start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        print("=> resume from checkpoint '{}'\n".format(params.resume))

    ''' load image '''
    train_set = MXFaceDataset(root_dir=params.data_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=params.train_batch_size*params.bs_mult, shuffle=True,
        num_workers=params.workers, pin_memory=True, drop_last=params.drop_last,
        worker_init_fn=seed_worker, prefetch_factor=2, persistent_workers=True)

    if os.path.exists(params.model_dir) is False:
        os.makedirs(params.model_dir)

    scheduler = MultiStepLR(optimizer, milestones=[v + params.warmups for v in params.steps], gamma=params.gamma, last_epoch=params.start_epoch-1)

    if params.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(params.start_epoch, params.epochs):

        lr_scale = 1.0 * np.clip(max(epoch, 1e-6) * 10.0 / max(params.warmups, 1e-6), 1, 10) / 10
        loss_scale = 1.0 / params.bs_mult

        for param_group, lr in zip(optimizer.param_groups, scheduler._get_closed_form_lr()):
            param_group['lr'] = lr * lr_scale

        real_lr = optimizer.param_groups[-1]['lr']
        virtual_epoch = epoch + 1
        print('Epoch: {}\n'.format(virtual_epoch))

        ''' train for one epoch '''
        train(train_loader, model, optimizer, epoch, loss_scale, scaler)
        scheduler.step()

        save_name = params.model_dir + '/' + params.model_name + '_' + str(virtual_epoch) + 'th_epoch.pth'
        torch.save(net, save_name)
        save_name = params.model_dir + '/' + params.model_name + '_' + str(virtual_epoch) + 'th_checkpoint.tar'
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(), 'optimizer' : optimizer.state_dict()}, save_name)


def train(train_loader, model, optimizer, epoch, loss_scale, scaler):
    global params
    model.train()
    if params.cuda:
        # net = model.module
        net = model
    else:
        net = model

    acc = 0; loss = 0; loss_ = 0; count = 0
    for i, (data_batch, label_batch) in enumerate(train_loader):
        real_lr = optimizer.param_groups[-1]['lr']
        data_splits = data_batch.split(params.train_batch_size, dim=0)
        label_splits = label_batch.split(params.train_batch_size, dim=0)
        positive = torch.unique(label_batch, sorted=True)
        perm = torch.randperm(params.num_classes)
        perm[positive] = 0
        indices = torch.topk(perm, k=int(params.num_classes*net.loss.r), largest=False)[1]
        partial_index = indices.sort()[0]
        if params.cuda:
            partial_index = partial_index.cuda()

        for j in range(len(label_splits)):
            data = data_splits[j]
            label = label_splits[j]
            if params.cuda:
                data, label = data.cuda(), label.cuda()

            ''' forward and compute loss '''
            if scaler is None:
                face_loss = model(data, label, partial_index)
            else:
                with torch.cuda.amp.autocast():
                    face_loss = model(data, label, partial_index)

            batch_scale = 1.0 * label.size(0) / params.train_batch_size

            ''' compute gradient and do SGD step '''
            if j == 0:
                optimizer.zero_grad()
                if optimizer.state[optimizer.param_groups[-1]["params"][0]]:
                    optimizer.state[optimizer.param_groups[-1]["params"][0]]["momentum_buffer"][:] = 0
                    optimizer.state[optimizer.param_groups[-1]["params"][0]]["momentum_buffer"][partial_index] = model.loss.weight_mom[partial_index]
            if (j + 1) == len(label_splits):
                if scaler is None:
                    (face_loss * loss_scale * batch_scale).backward()
                else:
                    scaler.scale(face_loss * loss_scale * batch_scale).backward()

                if scaler is None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                else:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                net.restrict_weights()
                model.loss.weight_mom[partial_index] = optimizer.state[optimizer.param_groups[-1]["params"][0]]["momentum_buffer"][partial_index]
            else:
                if scaler is None:
                    (face_loss * loss_scale).backward()
                else:
                    scaler.scale(face_loss * loss_scale).backward()

            loss += face_loss.data
            count += 1

            if (i % params.display == 0 or (i + 1) == len(train_loader)) and (j + 1) == len(label_splits):
                loss /= count

                # print(net.loss.bias.data)
                print('{}, Iteration: {} ({}/{}={:.0f}%)  loss: {:.4f}  lr: {:.4f}\n'.format(
                      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i, params.train_batch_size*params.bs_mult*i+label_batch.size(0),
                      len(train_loader.dataset), 100.0 * (i + 1) / len(train_loader), loss, real_lr))

                loss = 0
                count = 0


if __name__ == '__main__':
    params = Hyperparameters()
    print('Hyperparameters:')
    print('  cuda: {}\n  cudnn: {}\n  device_id: {}\n  fp16: {}\n  base_lr: {}\n  momentum: {}\n  weight_decay: {}\n  gamma: {}\n'
          '  data_path: {}\n  img_size: {} \n  batch_size: {}\n  bs_mult: {}\n  drop_last: {}\n  steps: {}\n  epochs: {}\n'
          '  warmups: {}\n  workers: {}\n  num_classes: {}\n  model_name: {}\n  model_dir: {}\n  log_dir: {}\n'.format(
          params.cuda, params.cudnn, params.visible_devices, params.fp16, params.base_lr, params.momentum, params.weight_decay, params.gamma,
          params.data_path, params.img_size, params.train_batch_size, params.bs_mult, params.drop_last, params.steps, params.epochs,
          params.warmups, params.workers, params.num_classes, params.model_name, params.model_dir, params.log_dir))

    os.environ["CUDA_VISIBLE_DEVICES"] = params.visible_devices

    setup_seed(seed=1, cuda_deterministic=True)
    main()
