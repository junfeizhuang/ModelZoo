import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *
from config import cfg
from dataset import TinyImageNet
import MobileNetModels, ResNetModels, ResNextModels, EfficientNetModels

def prepare_data_loaders(input_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(70),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
    ])

    trainset = TinyImageNet(train_transform, cfg.train_txt)
    testset = TinyImageNet(train_transform, cfg.val_txt)
    
    trainloader = DataLoader(trainset, batch_size=cfg.train_batch_size, shuffle=True, pin_memory=True, num_workers=cfg.workers)
    testloader = DataLoader(testset, batch_size=cfg.eval_batch_size, shuffle=False, pin_memory=True, num_workers=cfg.workers)
    return trainloader, testloader


def train_eval():
    set_seed()
    logger = create_logger(cfg.log_dir, cfg.arch + '.log')
    train_loader, eval_loader = prepare_data_loaders(cfg.input_size)
    model = EfficientNetModels.__dict__[cfg.arch]()
    with torch.cuda.device(cfg.gpu):
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), \
                                    lr=cfg.lr, \
                                    momentum=cfg.momentum, \
                                    weight_decay=cfg.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, patience=2, mode='min', factor=0.5, threshold=0.0001, min_lr=0)

    
    best_acc = 0
    for epoch in range(cfg.epochs):
        # train
        train_loss = []
        model.train()
        losses = AverageMeter('Loss')
        for idx, (data, label) in enumerate(train_loader):
            with torch.cuda.device(cfg.gpu):
                data, label = data.cuda(), label.cuda()
            pred = model(data)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), data.size(0))
            train_loss.append(loss.detach().cpu().numpy())
            if idx % cfg.print_frep == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\tTrain Loss1: {loss.avg:.5f}'.format(
                    epoch + 1, idx + 1, len(train_loader), loss=losses))
        scheduler.step(np.mean(train_loss))

        # eval
        model.eval()
        top1 = AverageMeter('Top1Acc')
        cnt = 0
        num_eval_batches = 10
        with torch.no_grad():
            for data, label in eval_loader:
                with torch.cuda.device(cfg.gpu):
                    data, label = data.cuda(), label.cuda()
                pred = model(data)
                cnt += 1
                if cnt > num_eval_batches:
                    break
                acc = accuracy(pred, label)
                top1.update(acc[0].item(), data.size(0))

        
        if top1.avg > best_acc:
            best_acc = top1.avg

        cur_lr = get_lr(optimizer)
        logger.info('Epoch: {} lr:{:.4f} current accuray: {top1.avg:.3f}, best accuracy: {best_acc:.3f}'.format(epoch,cur_lr,top1=top1, best_acc=best_acc))

if __name__ == '__main__':
    train_eval()