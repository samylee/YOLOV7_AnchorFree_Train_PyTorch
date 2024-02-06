import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import numpy as np
from copy import deepcopy
from torch.cuda import amp

from data.dataset import VOCDataset
from models.YOLOV7 import YOLOV7
from utils.loss import YOLOV7Loss
from utils.utils import load_pretrain_weights, init_seeds


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    cuda = device.type != 'cpu'
    init_seeds(0)
    scaler = amp.GradScaler(enabled=cuda)

    base_lr = 0.01
    lrf = 0.01
    momentum = 0.937
    weight_decay = 0.0005

    warmup_epochs = 3.0
    warmup_momentum = 0.8  # warmup initial momentum
    warmup_bias_lr = 0.1  # warmup initial bias lr

    num_epochs = 150
    batch_size = 8
    C = 20
    net_size = 640
    strides = [8, 16, 32]
    reg_max = 16

    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    weight_decay *= batch_size * accumulate / nbs

    pretrain = 'pretrain/yolov7-u6-samylee.pt'
    train_label_list = 'data/voc0712/train.txt'

    print_freq = 5
    save_freq = 5

    # def model
    yolov7 = YOLOV7(C=C, strides=strides, reg_max=reg_max)
    load_pretrain_weights(yolov7, pretrain)
    yolov7 = yolov7.to(device)
    yolov7.half().float()

    # def loss
    criterion = YOLOV7Loss(C, net_size=net_size, strides=strides, reg_max=reg_max, device=device)

    # def optimizer
    optimizer = torch.optim.SGD(yolov7.parameters(), lr=base_lr, momentum=momentum, nesterov=True)
    lf = lambda x: (1 - x / num_epochs) * (1.0 - lrf) + lrf  # linear
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(yolov7)

    # def dataset
    train_dataset = VOCDataset(train_label_list, net_size=net_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=VOCDataset.collate_fn)

    print('Number of training images: ', len(train_dataset))

    nb = len(train_loader)
    nw = max(round(warmup_epochs * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    last_opt_step = -1

    # train
    for epoch in range(num_epochs):
        yolov7.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(train_loader):
            # Warmup
            ni = i + nb * epoch
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [warmup_momentum, momentum])

            current_lr = get_lr(optimizer)

            inputs = inputs.to(device)
            targets = targets.to(device)

            with amp.autocast(enabled=cuda):
                preds = yolov7(inputs)
                loss, loss_items = criterion(preds, targets)
                total_loss += loss.item()

            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(yolov7.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(yolov7)
                last_opt_step = ni

            # Print current loss.
            if i % print_freq == 0:
                print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, BoxLoss: %.4f, ClsLoss: %.4f, DflLoss: %.4f, Average Loss: %.4f'
                      % (epoch, num_epochs, i, len(train_loader), current_lr,
                         loss_items[0].item(), loss_items[1].item(), loss_items[2].item(), total_loss / (i+1)))

        scheduler.step()
        if epoch % save_freq == 0:
            ckpt = {
                'yolov7': deepcopy(yolov7).half().state_dict(),
                'ema': deepcopy(ema.ema).half().state_dict()
            }
            torch.save(ckpt, 'weights/yolov7_' + str(epoch) + '.pth')

    ckpt = {
        'yolov7': deepcopy(yolov7).half().state_dict(),
        'ema': deepcopy(ema.ema).half().state_dict()
    }
    torch.save(ckpt, 'weights/yolov7_final.pth')