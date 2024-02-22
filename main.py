from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np

import os
import shutil
import argparse
import time
import logging
import sys


import lowbinary.model_bin
import lowbinary.bin_recu as bin_recu
import lowbinary.model_imagenet as model_imagenet
from lowbinary.modules.data import *
import lowbinary.util 
import lowbinary.resnet_bin as model_reactres
import lowbinary.vgg_bin as vgg_bin
import wandb

os.environ['WANDB_CONFIG_DIR'] = "."

def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR training with CPT')
    parser.add_argument('--dir', help='annotate the working directory')
    parser.add_argument('--cmd', choices=['train', 'test'], default='train')
    parser.add_argument('--arch', metavar='ARCH', default='cifar10_resnet_18',choices=['imagenet_resnet_18',"imagenet_resnet_34",'cifar10_resnet_18','cifar10_resnet_20', 'cifar10_vgg_small'])
    parser.add_argument('--dataset', '-d', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100',"imagenet"],
                        help='dataset choice')
    parser.add_argument('--datadir', default='./data/', type=str,
                        help='path to dataset')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--iters', default=600000, type=int,
                        help='number of total iterations (default: 600,000)')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='mini-batch size (default: 512)')
    parser.add_argument('--lr_schedule', default='linear', type=str,
                        help='learning rate schedule')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--print_freq', default=1, type=int,
                        help='print frequency (default: 1)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step_ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warmup', default=0, type=int,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm_up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save_folder', default='save_checkpoints',
                        type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--eval_every', default=10000, type=int,
                        help='evaluate model every (default: 1000) iterations')

    parser.add_argument('--num_bits', default=0, type=int,
                        help='num bits for weight and activation')
    parser.add_argument('--num_grad_bits', default=0, type=int,
                        help='num bits for gradient')
    parser.add_argument('--num_bits_schedule', default=None, type=int, nargs='*',
                        help='schedule for weight/act precision')
    parser.add_argument('--num_grad_bits_schedule', default=None, type=int, nargs='*',
                        help='schedule for grad precision')

    parser.add_argument('--is_cyclic_precision', action='store_true',
                        help='cyclic precision schedule')
    parser.add_argument('--cyclic_num_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for weight/act precision')
    parser.add_argument('--cyclic_num_grad_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for grad precision')
    parser.add_argument('--num_cyclic_period', default=1, type=int,
                        help='number of cyclic period for precision, same for weights/activation and gradients')
    parser.add_argument('--num_cyclic_annealing', default=1, type=int,
                        help='number of cyclic annealing scheduler')
    parser.add_argument('--optimizer', '-opt', type=str, default='sgd',
                        choices=['sgd', 'adam',"adamw"],
                        help='optimizer choice')
    parser.add_argument('--clamping_tau', '-cmptau', type=str, default='yes',
                        choices=['yes', 'no'],
                        help='use clamping')
    args = parser.parse_args()
    return args


def find_conv2d_modules(model):
    conv2d_modules = []

    def find_conv2d_modules_recursive(module):
        for name, sub_module in module.named_children():
            if isinstance(sub_module, bin_recu.BinarizeConv2d):
                conv2d_modules.append(sub_module)
            elif isinstance(sub_module, nn.Module):
                find_conv2d_modules_recursive(sub_module)

    find_conv2d_modules_recursive(model)
    return conv2d_modules

def get_model(arch):
    if arch == "cifar10_resnet_18":
        return model_reactres.resnet18_1w1a(num_classes = 10)
    elif arch == "cifar10_vgg_small":
        return vgg_bin.vgg_small_1w1a(num_classes = 10)
    elif arch == "cifar10_resnet_20":  
        return model_reactres.resnet20_1w1a(num_classes = 10)
    elif arch == "imagenet_resnet_18":
        return model_imagenet.resnet18_1w1a(num_classes = 1000)
    elif arch == "imagenet_resnet_34":
        return model_imagenet.resnet34_1w1a(num_classes = 1000)
    else:
        return None

def run_training(args):
    # create model    
    model = nn.DataParallel(get_model(args.arch)).cuda()
    logging.info(str(model))
    logging.info(str(args.optimizer)+' training')

    best_prec1 = 0
    best_iter = 0


    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])

            
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False

    train_loader = prepare_train_data(dataset=args.dataset,
                                      datadir=args.datadir,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    datadir=args.datadir,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
   

    criterion = nn.CrossEntropyLoss().cuda()
    if args.optimizer == "sgd":
        optimizer =  torch.optim.SGD(model.parameters(), args.lr,  momentum=args.momentum, weight_decay=args.weight_decay)     
    if args.optimizer == "adam":
            optimizer =    torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
            optimizer =  torch.optim.AdamW(model.parameters(), args.lr,  weight_decay=args.weight_decay)
  
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (args.iters -args.warmup*4)/args.num_cyclic_annealing)


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cr = AverageMeter()

    end = time.time()
    conv_modules=[]
    for name, module in model.named_modules():
        if isinstance(module,bin_recu.BinarizeConv2d):
            conv_modules.append(module)
    i = 0# args.start_iter
    while i < args.iters:
        if args.clamping_tau == "yes":
                tau = cpt_tau(i,args)
                for module in conv_modules:
                    module.tau = tau.cuda()
        for input, target in train_loader:
            # measuring data loading time
            data_time.update(time.time() - end)

            model.train()
            if i >= 4 * args.warmup:
                cyclic_period = int((args.iters - 4 * args.warmup) / args.num_cyclic_period)
                cyclic_adjust_precision(args, i- 4 * args.warmup, cyclic_period)
                lr_scheduler.step()
                if i % 10 == 0:
                    logging.info('Iter [{}] learning rate = {}'.format(i, lr_scheduler.get_lr()))
            i += 1

            fw_cost = args.num_bits * args.num_bits / 32 / 32
            eb_cost = args.num_bits * args.num_grad_bits / 32 / 32
            gc_cost = eb_cost
            cr.update((fw_cost + eb_cost + gc_cost) / 3)
            target = target.squeeze().long().cuda()
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            # compute output
            output = model(input_var, args.num_bits, args.num_grad_bits)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, = accuracy(output.data, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print log
            if i % args.print_freq == 0:
                wandb.log({
                    "iter":i,
                   # "tau":tau.item(),
                    "loss":loss.item(),
                    "top1":prec1.item(),
                    "lr":lr_scheduler.optimizer.param_groups[0]['lr'],
                    "cr":(fw_cost + eb_cost + gc_cost) / 3,
                    "num_bits":args.num_bits, 
                    "num_grad_bits":args.num_grad_bits})
                logging.info("Num bit {}\t"
                             "Num grad bit {}\t".format(args.num_bits, args.num_grad_bits))
                logging.info("Iter: [{0}/{1}]\t"
                             "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                             "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                             "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                             "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                             "Training FLOPS ratio: {cr.val:.6f} ({cr.avg:.6f})\t".format(
                    i,
                    args.iters,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    cr=cr
                )
                )


           
            if (i % args.eval_every == 0 and i > 0) or (i == args.iters):
                with torch.no_grad():
                    prec1 = validate(args, test_loader, model, criterion, i)

                is_best = prec1 > best_prec1
                if is_best:
                    best_prec1 = prec1
                    best_iter = i

                print("Current Best Prec@1: ", best_prec1)
                print("Current Best Iteration: ", best_iter)
                print("Current cr val: {}, cr avg: {}".format(cr.val, cr.avg))
                wandb.log({
                    "iter_val":i,
                    "prec1_eval":prec1,
                    "best_prec1_eval":best_prec1,
                    "best_iteration_eval":best_iter})

                checkpoint_path = os.path.join(args.save_path, str(wandb.run.name)+'_checkpoint_{:05d}_{:.2f}.pth.tar'.format(i, prec1))
                save_checkpoint({
                    'iter': i,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                   
                },
                    is_best, filename=checkpoint_path)
                shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                                                              'checkpoint_latest'
                                                              '.pth.tar'))

                if i == args.iters:
                    break


def validate(args, test_loader, model, criterion, step):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze().long().cuda()
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var, args.num_bits, args.num_grad_bits)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) or (i == len(test_loader) - 1):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )
    logging.info('Step {} * Prec@1 {top1.avg:.3f}'.format(step, top1=top1))


    return top1.avg




def test_model(args):
    # create model
    model = get_model(args.arch).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        prec1 = validate(args, test_loader, model, criterion, args.start_iter)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def  update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cyclic_adjust_precision(args, _iter, cyclic_period):
    if args.is_cyclic_precision:
        assert len(args.cyclic_num_bits_schedule) == 2
        assert len(args.cyclic_num_grad_bits_schedule) == 2

        num_bit_min = args.cyclic_num_bits_schedule[0]
        num_bit_max = args.cyclic_num_bits_schedule[1]

        num_grad_bit_min = args.cyclic_num_grad_bits_schedule[0]
        num_grad_bit_max = args.cyclic_num_grad_bits_schedule[1]

        args.num_bits = np.rint(num_bit_min +
                                0.5 * (num_bit_max - num_bit_min) *
                                (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi)))
        args.num_grad_bits = np.rint(num_grad_bit_min +
                                     0.5 * (num_grad_bit_max - num_grad_bit_min) *
                                     (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi)))

        if _iter % args.eval_every == 0:
            logging.info('Iter [{}] num_bits = {} num_grad_bits = {} cyclic precision'.format(_iter, args.num_bits,
                                                                                                  args.num_grad_bits))





def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

tau_min =  0.85
tau_max = 0.99
def cpt_tau(epoch,args):

        a = torch.tensor(np.e)
        T_min, T_max = torch.tensor(tau_min).float(), torch.tensor(tau_max).float()
        A = (T_max - T_min) / (a - 1)
        B = T_min - A
        tau = A * torch.tensor([torch.pow(a, epoch/args.iters)]).float() + B
        return tau


args = parse_args()
save_path = args.save_path = os.path.join(args.save_folder, args.arch)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# config logging file
args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
handlers = [logging.FileHandler(args.logger_file, mode='w'),
            logging.StreamHandler()]
logging.basicConfig(level=logging.INFO,
                    datefmt='%m-%d-%y %H:%M',
                    format='%(asctime)s:%(message)s',
                    handlers=handlers)


wandb.login(key="042af338e26392be4f9f15361a0fcc7d53f9679d")


logging.info('args {}'.format(vars(args)))
# 1. Start a W&B Run
run = wandb.init(
    project="cyclebnn_real",
    notes="My first experiment",
    tags=["hardtanh"],
    config=vars(args),
)
logging.info('subito dopo')


if args.cmd == 'train':
    logging.info('start training {}'.format(args.arch))
    run_training(args)

elif args.cmd == 'test':
    logging.info('start evaluating {} with checkpoints from {}'.format(
        args.arch, args.resume))
    test_model(args)
