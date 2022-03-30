import argparse
import os
import shutil
import time
import numpy as np
import statistics 
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
import torch.nn.functional as F
from math import ceil
from random import Random

# Importing modules related to distributed processing
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.autograd import Variable
from torch.multiprocessing import spawn

###########
from gossip_choco import GossipDataParallel
from gossip_choco import RingGraph, GridGraph
from gossip_choco import UniformMixing
from gossip_choco import *
from quantized_training import *
import xlsxwriter



parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--quantized_train', default=0, type=int,    help='enable low precision training (8-bit): 0-false 1-true')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet', help = 'resnet or vgg or resquant' )
parser.add_argument('-depth', '--depth', default=20, type=int, help='depth of the resnet model')
parser.add_argument('--normtype',   default='evonorm', help = 'batchnorm or rangenorm or groupnorm or evonorm' )
parser.add_argument('--dataset', dest='dataset',     help='available datasets: cifar10, cifar100', default='cifar10', type=str)
parser.add_argument('--classes', default=10, type=int,     help='number of classes in the dataset')
parser.add_argument('-b', '--batch-size', default=256, type=int,  metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,     metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',     help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,  metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-world_size', '--world_size', default=8, type=int, help='total number of nodes')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',  help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',   help='number of total epochs to run')
parser.add_argument('--seed', default=1234, type=int,        help='set seed')
parser.add_argument('--run_no', default=1, type=str, help='parallel run number, models saved as model_{rank}_{run_no}.th')
parser.add_argument('--print-freq', '-p', default=30, type=int,    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',    help='The directory used to save the trained models',   default='save_temp', type=str)
parser.add_argument('--port', dest='port',   help='between 3000 to 65000',default='29500' , type=str)
parser.add_argument('--save-every', dest='save_every',  help='Saves checkpoints at every specified number of epochs',  type=int, default=5)
parser.add_argument('--biased', dest='biased', action='store_true',     help='biased compression')
parser.add_argument('--unbiased', dest='biased', action='store_false',     help='biased compression')
parser.add_argument('--level', default=32, type=int, metavar='k',  help='quantization level 1-32')
parser.add_argument('--eta',  default=1.0, type=float,  metavar='AR', help='averaging rate')
parser.add_argument('--compressor', dest='fn',    help='Compressor function: quantize, sparsify', default='quantize', type=str)
parser.add_argument('--k', default=0.0, type=float,    help='compression ratio for sparsification')
parser.add_argument('--skew', default=0.0, type=float,     help='obelongs to [0,1] where 0= completely iid and 1=completely non-iid')
parser.add_argument('--qgm', default=0, type=int,    help='quasi global momentum 0-false 1-true')

args = parser.parse_args()

class Partition(object):
    
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

def skew_sort(indices, skew, classes, class_size, seed):
    # skew belongs to [0,1]
    rng = Random()
    rng.seed(seed)
    class_indices = {}
    for i in range(0, classes):
        class_indices[i]=indices[0:class_size[i]]
        indices = indices[class_size[i]:]
    random_indices = []
    sorted_indices = []
    for i in range(0, classes):
        sorted_size    = int(skew*class_size[i])
        sorted_indices = sorted_indices + class_indices[i][0:sorted_size]
        random_indices = random_indices + class_indices[i][sorted_size:]
    rng.shuffle(random_indices)
    return random_indices, sorted_indices
            
    
class DataPartitioner(object):
    """ Partitions a dataset into different chunks"""
    def __init__(self, data, sizes, skew, classes, class_size, seed, device):
        
        self.data = data
        self.partitions = []
        data_len = len(data)
        dataset = torch.utils.data.DataLoader(data, batch_size=512, shuffle=False, num_workers=2)
        labels = []
        for batch_idx, (inputs, targets) in enumerate(dataset):
              labels = labels+targets.tolist()
        #labels  = [data[i][1] for i in range(0, data_len)]
        sort_index = np.argsort(np.array(labels))
        indices = sort_index.tolist()
        indices_rand, indices = skew_sort(indices, skew=skew, classes=classes, class_size=class_size, seed=seed)
        
        for frac in sizes:
            if skew==1:
                part_len = int(frac*data_len)
                self.partitions.append(indices[0:part_len])
                indices = indices[part_len:]
            elif skew==0:
                part_len = int(frac*data_len)
                self.partitions.append(indices_rand[0:part_len])
                indices_rand = indices_rand[part_len:] 
            else:
                part_len = int(frac*data_len*skew); 
                part_len_rand = int(frac*data_len*(1-skew))
                part_ind = indices[0:part_len]+indices_rand[0:part_len_rand]
                self.partitions.append(part_ind)
                indices = indices[part_len:]
                indices_rand = indices_rand[part_len_rand:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    
def partition_trainDataset(device):
    """Partitioning dataset""" 
    if args.dataset == 'cifar10':
        normalize   = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        classes    = 10
        class_size = {x:5000 for x in range(10)}

        dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    elif args.dataset == 'cifar100':
        normalize  = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
        classes    = 100
        class_size = {x:500 for x in range(10)}

        dataset = datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    elif args.dataset == 'imagenette':
        normalize  = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        classes    = 10
        class_size = {0: 963, 1: 955, 2: 993, 3: 858, 4: 941, 5: 956, 6: 961, 7: 931, 8: 951, 9: 960}

        data_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(), normalize,])

        data_dir = './data/imagenette'

        dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
                  
       
    size = dist.get_world_size()
    #print(size)
    bsz = int((args.batch_size) / float(size))
    
    partition_sizes = [1.0/size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes, skew=args.skew, classes=classes, class_size=class_size, seed=args.seed, device=device)

    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True, num_workers=2)
    return train_set, bsz


def test_Dataset():
  
    if args.dataset=='cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset=='cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
        dataset = datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset == 'imagenette':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        data_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(), normalize,])

        data_dir = './data/imagenette'

        dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),  data_transforms)

    val_bsz = 64
    val_set = torch.utils.data.DataLoader(dataset, batch_size=val_bsz, shuffle=False, num_workers=2)
    return val_set, val_bsz


def run(rank, size):
    global args, best_prec1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:{}".format(rank%4))
	    
    best_prec1 = 0
    ##############
    data_transferred = 0
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(os.path.join(args.save_dir, "excel_data"))
    if not os.path.exists(os.path.join(args.save_dir, "excel_data")):
        os.makedirs(os.path.join(args.save_dir, "excel_data"))
    
    if args.quantized_train==1:
        if args.arch == 'resnet':
            model = resnet_quantized(num_classes=args.classes, depth=args.depth, dataset=args.dataset)
        elif args.arch == 'vgg11':
            model = vgg11_quantized(num_classes=args.classes, dataset=args.dataset)
        elif args.arch == 'mobilenet':
            model = mobilenetv2_quantized(num_classes=args.classes, dataset=args.dataset)
        else:
            raise NotImplementedError
    else:
        if args.arch=='resnet':
            model = resnet(num_classes=args.classes, depth=args.depth, dataset=args.dataset, norm_type=args.normtype, groups=2)
        elif args.arch == 'vgg11':
            model = vgg11(num_classes=args.classes, dataset=args.dataset, norm_type=args.normtype, groups=2)
        elif args.arch == 'mobilenet':
            model = MobileNetV2(num_classes=args.classes, dataset=args.dataset, norm_type=args.normtype, groups=2)
        else:
            raise NotImplementedError
        
    if rank==0:
        print(args)
        print('Printing model summary...')
        if 'cifar' in args.dataset: print(summary(model, (3, 32, 32), batch_size=int(args.batch_size/size), device='cpu'))
        else: print(summary(model, (3, 224, 224), batch_size=int(args.batch_size/size), device='cpu'))
        

    graph = RingGraph(rank, size) #undirected/directed ring structure
    #graph = GridGraph(rank, size) # torus graph structure
   
    mixing = UniformMixing(graph, device)
    model = GossipDataParallel(model, 
				device_ids=[rank%4],
				rank=rank,
				world_size=size,
				graph=graph, 
				mixing=mixing,
				comm_device=device, 
                level = args.level,
                biased = args.biased,
                eta = args.eta,
                compress_ratio=args.k, 
                compress_fn = args.fn, 
                compress_op = 'top_k', 
                momentum=args.momentum,
                weight_decay = args.weight_decay,
                lr = args.lr,
                qgm = args.qgm) 
    model.to(device)
    cudnn.benchmark = True
    train_loader, bsz_train = partition_trainDataset(device=device)
    val_loader, bsz_val     = test_Dataset()
    
    # define loss function (criterion) and nvidia-smioptimizer
    criterion = nn.CrossEntropyLoss().to(device)#cuda()
    if args.qgm==1:
        optimizer = optim.SGD(model.parameters(), args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum = args.momentum, nesterov=True)

    if rank==0: print(optimizer)
    if 'res' in args.arch or 'mobile' in args.arch:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=[100, 150])
    elif 'vgg' in args.arch:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.5, milestones=[30, 60, 90, 120, 150, 180])
    
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr*0.1
    
    for epoch in range(0, args.epochs):  
        if epoch==1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        model.block()
        dt= train(train_loader, model, criterion, optimizer, epoch, bsz_train, optimizer.param_groups[0]['lr'], device, rank)
        data_transferred += dt
        lr_scheduler.step()
        prec1 = validate(val_loader, model, criterion, bsz_val,device, epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model_{}_{}.th'.format(rank, args.run_no)))

    #############################
    dt= gossip_avg(train_loader, model, criterion, optimizer, epoch, bsz_train, optimizer.param_groups[0]['lr'], device, rank)
    print('Final test accuracy')
    prec1 = validate(val_loader, model, criterion, bsz_val,device, epoch)
    print("Rank : ", rank, "Data transferred(in GB) during training: ", data_transferred/1.0e9, "Data transferred(in GB) in final gossip averaging rounds: ", dt/1.0e9, "\n")
    #Store processed data
    torch.save((prec1, (data_transferred+dt)/1.0e9), os.path.join(args.save_dir, "excel_data","rank_{}.sp".format(rank)))

#def train(train_loader, model, criterion, optimizer, epoch, batch_size, writer, device):
def train(train_loader, model, criterion, optimizer, epoch, batch_size, lr, device, rank):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_transferred = 0 
   
    # switch to train mode
    model.train()
    end = time.time()
    step = len(train_loader)*batch_size*epoch
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_var, target_var = Variable(input).to(device), Variable(target).to(device)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        _, amt_data_transfer = model.transfer_params(epoch=epoch+(1e-3*i), lr=lr)
        data_transferred += amt_data_transfer
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Rank: {0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      dist.get_rank(), epoch, i, len(train_loader),  batch_time=batch_time,
                      loss=losses, top1=top1))
        step += batch_size 
    return data_transferred

def gossip_avg(train_loader, model, criterion, optimizer, epoch, batch_size, lr, device, rank):
    """
       This function runs only gossip averaging for 50 iterations without local sgd updates - used to obtain the average model
    """
    data_transferred = 0 
    n = 50
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input_var, target_var = Variable(input).to(device), Variable(target).to(device)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.zero_grad()
        _, amt_data_transfer = model.transfer_params(epoch=epoch+(1e-3*i), lr=lr)
        data_transferred += amt_data_transfer
        if i==n: break
    return data_transferred

def validate(val_loader, model, criterion, batch_size, device, epoch=0):
#def validate(val_loader, model, criterion, batch_size, writer, device, epoch=0):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    step = len(val_loader)*batch_size*epoch

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var, target_var = Variable(input).to(device), Variable(target).to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Rank: {0}\t'
                      'Test: [{1}/{2}]\t'
                      #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          dist.get_rank(),i, len(val_loader), 
                          #batch_time=batch_time, 
                          loss=losses,
                          top1=top1))
            step += batch_size
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

def init_process(rank, size, fn, backend='nccl'):
    """Initialize distributed enviornment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank,size)

if __name__ == '__main__':
    size = args.world_size
    
    spawn(init_process, args=(size,run), nprocs=size,join=True)
    #read stored data
    excel_data = {
        'Algo': 'CHOCO-SGD',
        "learning rate": args.lr,
        "skew" : args.skew,
        "sparsification ratio" : int(args.k*100),
        "qgm" : args.qgm,
        "8 bit training": args.quantized_train, 
        "eta" : args.eta,
        "avg test acc":[0.0]*size,
        "data transferred": [0.0]*size
         }
    for i in range(size):
        acc, d_tfr = torch.load(os.path.join( args.save_dir, "excel_data","rank_{}.sp".format(i) ))
        excel_data["avg test acc"][i] = acc
        excel_data["data transferred"][i] = d_tfr
    torch.save(excel_data, os.path.join(args.save_dir, "excel_data","dict"))
    print(excel_data)
    

