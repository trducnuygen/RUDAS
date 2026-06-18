"""
train_warmup.py — Phase 1: warm-up training run.

Trains a backbone on full ImageNet and, after every epoch, runs a second
no-grad scoring pass that accumulates:

  age_scores.npy       — per-sample correct-prediction count  [0, E]
  forgetting_scores.npy— per-sample forgetting event count    [0, E-1]
  el2n_scores.npy      — per-sample mean EL2N (early epochs only)

All three signals are saved incrementally so the run can be resumed if
interrupted.

"""

import math
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # specify which GPU(s) to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used

import argparse
import time
import numpy as np
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import getModel as gM
import writeLogAcc as wA
from src.scores import AgeAccumulator, ForgettingAccumulator, EL2NAccumulator
from src.data import get_dataloader
from tqdm import tqdm

# ── Main ──────────────────────────────────────────────────────────────────────



def main():
    global args, best_prec1, config
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="dataset")
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
    parser.add_argument('--print-freq', default=100, type=int, dest='print_freq')
    parser.add_argument('--output_dir', default="age_scores", type=str,
                    help='directory to save scores and checkpoints')
    
    args = parser.parse_args()

    # for mobilenetv3
    args.lr = 0.05
    args.momentum = 0.9
    args.weight_decay = 4e-5

    out_dir = os.path.join(args.output_dir, "age_scoring")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    # Data loading code
    args.distributed = False

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    _, train_set = get_dataloader(traindir, 
                                  args.batch_size, 
                                  shuffle=False, 
                                  num_workers=args.num_workers, 
                                  return_dataset=True, transform=train_transform)
    _, val_set = get_dataloader(valdir, 
                                args.batch_size, 
                                shuffle=False, 
                                num_workers=args.num_workers, 
                                return_dataset=True, transform=train_transform)
    # this loader is for training.
    full_dataset = torch.utils.data.ConcatDataset([train_set, val_set])
    full_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    # for scoring
    scoring_trans = transforms.Compose([
        transforms.Resize(int(224/0.875)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    _, train_set = get_dataloader(traindir, 
                                  args.batch_size, 
                                  shuffle=False, 
                                  num_workers=args.num_workers, 
                                  return_dataset=True, transform=scoring_trans)
    _, val_set = get_dataloader(valdir, 
                                args.batch_size, 
                                shuffle=False, 
                                num_workers=args.num_workers, 
                                return_dataset=True, transform=scoring_trans)
    score_dataset= torch.utils.data.ConcatDataset([train_set, val_set])
    score_loader = torch.utils.data.DataLoader(
        score_dataset,
        batch_size=2048, # should be able to afford large batch since no backprop
        shuffle=False,  # to score
        num_workers=args.num_workers,
        pin_memory=True
    )

    N        = len(full_dataset)
    n_classes = 1000
    print(f"Training set size: {N}")

    # ── Model ─────────────────────────────────────────────────────────────────
    args.arch = "mobilenetv3_Age_ImgNet"
    print("=> creating model '{}'".format(args.arch))   
    model = gM.get_model(args.arch, num_class=n_classes) # ImageNet
    model = model.cuda(args.gpu)
    filenameLOG = "./checkpoints/%s/"%(args.arch) + '/' + args.arch + '.txt'    
    
    print('Number of models parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
    
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda(args.gpu)  # no smoothing for scoring accuracy
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    
    directory = "checkpoints/%s/"%(args.arch)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    age_acc  = AgeAccumulator(N)
    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        if os.path.isfile(args.resume):
            
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # ── Training loop ─────────────────────────────────────────────────────────
    Loss_plot = {}
    train_prec1_plot = {}
    train_prec5_plot = {}
    epoch_max = None
    best_prec1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        loss_temp, train_prec1_temp, train_prec5_temp = train(full_loader, model, criterion, optimizer, epoch)
        Loss_plot[epoch] = loss_temp
        train_prec1_plot[epoch] = train_prec1_temp
        train_prec5_plot[epoch] = train_prec5_temp 
        prec1, prec5 = train_prec1_temp, train_prec5_temp
        print(f"Epoch {epoch} training complete: loss={loss_temp:.4f}, prec@1={prec1:.2f}%, prec@5={prec5:.2f}%")

        adjust_learning_rate(optimizer, epoch, warmup_epochs=5, total_epochs=args.epochs)

        # ── Scoring pass (every epoch) ──────────────
        if epoch >= 0: 
            with torch.no_grad():
                sc_t0 = time.time()
                print(f"  → Scoring pass epoch {epoch}...")
                model.eval()
                for i, (inputs, labels) in tqdm(enumerate(score_loader), total=len(score_loader)):
                    
                    inputs = inputs.cuda(args.gpu, non_blocking=True)
                    labels = labels.cuda(args.gpu, non_blocking=True)

                    logits = model(inputs)
                    _, predicted = torch.max(logits, 1)
                    correct = predicted.eq(labels)

                    indices = torch.arange(i * score_loader.batch_size, min((i + 1) * score_loader.batch_size, N)).cuda(args.gpu)
                    indices = indices.detach().cpu().numpy()
                    correct = correct.detach().cpu().numpy()
                    age_acc.update(indices, correct)
                    # fgt_acc.update(indices, correct)
                    # el2n_acc.update(indices, logits.cpu().numpy(), labels.cpu().numpy(), epoch)

                # Save scores after every epoch (safe to interrupt)
                age_acc.save(os.path.join(out_dir, f"age_scores_{epoch:03d}.npy"))
                print(f"Scoring done in {time.time()-sc_t0:.1f}s")

        # ── Checkpoint ────────────────────────────────────────────────────────
        is_best = prec1 > best_prec1
        if is_best:
            epoch_max = epoch
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,directory = "checkpoints/%s/"%(args.arch))
        
        # Loss,train_prec1,train_prec5,val_prec1,val_prec5
        data_save(directory + 'Loss_plot.txt', Loss_plot)
        data_save(directory + 'train_prec1.txt', train_prec1_plot)
        data_save(directory + 'train_prec5.txt', train_prec5_plot)
        line = 'Epoch {}/{} summary: loss_train={:.5f}, acc_train={:.2f}%, loss_val={:.2f}, acc_val={:.2f}% (best: {:.2f}% @ epoch {})'.format(epoch, args.epochs, loss_temp, train_prec1_temp, 0, prec1, best_prec1, epoch_max)
        wA.writeLogAcc(filenameLOG,line)
        end_time = time.time()
        time_value = (end_time - start_time) / 3600
        print("-" * 80)
        print(time_value)
        print("-" * 80)

    print("\n Age scoring run complete.")
    print(f"  Scores saved to: {out_dir}/")
    print(f"  age_scores_{epoch:03d}.npy       shape={age_acc.get().shape}")

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_batch = {}
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
     
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:            
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',directory =None):
    #directory = "checkpoints/%s/"%(model_name_dataset)
    
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        lr = args.lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def data_save(root, file):
    if not os.path.exists(root):
        os.mknod(root)
    file_temp = open(root, 'r')
    lines = file_temp.readlines()
    if not lines:
        epoch = -1
    else:
        epoch = lines[-1][:lines[-1].index(' ')]
    epoch = int(epoch)
    file_temp.close()
    file_temp = open(root, 'a')
    for line in file:
        if line > epoch:
            file_temp.write(str(line) + " " + str(file[line]) + '\n')
    file_temp.close()

if __name__ == "__main__":
    main()
