import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import time
import numpy as np
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import getModel as gM
import writeLogAcc as wA
from src.scores import AgeScoring, IndexedConcatDataset
from src.data import get_dataloader
from tqdm import tqdm

def main():
    global args, best_prec1
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

    args.distributed = False
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    _, train_set = get_dataloader(traindir, args.batch_size, shuffle=False,
                                  num_workers=args.num_workers,
                                  return_dataset=True, transform=train_transform)
    _, val_set = get_dataloader(valdir,   args.batch_size, shuffle=False,
                                  num_workers=args.num_workers,
                                  return_dataset=True, transform=train_transform)

    raw_concat = torch.utils.data.ConcatDataset([train_set, val_set])

    indexed_dataset = IndexedConcatDataset(raw_concat)

    full_loader = torch.utils.data.DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    N = len(raw_concat)
    n_classes = 1000
    print(f"Training set size: {N}")
    args.arch = "mobilenetv3_Age_ImgNet"
    print("=> creating model '{}'".format(args.arch))
    model = gM.get_model(args.arch, num_class=n_classes)
    model = model.cuda(args.gpu)
    filenameLOG = "./checkpoints/%s/" % args.arch + '/' + args.arch + '.txt'

    print('Number of model parameters: {}'.format(
        sum(p.data.nelement() for p in model.parameters())))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda(args.gpu)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    directory = "checkpoints/%s/" % args.arch
    os.makedirs(directory, exist_ok=True)

    age_acc = AgeScoring(N)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        resume_score_path = os.path.join(
            out_dir, f"age_scores_{args.start_epoch - 1:03d}.npy")
        if os.path.isfile(resume_score_path):
            age_acc = AgeScoring.load(resume_score_path, N)
            print(f"=> resumed age scores from '{resume_score_path}'")

    cudnn.benchmark = True

    Loss_plot = {}
    train_prec1_plot = {}
    train_prec5_plot = {}
    epoch_max  = None
    best_prec1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        loss_temp, train_prec1_temp, train_prec5_temp = train(
            full_loader, model, criterion, optimizer, epoch, age_acc)
        age_acc.epoch_end() 

        Loss_plot[epoch] = loss_temp
        train_prec1_plot[epoch] = train_prec1_temp
        train_prec5_plot[epoch] = train_prec5_temp
        prec1 = train_prec1_temp
        print(f"Epoch {epoch} complete: loss={loss_temp:.4f}, "
              f"prec@1={prec1:.2f}%, prec@5={train_prec5_temp:.2f}%")

        score_path = os.path.join(out_dir, f"age_scores_{epoch:03d}.npy")
        age_acc.save(score_path)
        print(f"Age scores saved to {score_path}")

        adjust_learning_rate(optimizer, epoch,
                             warmup_epochs=5, total_epochs=args.epochs)

        is_best = prec1 > best_prec1
        if is_best:
            epoch_max = epoch
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch':      epoch + 1,
            'arch':       args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer':  optimizer.state_dict(),
        }, is_best, directory=directory)

        data_save(directory + 'Loss_plot.txt',   Loss_plot)
        data_save(directory + 'train_prec1.txt', train_prec1_plot)
        data_save(directory + 'train_prec5.txt', train_prec5_plot)
        line = ('Epoch {}/{} summary: loss_train={:.5f}, acc_train={:.2f}%, '
                'best: {:.2f}% @ epoch {}').format(
            epoch, args.epochs, loss_temp, prec1, best_prec1, epoch_max)
        wA.writeLogAcc(filenameLOG, line)

        elapsed = (time.time() - start_time) / 3600
        print("-" * 80)
        print(f"Epoch time: {elapsed:.3f}h")
        print("-" * 80)

    print("\nAge scoring run complete.")
    print(f"Scores saved to: {out_dir}/")
    print(f"age_scores_{epoch:03d}.npy with shape={age_acc.get().shape}")


def train(train_loader, model, criterion, optimizer, epoch, age_acc):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()

    for i, (inputs, targets, indices) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            correct = predicted.eq(targets).detach().cpu().numpy()
        age_acc.update(indices.numpy(), correct)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time, data_time=data_time,
                      loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', directory=None):
    filepath = os.path.join(directory, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(directory, 'model_best.pth.tar'))


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

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
    with open(root, 'r') as f:
        lines = f.readlines()
    epoch = int(lines[-1].split()[0]) if lines else -1
    with open(root, 'a') as f:
        for line in file:
            if line > epoch:
                f.write(f"{line} {file[line]}\n")


if __name__ == "__main__":
    main()