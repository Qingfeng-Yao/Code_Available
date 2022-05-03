# -*- coding:utf-8 -*-
import argparse
import warnings
import random
from tensorboardX import SummaryWriter
import time
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import *
from losses import LDAMLoss, FocalLoss
from dataset.imbalance_cifar import ImbalanceCIFAR10

'''
参考源码: 
    [https://github.com/YyzHarry/imbalanced-semi-self]
运行环境:
    python=3.6.13
    torch=1.4.0 torchvision=0.5.0 numpy=1.19.5 tensorboardX=2.5 scikit-learn=0.24.2 
核心代码思想:
    (1)模型
    resnet: 
    第1层卷积核大小为7x7, 数目为64, stride=2, padding=3(在输入的两边添加0); BN(64); F.relu; [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    (前向传播对第1层的输出进行.requires_grad_()和.retain_grad()存疑)
    包含四个部分: (1)64, stride=1; (2)128, stride=2; (3)256, stride=2; (4)512, stride=2
    (除了第一部分的stride=1, 其余均为stride=2存疑)
    平均池化F.avg_pool2d(out, 7)+全连接nn.Linear(512 * expansion, num_classes)
    不同resnet主要区别在于block类型和每个部分的blocks数目, 前面的数字表示总层数
        BasicBlock: 每个block共有两层(3x3卷积层+BN), expansion=1(影响特征维度), shortcut策略包括identity和1x1卷积
        Bottleneck: 每个block共有三层(1x1+3x3+1x1卷积层+BN), expansion=4(影响特征维度), shortcut策略包括identity和1x1卷积
        18: BasicBlock+[2, 2, 2, 2]: 18=1+2*2*4+1
        34: BasicBlock+[3, 4, 6, 3]: 34=1+3*2+4*2+6*2+3*2+1
        50: Bottleneck+[3, 4, 6, 3]: 50=1+3*3+4*3+6*3+3*3+1
        101: Bottleneck+[3, 4, 23, 3]: 101=1+3*3+4*3+23*3+3*3+1
        152: Bottleneck+[3, 8, 36, 3]: 152=1+3*3+8*3+36*3+3*3+1
    resnet_cifar:
    第1层卷积核大小为3x3, 数目为16, stride=1, padding=1(在输入的两边添加0); BN(16); F.relu
    包含三个部分: (1)16, stride=1; (2)32, stride=2; (3)64, stride=2
    平均池化F.avg_pool2d(out, out.size()[3])+全连接nn.Linear(64, num_classes)
    [权重初始化: 对线性层和卷积层进行初始化, init.kaiming_normal_(m.weight)]
    不同resnet主要区别在于每个部分的blocks数目, 前面的数字表示总层数
        BasicBlock: 每个block共有两层(3x3卷积层+BN), expansion=1(影响特征维度), shortcut策略包括identity和补0(F.pad)
        20: BasicBlock+[3, 3, 3]: 20=1+3*2*3+1
        32: BasicBlock+[5, 5, 5]: 32=1+5*2*3+1
        44: BasicBlock+[7, 7, 7]: 44=1+7*2*3+1
        56: BasicBlock+[9, 9, 9]: 56=1+9*2*3+1
        110: BasicBlock+[18, 18, 18]: 20=1+18*2*3+1
        1202: BasicBlock+[200, 200, 200]: 20=1+200*2*3+1
    预训练模型加载时不需要考虑线性层, 由于维度不一样
    (2)输入图像
    利用torchvision.transforms进行图像变换
        训练数据变换: transforms.RandomCrop+transforms.RandomHorizontalFlip+transforms.ToTensor+transforms.Normalize
        测试数据变换: transforms.ToTensor+transforms.Normalize
    训练数据是不平衡的, 测试数据是平衡的
        ImbalanceCIFAR10: 需要指定不平衡因子imb_factor和不平衡类型imb_type
        imb_type有三种类型:
            exp: 每类的样本数呈指数变化
            step: 只有多数类和少数类两类, 各占图像类别一半
            ave: 平衡数据
    数据重采样: 指定torch.utils.data.DataLoader的参数sampler
        train_sampler = ImbalancedDatasetSampler(train_dataset) # 继承torch.utils.data.sampler.Sampler
        确定原始数据中每个样本的权重, 重新采样获得新数据集; 每类样本的权重相同, 权重由每类样本数确定
        每类权重由如下公式确定: (1.0-0.9999)/(1.0-np.power(0.9999, label_to_count)), 对应类别样本数越多, 权重越小
    (3)训练
    每个epoch开始调整学习率: adjust_learning_rate
    是否重加权: per_cls_weights将用于损失函数
        否: per_cls_weights = None
        是: 重加权中确定类别权重的方式与重采样中的一致, 即(1.0-0.9999)/(1.0-np.power(0.9999, label_to_count))
            不过重加权还需要再加一步: per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        DRW: 延迟重加权, 即需要确定重加权的时机, 即与epoch有关
    损失函数的选择: 
        交叉熵: 仅考虑正负样本损失的加权
        focal: 对难易样本损失进行加权, 即针对预测概率进行加权
        LDAM: 对预测概率重新进行计算, 引入类依赖的margin
    每个epoch都调用train和validate函数, validate函数输出acc, 判断最优模型并进行模型的存储
        train函数
            model.train()
            batch迭代: 前向传播, 计算损失, 反向传播
        validate函数
            model.eval()以及with torch.no_grad()
            batch迭代: 前向传播
            指标计算


标准监督训练及一些基线模型: 原始不平衡数据集(cifar10)+标签信息
    [python3 train.py --dataset cifar10 --imb_factor 0.01 --loss_type CE][模型使用resnet32]
    [best acc:71.930]
    [python3 train.py --dataset cifar10 --imb_factor 0.01 --loss_type CE --train_rule Resample][模型使用resnet32]
    [best acc:71.420]
    [python3 train.py --dataset cifar10 --imb_factor 0.01 --loss_type CE --train_rule Reweight][模型使用resnet32]
    [best acc:73.100]
    [python3 train.py --dataset cifar10 --imb_factor 0.01 --loss_type Focal][模型使用resnet32]
    [best acc:73.210]
    [python3 train.py --dataset cifar10 --imb_factor 0.01 --loss_type LDAM][模型使用resnet32][LDAM需要使用归一化的线性层]
    [best acc:73.510]
    [python3 train.py --dataset cifar10 --imb_factor 0.01 --loss_type LDAM --train_rule DRW][模型使用resnet32]
    [best acc:76.810]

自监督预训练: 原始不平衡数据集(cifar10)+无标签信息
    首先预训练(pretrain_rot.py), 然后标准训练
    标准训练: [python train.py --dataset cifar10 --imb_factor 0.01 --loss_type CE --pretrained_model <path_to_ssp_model>][模型使用resnet32]
    其中<path_to_ssp_model>的格式如: checkpoint/cifar10_resnet32_CE_None_exp_0.01_pretrain_rot/ckpt.best.pth.tar
    [best acc:72.780]
'''

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
print(model_names)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--loss_type', default="CE", type=str, choices=['CE', 'Focal', 'LDAM']) # 'Focal'和'LDAM'都是基于CE做的改进
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type(确定每个类别的样本数): exp | step | ave')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor; 对应论文中的参数rho=100')
parser.add_argument('--train_rule', default='None', type=str,
                    choices=['None', 'Resample', 'Reweight', 'DRW']) # 确定每个类别的权重，进而影响损失函数
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='ss_pretrained', type=str,
                    help='(additional) name to indicate experiment')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
parser.add_argument('--epochs', default=200, type=int, metavar='N')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='./checkpoint')
best_acc1 = 0

def main():
    args = parser.parse_args()
    if args.pretrained_model:
        args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule,
                                args.imb_type, str(args.imb_factor), args.exp_str, 'pretrained_model'])
    else:
        args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule,
                                args.imb_type, str(args.imb_factor), args.exp_str])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, which can slow down training considerably! '
                      'You may see unexpected behavior when restarting from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    main_worker(args.gpu, args)

def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    print(f"===> Creating model '{args.arch}'")
    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset in {'cifar10', 'svhn'}:
        num_classes = 10
    else:
        raise NotImplementedError
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    mean = [0.4914, 0.4822, 0.4465] if args.dataset.startswith('cifar') else [.5, .5, .5]
    std = [0.2023, 0.1994, 0.2010] if args.dataset.startswith('cifar') else [.5, .5, .5]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if args.dataset == 'cifar10':
        train_dataset = ImbalanceCIFAR10(
            root=args.data_path, imb_type=args.imb_type, imb_factor=args.imb_factor,
            rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root=args.data_path,
                                       train=False, download=True, transform=transform_val)
        train_sampler = None
        if args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported!")

    # evaluate only
    if args.evaluate: # 只需要下载训练好的模型, 然后进行验证即可
        assert args.resume, 'Specify a trained model using [args.resume]'
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
        model.load_state_dict(checkpoint['state_dict'])
        print(f"===> Checkpoint '{args.resume}' loaded, testing...")
        validate(val_loader, model, nn.CrossEntropyLoss(), 0, args)
        return

    if args.resume: # 需要下载模型和优化器, 还有开始的epoch和best_acc, 适合中断训练后继续训练
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            raise ValueError(f"No checkpoint found at '{args.resume}'")

    # load self-supervised pre-trained model
    if args.pretrained_model:
        checkpoint = torch.load(args.pretrained_model, map_location=torch.device(f'cuda:{str(args.gpu)}'))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'linear' not in k and 'fc' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
        print(f'===> Pre-trained model loaded: {args.pretrained_model}')

    cudnn.benchmark = True

    if args.dataset.startswith(('cifar', 'svhn')):
        cls_num_list = train_dataset.get_cls_num_list()
        print('cls num list for {}:'.format(args.dataset))
        print(cls_num_list)
        args.cls_num_list = cls_num_list

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == 'Reweight':
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            per_cls_weights = None

        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        acc1 = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best acc@1: %.3f\n' % best_acc1
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

def train(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        target = target.cuda()
        output = model(inputs)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            inputs = inputs.cuda()
            target = target.cuda()

            output = model(inputs)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
            flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        if tf_writer is not None:
            tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
            tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
            tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
            tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg


if __name__ == '__main__':
    main()