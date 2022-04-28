# -*- coding:utf-8 -*-
import argparse
import random
import time
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchtext.data as data

from utils import *
import model

'''
参考源码:
    [https://github.com/bigboNed3/chinese_text_cnn]
    [https://github.com/YyzHarry/imbalanced-semi-self]
运行环境:
    python=3.5.6 
    库: torch=1.0.0 torchtext=0.3.1 jieba=0.39 scikit-learn
    注释掉"cudnn.benchmark = True", 否则会出错(存疑)
核心代码思想: 
    (1)输入数据: 使用torchtext.data
        先处理数据的目的是获得vocabulary_size
    (2)模型: 使用TextCNN
    (3)训练: 每个epoch都调用train和validate函数, validate函数输出acc, 判断最优模型并进行模型的存储

不平衡数据heybox的textcnn分类
    将原始数据分成三份: 一份用于训练(不均衡), 一份用于测试(均衡, 每类取100个样本), 还有一份用于半监督(不均衡, 取剩下的30%)
    三份数据均组织成tsv格式
    [标准分类]
        [python3 train_textcnn.py --static --non_static --multichannel][单块小GPU无法成功运行]
        [best acc: 91.278]
    [半监督分类]
        首先进行伪标签的生成, 即运行文件gen_pseudolabels.py
        [python3 train_textcnn.py --exp_str semi_training]
        [best acc: ]
    [自监督分类]
        首先进行预训练(文本逆序增强), 即运行文件[python3 train_textcnn.py --exp_str pre_training]
        [python3 train_textcnn.py --exp_str self_training --pretrained_model checkpoint/heybox_textcnn_pre_training/ckpt.best.pth.tar]
        [best acc: ]
'''

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='heybox', choices=['heybox'])
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--exp_str', default='standard_training', choices=['standard_training', 'semi_training', 'self_training', 'pre_training'],
                    help='(additional) name to indicate experiment')
parser.add_argument('--device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: 0]')
parser.add_argument('--pretrained_model', type=str, default='', help='for self trainging')

parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('--embedding_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('--filter_num', type=int, default=100, help='number of each size of filter')
parser.add_argument('--filter_sizes', type=str, default='3,4,5',
                    help='comma-separated filter sizes to use for convolution')
parser.add_argument('--static', action='store_true', help='whether to use static pre-trained word vectors')
parser.add_argument('--non_static', action='store_true', help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('--multichannel', action='store_true', help='whether to use 2 channel of word vectors')
parser.add_argument('--pretrained_name', type=str, default='sgns.zhihu.word',
                    help='filename of pre-trained word vectors')
parser.add_argument('--pretrained_path', type=str, default='pretrained', help='path of pre-trained word vectors')

parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs for train [default: 200]')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 128]')

parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='./checkpoint')
args = parser.parse_args()
best_acc1 = 0

args.store_name = '_'.join([args.dataset, 'textcnn', args.exp_str])
prepare_folders(args)
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    print('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, which can slow down training considerably! '
                    'You may see unexpected behavior when restarting from checkpoints.')
if args.device is not None:
    print('You have chosen a specific GPU. This will completely disable data parallelism.')
    print("Use GPU: {} for training".format(args.device))

print('Preparing data...')
if args.dataset == 'heybox':
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    if args.exp_str == 'semi_training':
        train_iter, _, test_iter = load_heybox_dataset(os.path.join(args.data_path, args.dataset, 'input_data'), \
            text_field, label_field, args, is_semi=True, device=args.device, repeat=False, shuffle=True)
    else:
        train_iter, _, test_iter = load_heybox_dataset(os.path.join(args.data_path, args.dataset, 'input_data'), \
            text_field, label_field, args, device=args.device, repeat=False, shuffle=True)
    
    args.vocabulary_size = len(text_field.vocab)
    if args.static:
        args.embedding_dim = text_field.vocab.vectors.size()[-1]
        args.vectors = text_field.vocab.vectors
    if args.multichannel:
        args.static = True
        args.non_static = True
    args.class_num = len(label_field.vocab)
else:
    raise NotImplementedError("Dataset {} is not supported!".format(args.dataset))
print("vocabulary_size: {}".format(args.vocabulary_size))

print("===> Creating model textcnn")
args.cuda = args.device != -1 and torch.cuda.is_available()
args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]
text_cnn = model.TextCNN(args)
if args.cuda:
    torch.cuda.set_device(args.device)
    text_cnn = text_cnn.cuda()
optimizer = torch.optim.Adam(text_cnn.parameters(), lr=args.lr)

if args.pretrained_model:
    checkpoint = torch.load(args.pretrained_model, map_location=torch.device('cuda:{}'.format(str(args.device))))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if 'linear' not in k and 'fc' not in k:
            new_state_dict[k] = v
    text_cnn.load_state_dict(new_state_dict, strict=False)
    print('===> Pretrained weights found in total: [{}]'.format(len(list(new_state_dict.keys()))))
    print('===> Pre-trained model loaded: {}'.format(args.pretrained_model))

# cudnn.benchmark = True

def train(train_iter, text_cnn, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    text_cnn.train()

    end = time.time()
    for i, batch in enumerate(train_iter):
        data_time.update(time.time() - end)

        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)
        if args.exp_str == 'pre_training':
            feature, target = reorder(feature)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logits = text_cnn(feature)
        loss = F.cross_entropy(logits, target)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), feature.size(0))
        top1.update(acc1[0], feature.size(0))
        top5.update(acc5[0], feature.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_iter), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            print(output)

def validate(val_iter, text_cnn, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    text_cnn.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_iter):
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)
            if args.exp_str == 'pre_training':
                feature, target = reorder(feature)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logits = text_cnn(feature)
            loss = F.cross_entropy(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), feature.size(0))
            top1.update(acc1[0], feature.size(0))
            top5.update(acc5[0], feature.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(logits, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_iter), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)

        if args.exp_str != 'pre_training':
            cf = confusion_matrix(all_targets, all_preds).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt

            rs = recall_score(all_targets, all_preds, average='macro')
            ps = precision_score(all_targets, all_preds, average='macro')
            fs = f1_score(all_targets, all_preds, average='macro')
            output = ('Test Results: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.5f}\
                 Recall {rs:.3f} Precision {ps:.3f} f1 {fs:.3f}'
                    .format(top1=top1, top5=top5, loss=losses, rs=rs, ps=ps, fs=fs))
            out_cls_acc = 'Test Class Accuracy: %s' % (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x}))
            print(output)
            print(out_cls_acc)
            return top1.avg, out_cls_acc, rs, ps, fs
        else:
            return top1.avg

cur_cls_acc, cur_fs, cur_ps, cur_rs = None, None, None, None 
for epoch in range(args.start_epoch, args.epochs):
    train(train_iter, text_cnn, optimizer, epoch, args)
    if args.exp_str != 'pre_training':
        acc1, out_cls_acc, out_rs, out_ps, cur_fs = validate(test_iter, text_cnn, epoch, args)
    else:
        acc1 = validate(test_iter, text_cnn, epoch, args)

    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)
    output_best = 'Best Acc@1: %.3f' % best_acc1
    print(output_best)
    if args.exp_str != 'pre_training':
        if cur_cls_acc is None or is_best:
            cur_cls_acc = out_cls_acc
            cur_rs, cur_ps, cur_fs = out_rs, out_ps, cur_fs
        print("Best F1 {}, Prec {}, Recall {}".format(cur_fs, cur_ps, cur_rs))
        print("Best Cls Acc {}\n".format(cur_cls_acc))

    save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': text_cnn.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)