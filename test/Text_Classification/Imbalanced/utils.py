import os
import re
import shutil
import torch
from torchtext import data
from torchtext.vocab import Vectors

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')

def prepare_folders(args):
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('Creating folder: {}'.format(folder))
            os.mkdir(folder)

def word_cut(text):
    text = regex.sub(' ', text)
    return [word.strip() for word in text.split()]

def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors

def load_heybox_dataset(path, text_field, label_field, args, **kwargs):
    text_field.tokenize = word_cut
    train_dataset, test_dataset = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train='train.tsv', validation='test.tsv',
        fields=[
            ('index', None),
            ('label', label_field),
            ('text', text_field)
        ]
    )
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, test_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, test_dataset)
    label_field.build_vocab(train_dataset, test_dataset)
    train_iter, test_iter = data.Iterator.splits(
        (train_dataset, test_dataset),
        batch_sizes=(args.batch_size, 100),
        sort_key=lambda x: len(x.text),
        **kwargs)
    return train_iter, test_iter

def save_checkpoint(args, state, is_best):
    filename = '{}/{}/ckpt.pth.tar'.format(args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res