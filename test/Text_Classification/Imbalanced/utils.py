import os
import re
import shutil
import torch
import numpy as np
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

def load_extra_heybox_dataset(path, text_field, label_field, index_field, args, **kwargs):
    text_field.tokenize = word_cut
    data_name = "extra_"+args.extra_tag+".tsv"
    train_dataset, val_dataset, test_dataset = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train=data_name, validation='train_extra.tsv', test='test.tsv',
        fields=[
            ('index', index_field),
            ('label', label_field),
            ('text', text_field)
        ]
    )
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, val_dataset, test_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, val_dataset, test_dataset)
    label_field.build_vocab(train_dataset, val_dataset, test_dataset)
    index_field.build_vocab(train_dataset, val_dataset, test_dataset)
    train_iter, _, _ = data.Iterator.splits(
        (train_dataset, val_dataset, test_dataset),
        batch_sizes=(args.batch_size, args.batch_size, 20),
        sort_key=lambda x: len(x.text),
        **kwargs)
    return train_iter

def load_heybox_dataset(path, text_field, label_field, args, is_semi=False, **kwargs):
    text_field.tokenize = word_cut
    if is_semi:
        data_name = 'pseudo_train_'+args.extra_tag+".tsv"
        train_dataset, val_dataset, test_dataset = data.TabularDataset.splits(
            path=path, format='tsv', skip_header=True,
            train=data_name, validation='train_extra.tsv', test='test.tsv',
            fields=[
                ('index', None),
                ('label', label_field),
                ('text', text_field)
            ]
        )
    else:
        train_dataset, val_dataset, test_dataset = data.TabularDataset.splits(
            path=path, format='tsv', skip_header=True,
            train='train.tsv', validation='train_extra.tsv', test='test.tsv',
            fields=[
                ('index', None),
                ('label', label_field),
                ('text', text_field)
            ]
        )
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, val_dataset, test_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, val_dataset, test_dataset)
    label_field.build_vocab(train_dataset, val_dataset, test_dataset)
    train_iter, _, test_iter = data.Iterator.splits(
        (train_dataset, val_dataset, test_dataset),
        batch_sizes=(args.batch_size, args.batch_size, 20),
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

def text_aug(inputs, text_aug="reverse"):
    batch = inputs.shape[0]
    if text_aug == "mix":
        target = torch.Tensor(np.random.permutation([0, 1, 2, 3, 4] * (int(batch / 5) + 1)), device=inputs.device)[:batch]
    elif text_aug == "reverse":
        target = torch.Tensor(np.random.permutation([0, 1] * (int(batch / 2) + 1)), device=inputs.device)[:batch]
    else:
        # target = torch.Tensor(np.random.permutation([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * (int(batch / 10) + 1)), device=inputs.device)[:batch]
        target = torch.Tensor(np.random.permutation([0, 1] * (int(batch / 2) + 1)), device=inputs.device)[:batch]
    target = target.long()
    doc = torch.zeros_like(inputs)
    doc.copy_(inputs)
    for i in range(batch):
        if text_aug == "reverse":
            doc[i, :] = inputs[i, :] if target[i] else inputs[i, torch.arange(inputs[i].size(0)-1, -1, -1).long()]
        elif text_aug == "delete":
            doc[i, :] = inputs[i, :] if target[i] else torch.cat([torch.tensor(0).unsqueeze(0), inputs[i, 1:]])

            # if target[i] == 0:
            #     doc[i, :] = inputs[i, :]
            # elif target[i] == 1:
            #     doc[i, :] = torch.cat([torch.tensor(0).unsqueeze(0), inputs[i, 1:]])
            # else:
            #     doc[i, :] = torch.cat([torch.cat([inputs[i, :target[i]-1], torch.tensor(0).unsqueeze(0)]), inputs[i, target[i]:]])
        elif text_aug == "crop":
            doc[i, :] = inputs[i, :] if target[i] else torch.cat([torch.tensor([0 for _ in range(5)]), inputs[i, 5:]])

            # if target[i] == 0:
            #     doc[i, :] = inputs[i, :]
            # elif target[i] == 1:
            #     doc[i, :] = torch.cat([torch.tensor([0 for _ in range(5)]), inputs[i, 5:]])
            # else:
            #     doc[i, :] = torch.cat([torch.cat([inputs[i, :target[i]-1], torch.tensor([0 for _ in range(5)])]), inputs[i, target[i]+5-1:]])
        elif text_aug == "exchange":
            doc[i, :] = inputs[i, :] if target[i] else torch.cat([inputs[i, torch.arange(1, -1, -1).long()], inputs[i, 2:]])

            # if target[i] == 0:
            #     doc[i, :] = inputs[i, :]
            # elif target[i] == 1:
            #     doc[i, :] = torch.cat([inputs[i, torch.arange(1, -1, -1).long()], inputs[i, 2:]])
            # else:
            #     doc[i, :] = torch.cat([torch.cat([inputs[i, :target[i]-1], inputs[i, torch.arange(target[i], target[i]-2, -1).long()]]), inputs[i, target[i]+2-1:]])
        else:
            if target[i] == 0:
                doc[i, :] = inputs[i, :]
            elif target[i] == 1:
                doc[i, :] = inputs[i, torch.arange(inputs[i].size(0)-1, -1, -1).long()]
            elif target[i] == 2:
                doc[i, :] = torch.cat([torch.tensor(0).unsqueeze(0), inputs[i, 1:]])
            elif target[i] == 3:
                doc[i, :] = torch.cat([torch.tensor([0 for _ in range(5)]), inputs[i, 5:]])
            elif target[i] == 4:
                doc[i, :] = torch.cat([inputs[i, torch.arange(1, -1, -1).long()], inputs[i, 2:]])
            else:
                print("target error!!!")
            
    return doc, target