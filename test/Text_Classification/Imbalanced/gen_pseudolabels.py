# -*- coding:utf-8 -*-
import argparse
import numpy as np
import csv
import torchtext.data as data

from utils import *
import model

'''
在未标记数据集extra.tsv上生成伪标签
    [python3 gen_pseudolabels.py --resume xxx]
    其中xxx格式如: checkpoint/heybox_textcnn_standard_training/ckpt.best.pth.tar
'''

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='heybox', choices=['heybox'])
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: 0]')
parser.add_argument('--resume', type=str, default='')

parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('--embedding_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('--filter_num', type=int, default=100, help='number of each size of filter')
parser.add_argument('--filter_sizes', type=str, default='3,4,5',
                    help='comma-separated filter sizes to use for convolution')
parser.add_argument('--static', action='store_true', help='whether to use static pre-trained word vectors')
parser.add_argument('--non_static', action='store_true', help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('--pretrained_name', type=str, default='sgns.zhihu.word',
                    help='filename of pre-trained word vectors')
parser.add_argument('--pretrained_path', type=str, default='pretrained', help='path of pre-trained word vectors')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 128]')

parser.add_argument('--data_filename', default='extra.tsv', type=str)
parser.add_argument('--output_dir', default='./data', type=str)
parser.add_argument('--output_filename', default='pseudo_train.tsv', type=str)

args = parser.parse_args()

if args.device is not None:
    print('You have chosen a specific GPU. This will completely disable data parallelism.')
    print("Use GPU: {} for training".format(args.device))

print("Preparing unlabeled data...")
if args.dataset == 'heybox':
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    _, train_iter, _= load_heybox_dataset(os.path.join(args.data_path, args.dataset, 'input_data'), \
        text_field, label_field, args, device=args.device, repeat=False, shuffle=True)

    args.vocabulary_size = len(text_field.vocab)
    if args.static:
        args.embedding_dim = text_field.vocab.vectors.size()[-1]
        args.vectors = text_field.vocab.vectors
    args.class_num = len(label_field.vocab)
print("vocabulary_size: {}".format(args.vocabulary_size))

print("===> Creating and loading model textcnn")
args.cuda = args.device != -1 and torch.cuda.is_available()
args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]
text_cnn = model.TextCNN(args)
if args.cuda:
    torch.cuda.set_device(args.device)
    text_cnn = text_cnn.cuda()

assert args.resume is not None
if os.path.isfile(args.resume):
    print("===> Loading checkpoint {}".format(args.resume))
    checkpoint = torch.load(args.resume, map_location=torch.device('cuda:{}'.format(str(args.device))))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if 'linear' in k:
            new_state_dict[k.replace('linear', 'fc')] = v
        else:
            new_state_dict[k] = v
    text_cnn.load_state_dict(new_state_dict)
    print('===> Checkpoint weights found in total: [{}]'.format(len(list(new_state_dict.keys()))))
else:
    raise ValueError("No checkpoint found at {}".format(args.resume))

text_cnn.eval()

print("Running model on unlabeled data...")
predictions, truths = [], []
for i, batch in enumerate(train_iter):
    feature = batch.text
    feature.data.t_()
    if args.cuda:
        feature = feature.cuda()
    _, preds = torch.max(text_cnn(feature), dim=1)

    predictions.append(preds.cpu().numpy())

    if (i+1) % 10 == 0:
        print('Done %d/%d' % (i+1, len(train_iter)))

new_extrapolated_targets = np.concatenate(predictions)
if args.dataset == 'heybox':
    train_file_path = os.path.join(args.data_path, 'heybox', 'input_data', 'train.tsv')
    out_path = os.path.join(args.output_dir, 'heybox', 'input_data', args.output_filename)
    if os.path.exists(out_path):
        os.remove(out_path)
    shutil.copyfile(train_file_path, out_path)

    extra_file_path = os.path.join(args.data_path, 'heybox', 'input_data', args.data_filename)
    extra_data = []
    with open(extra_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader: 
            assert len(row) == 1
            content = row[0].split('\t')
            assert len(content) == 3
            extra_data.append(content[2])

    with open(out_path, 'a') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        for i in range(len(extra_data)):
            tsv_w.writerow([i, new_extrapolated_targets[i], extra_data[i]]) 

        
    
