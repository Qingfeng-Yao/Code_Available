# -*- coding:utf-8 -*-
import argparse
import numpy as np
import csv
import torchtext.data as data

from utils import *
import model

'''
在未标记数据集上生成伪标签
    extra_b_60.tsv: [python3 gen_pseudolabels.py --static --non_static --multichannel --extra_tag b_60 --resume checkpoint/heybox_textcnn_standard_training/ckpt.best.pth.tar]
    extra_b_60_70.tsv: [python3 gen_pseudolabels.py --static --non_static --multichannel --extra_tag b_60_70 --resume checkpoint/heybox_textcnn_standard_training/ckpt.best.pth.tar]
    extra_b_60_80.tsv: [python3 gen_pseudolabels.py --static --non_static --multichannel --extra_tag b_60_80 --resume checkpoint/heybox_textcnn_standard_training/ckpt.best.pth.tar]
    extra_b_60_90.tsv: [python3 gen_pseudolabels.py --static --non_static --multichannel --extra_tag b_60_90 --resume checkpoint/heybox_textcnn_standard_training/ckpt.best.pth.tar]
    extra_b_60_100.tsv: [python3 gen_pseudolabels.py --static --non_static --multichannel --extra_tag b_60_100 --resume checkpoint/heybox_textcnn_standard_training/ckpt.best.pth.tar]
    extra_b_60_106.tsv: [python3 gen_pseudolabels.py --static --non_static --multichannel --extra_tag b_60_106 --resume checkpoint/heybox_textcnn_standard_training/ckpt.best.pth.tar]
'''

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='heybox', choices=['heybox'])
parser.add_argument('--extra_tag', default='b_60', choices=['b_60', 'b_60_70', 'b_60_80', 'b_60_90', 'b_60_100', 'b_60_106'])
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
parser.add_argument('--multichannel', action='store_true', help='whether to use 2 channel of word vectors')
parser.add_argument('--pretrained_name', type=str, default='sgns.zhihu.word',
                    help='filename of pre-trained word vectors')
parser.add_argument('--pretrained_path', type=str, default='pretrained', help='path of pre-trained word vectors')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 128]')

parser.add_argument('--output_dir', default='./data', type=str)

args = parser.parse_args()

if args.device is not None:
    print('You have chosen a specific GPU. This will completely disable data parallelism.')
    print("Use GPU: {} for training".format(args.device))

print("Preparing unlabeled data...")
if args.dataset == 'heybox':
    text_field = data.Field(lower=True)
    label_field = data.LabelField(sequential=False, use_vocab=False)
    index_field = data.LabelField(sequential=False, use_vocab=False)
    train_iter = load_extra_heybox_dataset(os.path.join(args.data_path, args.dataset, 'input_data'), \
        text_field, label_field, index_field, args, device=torch.device('cuda:{}'.format(str(args.device))), repeat=False, shuffle=True)

    args.vocabulary_size = len(text_field.vocab)
    if args.static:
        args.embedding_dim = text_field.vocab.vectors.size()[-1]
        args.vectors = text_field.vocab.vectors
    if args.multichannel:
        args.static = True
        args.non_static = True
    args.class_num = len(label_field.vocab)
print("vocabulary_size: {}".format(args.vocabulary_size))
print("class_num: {}".format(args.class_num))
print(label_field.vocab.stoi)
print("index size: {}".format(len(index_field.vocab)))
# print(index_field.vocab.stoi)

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
predictions, indexes = [], []
truths = []
for i, batch in enumerate(train_iter):
    index = batch.index
    label = batch.label
    feature = batch.text
    feature.data.t_()
    if args.cuda:
        feature = feature.cuda()
    _, preds = torch.max(text_cnn(feature), dim=1)

    predictions.append(preds.cpu().numpy())
    indexes.append(index.cpu().numpy())
    truths.append(label.cpu().numpy())

    if (i+1) % 10 == 0:
        print('Done %d/%d' % (i+1, len(train_iter)))

new_extrapolated_targets = np.concatenate(predictions)
true_indexes = np.concatenate(indexes)
true_labels = np.concatenate(truths)

if args.dataset == 'heybox':
    train_file_path = os.path.join(args.data_path, 'heybox', 'input_data', 'train.tsv')
    out_path = os.path.join(args.output_dir, 'heybox', 'input_data', 'pseudo_train_'+args.extra_tag+".tsv")
    if os.path.exists(out_path):
        print("removing existing {}".format(out_path))
        os.remove(out_path)
    shutil.copyfile(train_file_path, out_path)

    extra_file_path = os.path.join(args.data_path, 'heybox', 'input_data', 'extra_'+args.extra_tag+".tsv")
    extra_data = []
    e_l = []
    with open(extra_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader: 
            assert len(row) == 1
            content = row[0].split('\t')
            assert len(content) == 3
            e_l.append(content[1])
            extra_data.append(content[2])

    correct_num = 0
    with open(out_path, 'a') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        for i in range(len(new_extrapolated_targets)):
            assert true_labels[i] == int(e_l[true_indexes[i]])
            if new_extrapolated_targets[i] == true_labels[i]:
                correct_num += 1
            tsv_w.writerow([i, new_extrapolated_targets[i], extra_data[true_indexes[i]]]) 

    print("[pred correct num: {}/{}={}]".format(correct_num, len(new_extrapolated_targets), correct_num/len(new_extrapolated_targets)))

        
    
