# -*- coding:utf-8 -*-
import warnings
import os
import argparse
from pprint import pprint

import dataset
import dch

'''
python=3.6.15
库: tensorflow-gpu==1.15.0 opencv-python==4.5.4.58 matplotlib==3.3.4

参考: [https://github.com/thulab/DeepHash]

cifar数据下载[https://github.com/thulab/DeepHash/releases/download/v0.1/cifar10.zip]
预训练模型下载[https://github.com/thulab/DeepHash/releases/download/v0.1/reference_pretrain.npy.zip]
python3 train.py
    [i2i_by_feature  0.6821789315325325
    i2i_after_sign  0.6311438900013032
    i2i_prec_radius_2       0.6427404290628373
    i2i_recall_radius_2     0.6279094444444444
    i2i_map_radius_2        0.7517326035342455]
'''

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--img_model', default='alexnet', type=str)
parser.add_argument('--model_weights', type=str,
                    default='pretrained_model/reference_pretrain.npy')
parser.add_argument('--gpus', default='0', type=str)

parser.add_argument('--output_dim', default=64, type=int)   # 256/128：比特位
parser.add_argument('--q_lambda', default=0, type=float) # 平衡损失的超参数
parser.add_argument('--gamma', default=20, type=float) # 柯西分布的scale参数

parser.add_argument('--lr', '--learning-rate', default=0.005, type=float)
parser.add_argument('--iter_num', default=2000, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-vb', '--val-batch-size', default=16, type=int)
parser.add_argument('--decay_step', default=10000, type=int)
parser.add_argument('--decay_factor', default=0.1, type=float)
parser.add_argument('--log_dir', default='tflog', type=str)

tanh_parser = parser.add_mutually_exclusive_group(required=False)
tanh_parser.add_argument('--with_tanh', dest='with_tanh', action='store_true')
tanh_parser.add_argument('--without_tanh', dest='with_tanh', action='store_false')
parser.set_defaults(with_tanh=True)

parser.add_argument('--finetune_all', default=True, type=bool)
parser.add_argument('--save_dir', default="models/", type=str)
parser.add_argument('--data_dir', default="data/", type=str)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_81': 81, 'coco': 80}
Rs = {'cifar10': 54000, 'nuswide_81': 5000, 'coco': 5000}
args.R = Rs[args.dataset]
args.label_dim = label_dims[args.dataset]

args.img_tr = os.path.join(args.data_dir, args.dataset, "train.txt")
args.img_te = os.path.join(args.data_dir, args.dataset, "test.txt")
args.img_db = os.path.join(args.data_dir, args.dataset, "database.txt")

pprint(vars(args))

data_root = os.path.join(args.data_dir, args.dataset)
query_img, database_img = dataset.import_validation(data_root, args.img_te, args.img_db)

if not args.evaluate:
    train_img = dataset.import_train(data_root, args.img_tr)
    model_weights = dch.train(train_img, database_img, query_img, args)
    args.model_weights = model_weights

maps = dch.validation(database_img, query_img, args)
for key in maps:
    print(("{}\t{}".format(key, maps[key])))

pprint(vars(args))