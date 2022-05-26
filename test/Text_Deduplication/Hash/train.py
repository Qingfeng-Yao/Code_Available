# -*- coding:utf-8 -*-
import json
import numpy as np
import os
import random
import torch
import shutil

import TfidfLsa
import Simhash

from utils import parse_args, compute_metrics, cmp_gt

'''
参考源码: 
    [https://github.com/1e0ng/simhash]
    [https://github.com/TreezzZ/LSH_PyTorch]
运行环境:
    torch==1.11.0 
核心代码思路:
    均需要先对中文文本进行分词, 去停用词等操作
    对query_data本身进行相似性或距离计算
    TfidfLsa: 得到文本的tfidf向量, 并进行lsa降维; 使用内积进行相似性计算
    Simhash
        随机映射: 得到文本的tfidf向量, 并进行随机映射; 使用海明距离进行距离计算
        特征哈希: 得到文本的词-权重对集合, 并进行哈希计算得到文本对应的指纹; 使用海明距离进行距离计算

去重数据集heybox/input_data
    TfidfLsa+data:
        [python3 train.py]
        [MAP: 0.610]
    Simhash(随机映射)+data
        [python3 train.py --model_name simhash]
        [MAP: 0.574]
    Simhash(特征哈希)+data
        [python3 train.py --model_name simhash --model_type keywords --code_length 128]
        [MAP: 0.528]
Thesis results:(其中较大的数据集没有ground truth, 仅用于比较运行时间)
    TfidfLsa+data:
        [python3 train.py --thesis]
        [Recall: 0.736, Precision: 0.866, F1-score: 0.796]
        [cutWord] runs for 1.91 second
        [calc] runs for 1.53 second
        [deduplicate] runs for 3.44 second
    Simhash(特征哈希)+data
        [python3 train.py --thesis --model_name simhash --model_type keywords]
        [Recall: 0.659, Precision: 0.920, F1-score: 0.768]
        [extractKeyWord] runs for 2.24 second
        [calc] runs for 0.71 second
        [deduplicate] runs for 2.94 second
    Simhash(特征哈希)+data+optim:
        [python3 train.py --thesis --model_name simhash --model_type keywords --optim]
        [Recall: 0.715, Precision: 0.866, F1-score: 0.783]
        [extractKeyWord] runs for 3.98 second
        [calc] runs for 0.78 second
        [deduplicate] runs for 4.76 second
    TfidfLsa+big_data:
        [python3 train.py --thesis --big_data]
        [cutWord] runs for 21.28 second
        [calc] runs for 29.89 second
        [deduplicate] runs for 51.17 second
    Simhash(特征哈希)+big_data
        [python3 train.py --thesis --model_name simhash --model_type keywords --big_data]
        [extractKeyWord] runs for 21.55 second
        [calc] runs for 23.44 second
        [deduplicate] runs for 44.99 second
    Simhash(特征哈希)+big_data+optim
        [python3 train.py --thesis --model_name simhash --model_type keywords --big_data --optim]
        [extractKeyWord] runs for 55.52 second
        [calc] runs for 23.74 second
        [deduplicate] runs for 79.27 second
    对未优化的simhash结果进行深入分析: result_analysis.py
        首先运行[python3 train.py --thesis --model_name simhash --model_type keywords --cmp]生成false_maps.npz
        然后运行result_analysis.py
'''

args = parse_args()

# GPU
if args.gpu is None:
    args.device = torch.device("cpu")
else:
    args.device = torch.device("cuda:%d" % args.gpu)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.big_data:
    data_path = 'heybox/input_data/big_data.json'
else:
    data_path = 'heybox/input_data/data.json'
with open(data_path, "r") as f:
    query_data = json.load(f)

if args.model_name == 'tfidf_lsa':
    out_path = "{}tfidf_lsa_{}/".format(args.out_path, args.cos_threshold)
    deduplication = TfidfLsa.Deduplication()
    if args.thesis:
        bag, query_items = deduplication.deduplicate(query_data, args)
    else:
        mean_AP = deduplication.deduplicate(query_data, args)
        print("{}: MAP[{:.3f}], lsa_components[{}]".format(args.model_name, mean_AP, args.lsa_n_components))

elif args.model_name == 'simhash':
    if args.optim:
        out_path = "{}simhash_{}_optim/".format(args.out_path, args.dis_threshold)
    else:
        out_path = "{}simhash_{}/".format(args.out_path, args.dis_threshold)
    deduplication = Simhash.Deduplication()
    if args.thesis:
        bag, query_items = deduplication.deduplicate(query_data, args)
    else:
        mean_AP = deduplication.deduplicate(query_data, args)
        print("{}[{}]: MAP[{:.3f}], code_length[{}]".format(args.model_name, args.model_type, mean_AP, args.code_length))

if args.thesis and (not args.big_data):
    # test on ground truth
    gt_dirs = [d for d in os.listdir(args.gt_path) if '.txt' in d]
    print("gt classes: {}".format(len(gt_dirs)))
    print("pred classes: {}".format(len(bag)))
    ave_p,ave_r,f1 = compute_metrics(gt_dirs, args.gt_path, bag, query_items)
    print("recall: %f, precision: %f, f1-score: %f" % (ave_r,ave_p,f1))

    if args.cmp:
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        cmp_gt(bag, gt_dirs, out_path, args.gt_path, query_items)