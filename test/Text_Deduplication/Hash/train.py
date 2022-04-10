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
去重数据集heybox/input_data(其中较大的数据集没有ground truth，仅用于比较运行时间)
    TfidfLsa+data:
        [python3 train.py]
        [Recall: 0.736, Precision: 0.866, F1-score: 0.796]
    Simhash+data
        [python3 train.py --model_name simhash]
        [Recall: 0.692, Precision: 0.857, F1-score: 0.766]
    Simhash+data+optim:
        [python3 train.py --model_name simhash --optim]

    TfidfLsa+big_data:
        [python3 train.py --big_data]
    Simhash+data
        [python3 train.py --model_name simhash --big_data]
'''

args = parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

if args.big_data:
    data_path = 'heybox/input_data/big_data.json'
else:
    data_path = 'heybox/input_data/data.json'
with open(data_path, "r") as f:
    query_data = json.load(f)

if args.model_name == 'tfidf_lsa':
    out_path = "{}tfidf_lsa_{}/".format(args.out_path, args.cos_threshold)

    deduplication = TfidfLsa.Deduplication(threshold=args.cos_threshold)
    bag, query_items = deduplication.deduplicate(query_data, args.lsa_n_components)

elif args.model_name == 'simhash':
    if args.optim:
        out_path = "{}simhash_{}_optim/".format(args.out_path, args.dis_threshold)
    else:
        out_path = "{}simhash_{}/".format(args.out_path, args.dis_threshold)
    deduplication = Simhash.Deduplication(threshold=args.dis_threshold)
    bag, query_items = deduplication.deduplicate(query_data, args)

if not args.big_data:
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