# -*- coding:utf-8 -*-
import argparse
import torch
import random
import numpy as np
from loguru import logger

from data.dataloader import load_data
import lsh

'''
参考源码:
    [https://github.com/TreezzZ/LSH_PyTorch]
运行环境:
    loguru==0.6.0 torch==1.11.0 
核心代码思路:
    query_data和retrieval_data各自通过与随机映射矩阵相乘并获取符号得到对应的码, 即(num, emn_dim)-->(num, code_length)
    计算query_code中的每一个与retrieval_code之间的海明距离(利用向量与矩阵的@运算)
    利用海明距离重新排序retrieval_target, 然后计算mean_AP和P_R曲线

cifar:
    [python3 train.py]
    [code length:8][map:0.1138]
    [code length:16][map:0.1191]
    [code length:24][map:0.1195]
    [code length:32][map:0.1288]
    [code length:48][map:0.1349]
    [code length:64][map:0.1436]
    [code length:96][map:0.1536]
    [code length:128][map:0.1521]
'''
def run():
    # Load configuration
    args = load_config()
    logger.add('logs/{}.log'.format(args.dataset), rotation='500 MB', level='INFO')
    logger.info(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    _, _, query_data, query_targets, retrieval_data, retrieval_targets = load_data(args.dataset, args.root)

    # Training
    for code_length in args.code_length:
        checkpoint = lsh.train(
            query_data,
            query_targets,
            retrieval_data,
            retrieval_targets,
            code_length,
            args.device,
            args.topk,
        )
        logger.info('[code length:{}][map:{:.4f}]'.format(code_length, checkpoint['map']))

def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='LSH_PyTorch')
    parser.add_argument('--dataset', type=str, default='cifar_10_gist.mat',
                        help='Dataset name.')
    parser.add_argument('--root', type=str, default='dataset/',
                        help='Path of dataset')
    parser.add_argument('--code_length', default='8,16,24,32,48,64,96,128', type=str,
                        help='Binary hash code length.(default: 8,16,24,32,48,64,96,128)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate top k data map.(default: all)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--seed', default=3367, type=int,
                        help='Random seed.(default: 3367)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))

    return args


if __name__ == "__main__":
    run()