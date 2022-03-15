# -*- coding:utf-8 -*-
import click
import logging
import random
import numpy as np
import torch

from utils.config import Config
from cvdd import CVDD
from datasets.main import load_dataset

'''
参考: [https://github.com/lukasruff/CVDD-PyTorch]

python=3.7
库: requirements.txt

文本异常检测+CVDD
    需要提前建立数据文件夹和日志文件夹，然后进入src运行程序
    若使用spacy进行分词，则需要python3 -m spacy download en
	reuters: [python3 main.py reuters cvdd_Net ../log/test_reuters ../data --clean_txt --lr_milestone 40 --pretrained_model GloVe_6B --normal_class 6]
        Test AUC: 96.63%
    newsgroups20
    imdb
'''
################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['reuters', 'newsgroups20', 'imdb']))
@click.argument('net_name', type=click.Choice(['cvdd_Net']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=1, help='Set seed. If -1, use randomization.')
@click.option('--tokenizer', default='spacy', type=click.Choice(['spacy', 'bert']), help='Select text tokenizer.')
@click.option('--clean_txt', is_flag=True, help='Specify if text should be cleaned in a pre-processing step.')
@click.option('--embedding_size', type=int, default=300, help='Size of the word vector embedding.')
@click.option('--pretrained_model', default=None,
              type=click.Choice([None, 'GloVe_6B', 'GloVe_42B', 'GloVe_840B', 'GloVe_twitter.27B', 'FastText_en',
                                 'bert']),
              help='Load pre-trained word vectors or language models to initialize the word embeddings.')
@click.option('--ad_score', default='context_dist_mean', type=click.Choice(['context_dist_mean', 'context_best']),
              help='Choose the AD score function')
@click.option('--n_attention_heads', type=int, default=3, help='Number of attention heads in self-attention module.')
@click.option('--attention_size', type=int, default=150, help='Self-attention module dimensionality.')
@click.option('--lambda_p', type=float, default=1.0,
              help='Hyperparameter for context vector orthogonality regularization P = (CCT - I)')
@click.option('--alpha_scheduler', default='logarithmic', type=click.Choice(['soft', 'linear', 'logarithmic', 'hard']),
              help='Set annealing strategy for temperature hyperparameter alpha.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for training.')
@click.option('--lr', type=float, default=0.01,
              help='Initial learning rate for training. Default=0.001')
@click.option('--n_epochs', type=int, default=100, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=64, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=0.5e-6,
              help='Weight decay (L2 penalty) hyperparameter.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--n_threads', type=int, default=0,
              help='Sets the number of OpenMP threads used for parallelizing CPU operations')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, device, seed, tokenizer, clean_txt,
         embedding_size, pretrained_model, ad_score, n_attention_heads, attention_size, lambda_p, alpha_scheduler,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, n_jobs_dataloader, n_threads,
         normal_class):
    """
    Context Vector Data Description (CVDD): An unsupervised anomaly detection method for text.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)
    logger.info('Tokenizer: %s' % cfg.settings['tokenizer'])
    logger.info('Clean text in pre-processing: %s' % cfg.settings['clean_txt'])
    if cfg.settings['embedding_size'] is not None:
        logger.info('Word vector embedding size: %d' % cfg.settings['embedding_size'])
    logger.info('Load pre-trained model: %s' % cfg.settings['pretrained_model'])

    # Print CVDD configuration)
    logger.info('Anomaly Score: %s' % cfg.settings['ad_score'])
    logger.info('Number of attention heads: %d' % cfg.settings['n_attention_heads'])
    logger.info('Attention size: %d' % cfg.settings['attention_size'])
    logger.info('Orthogonality regularization hyperparameter: %.3f' % cfg.settings['lambda_p'])
    logger.info('Temperature alpha annealing strategy: %s' % cfg.settings['alpha_scheduler'])

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Set seed for reproducibility
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)
    if n_threads > 0:
        torch.set_num_threads(n_threads)
        logger.info('Number of threads used for parallelizing CPU operations: %d' % n_threads)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, cfg.settings['tokenizer'],
                           clean_txt=cfg.settings['clean_txt'])

    # Initialize CVDD model and set word embedding
    cvdd = CVDD(cfg.settings['ad_score'])
    cvdd.set_network(net_name=net_name,
                     dataset=dataset,
                     pretrained_model=cfg.settings['pretrained_model'],
                     embedding_size=cfg.settings['embedding_size'],
                     attention_size=cfg.settings['attention_size'],
                     n_attention_heads=cfg.settings['n_attention_heads'])

    # If specified, load model parameters from already trained model
    if load_model:
        cvdd.load_model(import_path=load_model, device=device)
        logger.info('Loading model from %s.' % load_model)

    # Train model on dataset
    cvdd.train(dataset,
               optimizer_name=cfg.settings['optimizer_name'],
               lr=cfg.settings['lr'],
               n_epochs=cfg.settings['n_epochs'],
               lr_milestones=cfg.settings['lr_milestone'],
               batch_size=cfg.settings['batch_size'],
               lambda_p=cfg.settings['lambda_p'],
               alpha_scheduler=cfg.settings['alpha_scheduler'],
               weight_decay=cfg.settings['weight_decay'],
               device=device,
               n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    cvdd.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Save results, model, and configuration
    cvdd.save_results(export_json=xp_path + '/results.json')
    cvdd.save_model(export_path=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')


if __name__ == '__main__':
    main()