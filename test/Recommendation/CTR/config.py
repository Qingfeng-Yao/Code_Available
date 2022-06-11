"""
Configuration for each dataset

"""

import os

class CONFIG:
    """
        input_parser: tfrecord
        padded_shape: if varlen feature included, pad shape is needed for padded_batch
    """
    CHECKPOINT_DIR = './{}_checkpoint/{}'

    DATA_MAP = {
        'amazon': 'amazon_{}.tfrecords',
        # 'amazon': 'amazon_{}_small.tfrecords'
        # 'amazon': 'amazon_{}_small_group.tfrecords',
        'movielens': 'movielens_{}.tfrecords',
        'heybox': 'heybox_{}.tfrecords'
    }

    PARSER_MAP = {
        'amazon': 'tfrecord',
        'movielens': 'tfrecord',
        'heybox': 'tfrecord'
    }

    TYPE_MAP = {
        'amazon': 'varlen-sparse',
        'movielens': 'varlen-sparse',
        'heybox': 'varlen-sparse'
    }

    PADDED_SHAPE = {
        'amazon': ({
                'user_id': [],
                'hist_item_list': [None],
                'hist_cate_list':[None],
                'hist_length': [],
                'item': [],
                'item_cate':[],
                # 'user_group': []
        },[1]), 
        'heybox': ({
                'user_id': [],
                'hist_item_list': [None],
                'hist_cate_list':[None],
                'hist_length': [],
                'item': [],
                'item_cate':[]
        },[1]), 
        'movielens': ({
                'user_id': [],
                'hist_item_list': [None],
                'hist_cate_list':[None],
                'hist_length': [],
                'item': [],
                'item_cate':[],
                # 'user_group': []
        },[1])
    }

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name
        self.input_check()

    def input_check(self):
        if self.data_name not in CONFIG.DATA_MAP.keys():
            raise Exception('Currenlty only [{}] is supported'.format(' | '.join(CONFIG.DATA_MAP.keys())))

    @property
    def data_dir(self):
        return os.path.join('data', self.data_name, CONFIG.DATA_MAP[self.data_name])

    @property
    def checkpoint_dir(self):
        return CONFIG.CHECKPOINT_DIR.format(self.data_name, self.model_name)

    @property
    def input_parser(self):
        return CONFIG.PARSER_MAP[self.data_name]

    @property
    def pad_shape(self):
        return CONFIG.PADDED_SHAPE[self.data_name]

    @property
    def input_type(self):
        return CONFIG.TYPE_MAP[self.data_name]

MODEL_PARAMS = {
    'batch_size': 32,
    'num_epochs': 5000,
    'buffer_size': 512
}