from .preprocess import preprocess
from .method import lsa
from . import utils

class Deduplication(object):
    def __init__(self):
        super().__init__()

    @utils.get_run_time
    def deduplicate(self, query_data, args) -> dict:
        q_data = preprocess.cutWord(query_data, args)
        res = lsa.calc(q_data, args)
        return res