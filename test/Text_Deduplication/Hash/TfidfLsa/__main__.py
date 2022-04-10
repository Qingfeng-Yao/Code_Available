from .preprocess import preprocess
from .method import lsa
from . import utils

class Deduplication(object):
    def __init__(self, threshold=0.7):
        super().__init__()
        self.threshold = threshold

    @utils.get_run_time
    def deduplicate(self, query_data, lsa_n_components) -> dict:
        q_data = preprocess.cutWord(query_data)
        res = lsa.calc(q_data, lsa_n_components, self.threshold)
        return res