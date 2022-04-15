# -*- coding: utf-8 -*-
import os
import time
import torch

from . import setting


def get_run_time(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        print("[%s] runs for %.2f second" % (func.__name__, time.time() - t0))
        return res

    return wrapper

def readStopwords(path=os.path.join(setting.STATIC_PATH, "stopwords.txt")):
    stopwords = []
    with open(path, "r", encoding="utf-8") as f:
        stopwords = [word.strip() for word in f.readlines()]
    return stopwords

def mean_average_precision(query_data, query_targets, device, topk):
    num_query = query_targets.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        retrieval = (query_targets[i, :] @ query_targets.t() > 0).float()
        retrieval = torch.cat((retrieval[:i], retrieval[i+1:])) # Remove self

        # Calculate dot product similarity
        dot_sim = query_data[i, :] @ query_data.t()
        dot_sim = torch.cat((dot_sim[:i], dot_sim[i+1:])) # Remove self

        # Arrange position according to dot sim
        retrieval = retrieval[torch.argsort(dot_sim, descending=True)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()
        
        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP.item()