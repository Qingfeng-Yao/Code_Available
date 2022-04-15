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

def readIdfdict(path=os.path.join(setting.STATIC_PATH, "idf.txt")):
    idf = {}
    idf_sum = 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            word, w = line.strip().split()
            idf[word] = float(w)
            idf_sum += float(w)
            count += 1
    return idf, idf_sum/count

def distance(v1, v2):
    x = (v1 ^ v2) & ((1 << 128) - 1)
    ans = 0
    while x:
        ans += 1
        x &= x - 1
    return ans

def mean_average_precision(query_code, query_targets, model_type, device, topk):
    num_query = query_targets.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        retrieval = (query_targets[i, :] @ query_targets.t() > 0).float()
        retrieval = torch.cat((retrieval[:i], retrieval[i+1:])) # Remove self
        if model_type == "keywords":
            hamming_dist = [distance(query_code[i], query_code[j]) for j in range(num_query) if i!=j]
            hamming_dist = torch.Tensor(hamming_dist).to(device)
        else:
            # Calculate hamming distance
            hamming_dist = 0.5 * (query_code.shape[1] - query_code[i, :] @ query_code.t())
            hamming_dist = torch.cat((hamming_dist[:i], hamming_dist[i+1:])) # Remove self

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

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
