# -*- coding: utf-8 -*-
import os
import time

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

