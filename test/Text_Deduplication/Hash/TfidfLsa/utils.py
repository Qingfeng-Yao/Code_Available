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