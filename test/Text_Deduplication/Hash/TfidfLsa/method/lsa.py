# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import numpy as np
from collections import defaultdict
import torch

from .. import utils

def tfidfTransform(texts):
    token_pattern = "(?u)\\b\\w+\\b"
    tfidf_vectorizer = TfidfVectorizer(token_pattern=token_pattern)
    tfidf_feature = tfidf_vectorizer.fit_transform(texts)
    return tfidf_feature, tfidf_vectorizer

def lsaTransform(X, n_components=1000):
    n_components = min(n_components, X.shape[-1])
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    lsa_feature = lsa.fit_transform(X)
    return lsa_feature, svd

def encode_onehot(labels, num_classes):
    onehot_labels = np.zeros((len(labels), num_classes))

    for i in range(len(labels)):
        onehot_labels[i, labels[i]] = 1

    return onehot_labels

def calcSimilarity(q_x, a_x):
    return q_x @ a_x.T

@utils.get_run_time
def calc(query, args):
    query_ids = []
    query_texts = []
    query_targets = []
    query_items = query.items()
    for idx, (id_, v) in enumerate(query_items):
        query_ids.append(id_)
        query_texts.append(v["text"])
        if not args.thesis:
            query_targets.append(int(v["target"]))
    if not args.thesis:
        query_targets = encode_onehot(query_targets, max(query_targets)+1)
    
    def getFeature(query_texts):
        texts = query_texts
        tfidf_feature, _ = tfidfTransform(texts)
        lsa_feature, _ = lsaTransform(tfidf_feature, n_components=args.lsa_n_components)
        query_feature = np.array(lsa_feature)
        return query_feature

    query_feature = getFeature(query_texts)
    if args.thesis:
        similarity = calcSimilarity(query_feature, query_feature)
        bag = defaultdict(list)
        for query_idx, (q_id, v) in enumerate(query_items):
            if len(bag) == 0:
                bag[q_id].append(query_idx)
                continue
            
            score = 0
            sim_bid = ""
            for bid, group in bag.items():
                for idx in group:
                    sim_dot = similarity[query_idx, idx]
                    source_vec = query_feature[query_idx]
                    target_vec = query_feature[idx]
                    source_norm = np.linalg.norm(source_vec)
                    target_norm = np.linalg.norm(target_vec)
                    sim = sim_dot/(source_norm * target_norm)
                    if sim>score:
                        score = sim
                        sim_bid = bid
            if score>args.cos_threshold:
                bag[sim_bid].append(query_idx)
            else:
                bag[q_id].append(query_idx)

        return bag, list(query_items)
    else:
        query_data = torch.from_numpy(query_feature).float()
        query_targets = torch.from_numpy(query_targets).float()

        @utils.get_run_time
        def train(query_data, query_targets, device, topk):
            query_data, query_targets = query_data.to(device), query_targets.to(device)
            mAP = utils.mean_average_precision(query_data, query_targets, device, topk)
            return mAP

        mAP = train(query_data, query_targets, args.device, args.topk)

        return mAP
