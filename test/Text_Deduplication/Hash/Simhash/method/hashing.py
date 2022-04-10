# -*- coding: utf-8 -*-
import hashlib
import numpy as np
from collections import defaultdict
import time

from .. import utils

def distance(v1, v2):
    x = (v1 ^ v2) & ((1 << 128) - 1)
    ans = 0
    while x:
        ans += 1
        x &= x - 1
    return ans

@utils.get_run_time
def calc(query, threshold, args):
    def hashfunc(x):
        return hashlib.md5(x).digest()

    def bitarray_from_bytes(b):
        return np.unpackbits(np.frombuffer(b, dtype='>B'))

    def binary_to_int(b):
        res = 0
        for i in b:
            res <<= 1
            if i==1:
                res+=1
        return res
    def getHashing(features):
        sums = []
        features = features.items()
        for f in features:
            f, w = f
            h = hashfunc(f.encode('utf-8'))# [-8:]
            b = bitarray_from_bytes(h)
            sums.append((([1]*128-b)*(-1)+b)*w)

        combined_sums = np.sum(sums, 0)
        result = [1 if v>0 else 0 for v in combined_sums]
        value = binary_to_int(result)
        return value
    start_time = time.time()
    query_ids = []
    query_hashs = []
    query_t_hashs = []
    query_items = query.items()
    for idx, (id_, v) in enumerate(query_items):
        query_ids.append(id_)
        query_hashs.append(getHashing(v["tfidf"]))
        if args.optim:
            query_t_hashs.append(getHashing(v["tfidf_t"]))
    print("calc real time: {}".format(time.time()-start_time))


    bag = defaultdict(list)
    
    for query_idx, (q_id, v) in enumerate(query_items):
        if len(bag) == 0:
            bag[q_id].append(query_idx)
            continue
        
        dis = 200
        sim_bid = ""
        sim_idx = 0
        for bid, group in bag.items():
            for idx in group:
                cur_dis = distance(query_hashs[query_idx], query_hashs[idx])
                if cur_dis<dis:
                    dis = cur_dis
                    sim_bid = bid
                    sim_idx = idx
        # print examples
        # if q_id in ['37505304', '37500704', '37043172', '37046160', '37043001', '37043011']:
        #     print("examples...")
        #     print("id: {}, idx: {}".format(q_id, query_idx))
            # if sim_bid != "":
            #     print("dis: {}, sim_id: {}".format(dis, list(query_items)[sim_idx][0]))
            # else:
            #     print("None!")
        # -------
        if dis<threshold:
            bag[sim_bid].append(query_idx)
        else:
            if args.optim:
                t_dis = distance(query_t_hashs[query_idx], query_t_hashs[sim_idx])
                cur_len = v["length"]
                sim_len = list(query_items)[sim_idx][1]["length"]
                diff_len = abs(cur_len-sim_len)/cur_len
                if t_dis<40:
                    bag[sim_bid].append(query_idx)
                elif diff_len<0.01:
                    cur_word_set = v["word_set"]
                    sim_word_set = list(query_items)[sim_idx][1]["word_set"]
                    inter_words = cur_word_set&sim_word_set
                    overlap = len(inter_words)/len(cur_word_set)
                    if overlap>0.8:
                        bag[sim_bid].append(query_idx)
                else:
                    bag[q_id].append(query_idx)
            else:
                bag[q_id].append(query_idx)
    # special cases
    # def print_info(i, j):
    #     dis = distance(query_hashs[i], query_hashs[j])
    #     cur_len = list(query_items)[i][1]["length"]
    #     sim_len = list(query_items)[j][1]["length"]
    #     diff_len = abs(cur_len-sim_len)/cur_len
    #     cur_word_set = list(query_items)[i][1]["word_set"]
    #     sim_word_set = list(query_items)[j][1]["word_set"]
    #     inter_words = cur_word_set&sim_word_set
    #     overlap = len(inter_words)/len(cur_word_set)

    #     cur_len_t = list(query_items)[i][1]["length_t"]
    #     sim_len_t = list(query_items)[j][1]["length_t"]
    #     diff_len_t = abs(cur_len_t-sim_len_t)/cur_len_t
    #     cur_word_set_t = list(query_items)[i][1]["word_set_t"]
    #     sim_word_set_t = list(query_items)[j][1]["word_set_t"]
    #     inter_words_t = cur_word_set_t&sim_word_set_t
    #     overlap_t = len(inter_words_t)/len(cur_word_set_t)

    #     print("{} and {}".format(list(query_items)[i][0], list(query_items)[i][0]))
    #     print("dis: {}, diff_len: {}/{}/{}, overlap: {}/{}/{}/{}".format(dis, diff_len, cur_len, \
    #     sim_len, overlap, len(cur_word_set), len(sim_word_set), len(inter_words)))
    #     print("diff_len_t: {}/{}/{}, overlap_t: {}/{}/{}/{}".format(diff_len_t, cur_len_t, \
    #     sim_len_t, overlap_t, len(cur_word_set_t), len(sim_word_set_t), len(inter_words_t)))
    # print_info(77, 78)
    # print_info(91, 92)
    # print_info(160, 161)
    # --------------


    return bag, list(query_items)
