# -*- coding: utf-8 -*-
from .. import setting
from . import cut_word
from .. import utils

import jieba

import collections
import jieba.posseg as psg

@utils.get_run_time
def extractKeyWord(raw_data, args):
    data = {}
    c = cut_word.CutWord(setting.STOPWORDS_PATH)
    if args.model_type == "keywords":
        idf_map, ave_idf = utils.readIdfdict(setting.IDFDICT_PATH)
    for id_, text in raw_data.items():
        content = text["content"]
        d = {}
        d["raw"] = content
        if not args.thesis:
            d["target"] = text["target"]
        d["text"], d["list"] = c.deal(content)
        # 关键词权重计算
        if args.model_type == "keywords":
            tfidf_map = collections.Counter(d["list"])
            for k in tfidf_map.keys():
                if k in idf_map:
                    tfidf_map[k] *= idf_map[k]
                else:
                    tfidf_map[k] *= ave_idf
            d["tfidf"] = tfidf_map
        if args.optim:
            d["length"] = text["length"]

            title = text["title"]
            d["length_t"] = len(title)
            d["raw_t"] = title
            d["text_t"], d["list_t"] = c.deal(title)
            tfidf_map_t = collections.Counter(d["list_t"]) 
            if len(tfidf_map_t) == 0:
                continue
            for k in tfidf_map_t.keys():
                if k in idf_map:
                    tfidf_map_t[k] *= idf_map[k]
                else:
                    tfidf_map_t[k] *= ave_idf
            d["tfidf_t"] = tfidf_map_t

            def PartOfSpeech(text):
                seg = psg.cut(text)
                word_set = set()
                for s in seg:
                    pair = str(s).split('/')
                    assert len(pair) == 2
                    if pair[-1] in ['v', 'vn', 'n', 'nt', 'ns', 'nr', 'nrfg', 'nrt', 'nz']:
                        word_set.add(pair[0])
                return word_set
            d["word_set_t"] = PartOfSpeech(d["text_t"])
            d["word_set"] = PartOfSpeech(d["text"])
        data[id_] = d
    return data