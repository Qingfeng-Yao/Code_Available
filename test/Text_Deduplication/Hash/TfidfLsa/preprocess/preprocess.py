# -*- coding: utf-8 -*-
from .. import setting
from . import cut_word
from .. import utils

@utils.get_run_time
def cutWord(raw_data):
    data = {}
    c = cut_word.CutWord(setting.STOPWORDS_PATH)
    for id_, text in raw_data.items():
        content = text["content"]
        d = {}
        d["raw"] = content
        d["text"] = c.deal(content)
        data[id_] = d
    return data