import os
import jieba

def readFile(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        return text

def get_jsons(json_dir):
    json_list = []
    for j in os.listdir(json_dir):
        j_dir = os.path.join(json_dir,j)
        text = readFile(j_dir)
        json_list.append(text)

    return json_list

def clean_text(text, stopwords):
    seg_list = jieba.cut(text.strip())
    re_list = []
    for word in seg_list:
        word = word.strip()
        if word not in stopwords:
            if word != "\t" and word != "\n":
                re_list.append(word)
    return ' '.join(re_list)

def readStopwords(path):
    stopwords = []
    with open(path, "r", encoding="utf-8") as f:
        stopwords = [word.strip() for word in f.readlines()]
    return stopwords