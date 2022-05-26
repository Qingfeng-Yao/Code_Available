import numpy as np
import jieba
import utils
import collections
import hashlib
import jieba.posseg as psg

data = np.load('heybox/false_maps.npz', allow_pickle=True)
false_maps = data['false_maps'].tolist()
print(len(false_maps))
count_cluster = len(false_maps)

stopwords_path = "Simhash/static/stopwords.txt"
stopwords = utils.readStopwords(stopwords_path)

idf_path = "Simhash/static/idf.txt"
idf_map, ave_idf = utils.readIdfdict(idf_path)

def PartOfSpeech(text):
    seg = psg.cut(text)
    word_set = set()
    for s in seg:
        pair = str(s).split('/')
        # assert len(pair) == 2 , "{}".format(pair)
        if pair[-1] in ['v', 'vn', 'n', 'nt', 'ns', 'nr', 'nrfg', 'nrt', 'nz']:
            word_set.add(pair[0])
    return word_set

def deal(content):
    seg_list = jieba.cut(content.strip())
    ret = []
    for word in seg_list:
        word = word.strip()
        if word not in stopwords:
            if word != "\t" and word != "\n":
                ret.append(word)
    return ' '.join(ret), ret

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

theta = 0
haiming = 30

eta = 0
diff_len = 0.1
overlap = 0.8
for _, v in false_maps.items():
    query_t_hashs = []
    lengths = []
    word_sets = []
    for _, m in v.items():
        title = m["title"]
        # print(title)
        text_t, list_t = deal(title)
        tfidf_map_t = collections.Counter(list_t)
        for k in tfidf_map_t.keys():
            if k in idf_map:
                tfidf_map_t[k] *= idf_map[k]
            else:
                tfidf_map_t[k] *= ave_idf
        tfidf_t = tfidf_map_t
        query_t_hashs.append(getHashing(tfidf_t))

        content = m["title"] + '\n' + m["body"]
        length = len(content)
        lengths.append(length)
        word_set = PartOfSpeech(content)
        word_sets.append(word_set)

    count_total = 0
    count_true_1 = 0
    count_true_2 = 0
    for i in range(len(query_t_hashs)):
        for j in range(len(query_t_hashs)):
            if i>=j:
                continue
            t_dis = utils.distance(query_t_hashs[i], query_t_hashs[j])
            diff_len_t = abs(lengths[i]-lengths[j])/lengths[i]
            inter_words_t = word_sets[i]&word_sets[j]
            overlap_t = len(inter_words_t)/len(word_sets[i])
            count_total += 1
            if t_dis<haiming:
                count_true_1 += 1
            if diff_len_t<diff_len and overlap_t>overlap:
                count_true_2 += 1
    theta += (count_true_1/count_total)
    eta += (count_true_2/count_total)
print("haiming: {}, theta: {}".format(haiming, theta/count_cluster))
print("diff_len: {}, overlap: {}, eta: {}".format(diff_len, overlap, eta/count_cluster))





		