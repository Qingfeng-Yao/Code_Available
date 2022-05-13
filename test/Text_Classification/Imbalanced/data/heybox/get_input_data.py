# -*- coding:utf-8 -*-
import csv
import collections

from preprocess import *


'''
不平衡数据的组织
    刀塔霸业:            total/171, train/50, test/100, extra/21
    命运2:              total/227, train/89, test/100, extra/38
    怪物猎人世界:         total/302, train/142, test/100, extra/60
    炉石传说：魔兽英雄传:  total/304, train/143, test/100, extra/61
    刀塔自走棋:          total/326, train/159, test/100, extra/67
    魔兽世界:            total/368, train/188, test/100, extra/80
    彩虹六号围攻:         total/512, train/289, test/100, extra/123
    数码硬件:            total/513, train/290, test/100, extra/123
    守望先锋:            total/821, train/505, test/100, extra/216
    云顶之弈:            total/878, train/545, test/100, extra/233
    刀塔2:              total/1052, train/667, test/100, extra/285
    绝地求生:            total/1059, train/672, test/100, extra/287
    CS:GO:              total/1271, train/820, test/100, extra/351
    手机游戏:            total/1488, train/972, test/100, extra/416
    英雄联盟:            total/1776, train/1174, test/100, extra/502
    盒友杂谈:             total/2450, train/1645, test/100, extra/705
    主机游戏:             total/3038, train/2057, test/100, extra/881
    PC游戏:              total/3085, train/2090, test/100, extra/895
'''

stopwords_path = 'stopwords.txt'
stopwords = readStopwords(stopwords_path)

game_name_dict = {"彩虹六号围攻": "rainbow", "刀塔2": "daota2", "刀塔霸业": "daotabaye", "刀塔自走棋": "daotazizouqi", \
    "怪物猎人世界": "monster", "盒友杂谈": "zatan", "绝地求生": "qiusheng", "炉石传说：魔兽英雄传": "lushi", "命运2": "mingyun2",\
        "魔兽世界": "world", "手机游戏": "mobile", "守望先锋": "xianfeng", "数码硬件": "hardware", "英雄联盟": "union", \
            "云顶之弈": "cloud", "主机游戏": "zhuji", "CS:GO": "csgo", "PC游戏": "pc"}
label_dict= {"rainbow":0, "daota2":1, "daotabaye":2, "daotazizouqi":3, "monster":4, "zatan":5, "qiusheng":6, \
    "lushi":7, "mingyun2":8, "world":9, "mobile":10, "xianfeng":11, "hardware":12, "union":13, "cloud":14, "zhuji":15, "csgo":16, "pc":17}
game_dict = {}
data_dir = "raw_data"
for k, v in game_name_dict.items():
    game_dict[k] = {}
    game_dict[k]["english_abbr_name"] = v
    game_dict[k]["content_list"] = get_jsons(os.path.join(data_dir, k))
    game_dict[k]["test"] = game_dict[k]["content_list"][-20:]
    res_num = len(game_dict[k]["content_list"][:-20])
    train_num = int(0.3*res_num)
    extra_half_num = int(0.45*res_num)
    extra_same_num = int(0.6*res_num)
    extra_onehalf_num = int(0.75*res_num)
    extra_double_num = int(0.9*res_num)
    game_dict[k]["train"] = game_dict[k]["content_list"][:train_num]
    game_dict[k]["extra_all"] = game_dict[k]["content_list"][train_num:-20]
    game_dict[k]["extra_half"] = game_dict[k]["content_list"][train_num:extra_half_num]
    game_dict[k]["extra_same"] = game_dict[k]["content_list"][train_num:extra_same_num]
    game_dict[k]["extra_onehalf"] = game_dict[k]["content_list"][train_num:extra_onehalf_num]
    game_dict[k]["extra_double"] = game_dict[k]["content_list"][train_num:extra_double_num]
sorted_dict = collections.OrderedDict(sorted(game_dict.items(), key=lambda t: len(t[1]["content_list"])))
for k, v in sorted_dict.items():
    print("{}: total/{}, train/{}, test/{}, extra_all/{}".format(k, len(v["content_list"]), len(v["train"]), len(v["test"]), len(v["extra_all"])))
    print("{}: extra_half/{}, extra_same/{}, extra_onehalf/{}, extra_double/{}".format(k, len(v["extra_half"]), len(v["extra_same"]), len(v["extra_onehalf"]), len(v["extra_double"])))

train_label, data_train, test_label, data_test, extra_all_label, data_extra_all = [], [], [], [], [], []
extra_half_label, data_extra_half, extra_same_label, data_extra_same, extra_onehalf_label, data_extra_onehalf, extra_double_label, data_extra_double = [], [], [], [], [], [], [], []
for k, v in game_dict.items():
    label = label_dict[v["english_abbr_name"]]
    for tr in v["train"]:
        data_train.append(clean_text(tr, stopwords))
        train_label.append(label)
    for te in v["test"]:
        data_test.append(clean_text(te, stopwords))
        test_label.append(label)
    for ex in v["extra_all"]:
        data_extra_all.append(clean_text(ex, stopwords))
        extra_all_label.append(label)
    for ex in v["extra_half"]:
        data_extra_half.append(clean_text(ex, stopwords))
        extra_half_label.append(label)
    for ex in v["extra_same"]:
        data_extra_same.append(clean_text(ex, stopwords))
        extra_same_label.append(label)
    for ex in v["extra_onehalf"]:
        data_extra_onehalf.append(clean_text(ex, stopwords))
        extra_onehalf_label.append(label)
    for ex in v["extra_double"]:
        data_extra_double.append(clean_text(ex, stopwords))
        extra_double_label.append(label)

output_path = "input_data/"
if not os.path.exists(output_path):
    os.mkdir(output_path)

with open(output_path+'train.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['', 'label', 'text'])
    for i in range(len(data_train)):
        tsv_w.writerow([i, train_label[i], data_train[i]]) 

with open(output_path+'test.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['', 'label', 'text'])
    for i in range(len(data_test)):
        tsv_w.writerow([i, test_label[i], data_test[i]]) 

with open(output_path+'extra_all.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['', 'label', 'text'])
    for i in range(len(data_extra_all)):
        tsv_w.writerow([i, extra_all_label[i], data_extra_all[i]]) 

with open(output_path+'extra_half.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['', 'label', 'text'])
    for i in range(len(data_extra_half)):
        tsv_w.writerow([i, extra_half_label[i], data_extra_half[i]]) 

with open(output_path+'extra_same.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['', 'label', 'text'])
    for i in range(len(data_extra_same)):
        tsv_w.writerow([i, extra_same_label[i], data_extra_same[i]]) 

with open(output_path+'extra_onehalf.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['', 'label', 'text'])
    for i in range(len(data_extra_onehalf)):
        tsv_w.writerow([i, extra_onehalf_label[i], data_extra_onehalf[i]]) 

with open(output_path+'extra_double.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['', 'label', 'text'])
    for i in range(len(data_extra_double)):
        tsv_w.writerow([i, extra_double_label[i], data_extra_double[i]]) 
