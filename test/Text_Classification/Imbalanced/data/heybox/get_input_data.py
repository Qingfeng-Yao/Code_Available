# -*- coding:utf-8 -*-
import csv
import collections

from preprocess import *


'''
不平衡数据的组织
    
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
    game_dict[k]["train_extra"] = game_dict[k]["content_list"][:-20]
    game_dict[k]["extra_b_60"] = game_dict[k]["content_list"][train_num:train_num+60] # 各个类别取60

    # 总数与extra_b_60一致, 70-106分别对应最小类的样本数
    if v == "rainbow":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+62]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+65]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+67]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+70]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+72]
    elif v == "daota2":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+59]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+57]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+57]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+55]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+54]
    elif v == "daotabaye":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+70]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+80]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+90]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+100]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+106]
    elif v == "daotazizouqi":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+64]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+70]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+75]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+80]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+83]
    elif v == "monster":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+66]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+75]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+83]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+90]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+94]
    elif v == "zatan":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+54]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+45]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+37]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+30]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+26]
    elif v == "qiusheng":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+58]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+55]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+53]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+50]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+48]
    elif v == "lushi":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+65]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+73]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+79]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+85]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+88]
    elif v == "mingyun2":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+68]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+78]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+87]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+95]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+100]
    elif v == "world":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+63]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+68]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+71]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+75]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+77]
    elif v == "mobile":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+56]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+50]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+45]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+40]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+37]
    elif v == "xianfeng":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+60]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+60]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+60]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+60]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+60]
    elif v == "cloud":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+60]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+60]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+60]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+60]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+60]
    elif v == "hardware":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+61]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+63]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+63]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+65]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+66]
    elif v == "union":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+55]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+47]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+41]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+35]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+32]
    elif v == "zhuji":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+52]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+42]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+33]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+25]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+20]
    elif v == "csgo":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+57]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+52]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+49]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+45]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+43]
    elif v == "pc":
        game_dict[k]["extra_b_60_70"] = game_dict[k]["content_list"][train_num:train_num+50]
        game_dict[k]["extra_b_60_80"] = game_dict[k]["content_list"][train_num:train_num+40]
        game_dict[k]["extra_b_60_90"] = game_dict[k]["content_list"][train_num:train_num+30]
        game_dict[k]["extra_b_60_100"] = game_dict[k]["content_list"][train_num:train_num+20]
        game_dict[k]["extra_b_60_106"] = game_dict[k]["content_list"][train_num:train_num+14]
    

    game_dict[k]["train"] = game_dict[k]["content_list"][:train_num]
    game_dict[k]["extra_all"] = game_dict[k]["content_list"][train_num:-20]

sorted_dict = collections.OrderedDict(sorted(game_dict.items(), key=lambda t: len(t[1]["content_list"])))
for k, v in sorted_dict.items():
    print("{}: total/{}, train/{}, test/{}, extra_all/{}".format(k, len(v["content_list"]), len(v["train"]), len(v["test"]), len(v["extra_all"])))


train_extra_label, data_train_extra = [], []
extra_b_60_label, data_extra_b_60 = [], []
extra_b_60_70_label, data_extra_b_60_70, extra_b_60_80_label, data_extra_b_60_80, extra_b_60_90_label, data_extra_b_60_90, extra_b_60_100_label, data_extra_b_60_100, extra_b_60_106_label, data_extra_b_60_106 = [], [], [], [], [], [], [], [], [], []
train_label, data_train, test_label, data_test, extra_all_label, data_extra_all = [], [], [], [], [], []
for k, v in game_dict.items():
    label = label_dict[v["english_abbr_name"]]
    for tr in v["train_extra"]:
        data_train_extra.append(clean_text(tr, stopwords))
        train_extra_label.append(label)

    for tr in v["extra_b_60"]:
        data_extra_b_60.append(clean_text(tr, stopwords))
        extra_b_60_label.append(label)

    for tr in v["extra_b_60_70"]:
        data_extra_b_60_70.append(clean_text(tr, stopwords))
        extra_b_60_70_label.append(label)
    for tr in v["extra_b_60_80"]:
        data_extra_b_60_80.append(clean_text(tr, stopwords))
        extra_b_60_80_label.append(label)
    for tr in v["extra_b_60_90"]:
        data_extra_b_60_90.append(clean_text(tr, stopwords))
        extra_b_60_90_label.append(label)
    for tr in v["extra_b_60_100"]:
        data_extra_b_60_100.append(clean_text(tr, stopwords))
        extra_b_60_100_label.append(label)
    for tr in v["extra_b_60_106"]:
        data_extra_b_60_106.append(clean_text(tr, stopwords))
        extra_b_60_106_label.append(label)

    for tr in v["train"]:
        data_train.append(clean_text(tr, stopwords))
        train_label.append(label)
    for te in v["test"]:
        data_test.append(clean_text(te, stopwords))
        test_label.append(label)
    for ex in v["extra_all"]:
        data_extra_all.append(clean_text(ex, stopwords))
        extra_all_label.append(label)

output_path = "input_data/"
if not os.path.exists(output_path):
    os.mkdir(output_path)

with open(output_path+'train_extra.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['index', 'label', 'text'])
    for i in range(len(data_train_extra)):
        tsv_w.writerow([i, train_extra_label[i], data_train_extra[i]]) 

with open(output_path+'extra_b_60.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['index', 'label', 'text'])
    for i in range(len(data_extra_b_60)):
        tsv_w.writerow([i, extra_b_60_label[i], data_extra_b_60[i]]) 

with open(output_path+'extra_b_60_70.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['index', 'label', 'text'])
    for i in range(len(data_extra_b_60_70)):
        tsv_w.writerow([i, extra_b_60_70_label[i], data_extra_b_60_70[i]]) 

with open(output_path+'extra_b_60_80.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['index', 'label', 'text'])
    for i in range(len(data_extra_b_60_80)):
        tsv_w.writerow([i, extra_b_60_80_label[i], data_extra_b_60_80[i]]) 

with open(output_path+'extra_b_60_90.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['index', 'label', 'text'])
    for i in range(len(data_extra_b_60_90)):
        tsv_w.writerow([i, extra_b_60_90_label[i], data_extra_b_60_90[i]]) 

with open(output_path+'extra_b_60_100.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['index', 'label', 'text'])
    for i in range(len(data_extra_b_60_100)):
        tsv_w.writerow([i, extra_b_60_100_label[i], data_extra_b_60_100[i]]) 

with open(output_path+'extra_b_60_106.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['index', 'label', 'text'])
    for i in range(len(data_extra_b_60_106)):
        tsv_w.writerow([i, extra_b_60_106_label[i], data_extra_b_60_106[i]]) 

with open(output_path+'train.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['index', 'label', 'text'])
    for i in range(len(data_train)):
        tsv_w.writerow([i, train_label[i], data_train[i]]) 

with open(output_path+'test.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['index', 'label', 'text'])
    for i in range(len(data_test)):
        tsv_w.writerow([i, test_label[i], data_test[i]]) 

with open(output_path+'extra_all.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['index', 'label', 'text'])
    for i in range(len(data_extra_all)):
        tsv_w.writerow([i, extra_all_label[i], data_extra_all[i]]) 
