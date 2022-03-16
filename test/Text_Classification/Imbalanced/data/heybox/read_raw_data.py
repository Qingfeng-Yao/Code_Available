# -*- coding:utf-8 -*-
import collections

from preprocess import *

'''
原始heybox数据主题类别数据统计
    Total 18 classes...
    Total num: 19641
    刀塔霸业: 171
    命运2: 227
    怪物猎人世界: 302
    炉石传说：魔兽英雄传: 304
    刀塔自走棋: 326
    魔兽世界: 368
    彩虹六号围攻: 512
    数码硬件: 513
    守望先锋: 821
    云顶之弈: 878
    刀塔2: 1052
    绝地求生: 1059
    CS:GO: 1271
    手机游戏: 1488
    英雄联盟: 1776
    盒友杂谈: 2450
    主机游戏: 3038
    PC游戏: 3085
'''

data_dir = "raw_data"

game_name_dict = {"彩虹六号围攻": "rainbow", "刀塔2": "daota2", "刀塔霸业": "daotabaye", "刀塔自走棋": "daotazizouqi", \
    "怪物猎人世界": "monster", "盒友杂谈": "zatan", "绝地求生": "qiusheng", "炉石传说：魔兽英雄传": "lushi", "命运2": "mingyun2",\
        "魔兽世界": "world", "手机游戏": "mobile", "守望先锋": "xianfeng", "数码硬件": "hardware", "英雄联盟": "union", \
            "云顶之弈": "cloud", "主机游戏": "zhuji", "CS:GO": "csgo", "PC游戏": "pc"}
game_dict = {}
print("Total {} classes...".format(len(game_name_dict)))
total_num = 0
for k, v in game_name_dict.items():
    game_dict[k] = {}
    game_dict[k]["english_abbr_name"] = v
    game_dict[k]["content_list"] = get_jsons(os.path.join(data_dir, k))
    total_num += len(game_dict[k]["content_list"])
print("Total num: {}".format(total_num))
sorted_dict = collections.OrderedDict(sorted(game_dict.items(), key=lambda t: len(t[1]["content_list"])))
for k, v in sorted_dict.items():
    print("{}: {}".format(k, len(v["content_list"])))