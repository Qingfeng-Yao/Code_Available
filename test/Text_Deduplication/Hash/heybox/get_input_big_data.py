import os
import json

raw_path = "raw_big_data"
input_path = "input_data"

num_jsonfiles = 0
uni_jsonfiles = set()
num_single_dir = 0
num_multi_dir = 0
num_dir = 0

content_len = 0
max_len = 0
min_len = 100000

if not os.path.exists(input_path):
    os.makedirs(input_path)

input_dict = {}
for root, dirs, files in os.walk(raw_path):
    if len(dirs) != 0:
        num_dir = len(dirs)
        print("raw_data has {} classes".format(num_dir))
    if len(dirs) == 0:
        if len(files) == 1:
            num_single_dir += 1
        if len(files) > 1:
            num_multi_dir += 1
        for f in files:
            if f[-4:] == 'json':
                num_jsonfiles += 1
                uni_jsonfiles.add(f)

                file_path = root +'/'+ f
                input_dict[f.split('.')[0]] = {}

                with open(file_path,'r',encoding='utf-8') as load_f:
                    load_dict = json.load(load_f)

                    text = load_dict["title"] + '\n' + load_dict["content"]
                    current_len = len(text)
                    content_len += current_len
                    if current_len > max_len:
                        max_len = current_len
                    if current_len < min_len:
                        min_len = current_len

                    input_dict[f.split('.')[0]]["body"] = load_dict["content"]
                    input_dict[f.split('.')[0]]["title"] = load_dict["title"] 
                    input_dict[f.split('.')[0]]["content"] = text
                    input_dict[f.split('.')[0]]["length"] = len(text)
            
print("total documents: {}".format(len(input_dict)))
assert num_jsonfiles == len(uni_jsonfiles)
assert num_dir == (num_single_dir+num_multi_dir)
print("num json files: {}".format(num_jsonfiles))
print("{} dirs contain single file".format(num_single_dir))
print("{} dirs contain multi files".format(num_multi_dir))
print("ave file content(including title) len: {}".format(content_len/num_jsonfiles))
print("max file len: {}".format(max_len))
print("min file len: {}".format(min_len))
with open(input_path+'/big_data.json','w',encoding='utf-8') as w_f:
    json.dump(input_dict, w_f, indent=4)
                                
