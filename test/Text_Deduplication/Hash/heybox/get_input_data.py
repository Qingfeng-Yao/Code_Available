import os
import json

chinese_path = "chinese_data"
input_path = "input_data"

if not os.path.exists(input_path):
    os.makedirs(input_path)

input_dict = {}
for root, dirs, files in os.walk(chinese_path):
    for f in files:
        if f[-3:] == 'txt':
            file_path = root +'/'+ f
            input_dict[f.split('.')[0]] = {}

            with open(file_path,'r',encoding='utf-8') as load_f:
                text = load_f.read()
                input_dict[f.split('.')[0]]["content"] = text
            
print("total documents: {}".format(len(input_dict)))
with open(input_path+'/data.json','w',encoding='utf-8') as w_f:
    json.dump(input_dict, w_f, indent=4)
                                
