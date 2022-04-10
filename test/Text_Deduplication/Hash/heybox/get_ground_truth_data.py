import os

chinese_path = "chinese_data"
gt_path = "ground_truth"

if not os.path.exists(gt_path):
	os.makedirs(gt_path)
num_classes = 0
num_files = 0
num_single = 0
num_multi = 0

for root, dirs, files in os.walk(chinese_path):
    text = ""
    for f in files:
        if f[-3:] == 'txt':
            num_files += 1
            file_path = root +'/'+ f
            text += (f+"\n")
            with open(file_path,'r',encoding='utf-8') as load_f:
                text += load_f.read()
            text += "\n\n"
                    
    if text != "": 
        if len(files) == 1:
            num_single += 1
        if len(files) > 1:
            num_multi += 1
        new_file_path = gt_path+'/'+str(len(files))+'-'+root.split('/')[-1]+'.txt'
        with open(new_file_path,'w',encoding='utf-8') as w_f:
                w_f.write(text)
        num_classes += 1

print("total classes: {}".format(num_classes))
print("total files: {}".format(num_files))
print("single file dirs: {}".format(num_single))
print("multi file dirs: {}".format(num_multi))