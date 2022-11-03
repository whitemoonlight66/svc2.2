import re

import demjson


def replace_id(f_list, sp_id):
    if re.search('wav[|](\d+)[|]', f_list[0]):
        split_char = "wav|%s|" % re.search('wav[|](\d+)[|]', f_list[0]).group(1)
    else:
        split_char = "wav|"
    f_list = [x.replace(split_char, f"wav|{sp_id}|") for x in f_list]
    return f_list


def make_list(file_name):
    train_f = open(f"./{file_name}", "w", encoding="utf-8")
    for i in range(pre, len(speakers) + pre):
        with open(f"{dataset_path}/{speakers[i]}/{file_name}", "r", encoding="utf-8") as f:
            file_list = f.readlines()
            train_f.writelines(replace_id(file_list, i))
    train_f.close()


# data文件夹下放，各个 工程名/wavs/train(val)(这层可以不要)/xxx.wav
# 自动读取每个工程文件夹下的train(val).txt(xxx.wav|无符号文本)
pre = 0  # 如果使用预模型、id从10开始，预模型人数
dataset_path = "./dataset"  # 数据合集目录
file_train_name = "train.txt"
file_val_name = "val.txt"
# configs文件夹下放配置文件，json的speaker键放人物列表，人物名对应文件夹名，自动按顺序分配id
json_file = "nyarumul.json"
config = demjson.decode_file(f"./{json_file}")
speakers = config["speakers"] if 'speakers' in config.keys() else None

print(speakers)
make_list(file_train_name)
make_list(file_val_name)
