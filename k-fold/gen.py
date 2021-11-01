import random
import numpy as np
import pandas as pd
import json
import shutil
import os
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy

# load json
json_path = '/opt/ml/segmentation/input/data/train_modified.json' # train_all_modified.json
save_path = '/opt/ml/segmentation/input/jsons/' # batch_i.json's file path
train_dict = dict()

with open(json_path, 'r') as f:
    train_dict = json.load(f)


# precalc info
idx_to_image_info = dict()
idx_to_annotations_info = defaultdict(list)
idx_to_categories = defaultdict(list)

for i in train_dict['images']:
    idx_to_image_info[int(i['id'])] = i

for i in train_dict['annotations']:
    idx_to_annotations_info[int(i['image_id'])].append(i)
    idx_to_categories[int(i['image_id'])].append(int(i['category_id']))

def ArgMax(v):
    cnt = [0] * 11
    for i in v: cnt[i] += 1
    mx = int(-1e9)
    for i in range(1, 11): mx = max(mx, cnt[i])
    L = []
    for i in range(1, 11):
        if cnt[i] == mx: L.append(i)
    return random.choice(L)

def ArgMin(v):
    cnt = [0] * 11
    for i in v: cnt[i] += 1
    mn = int(1e9)
    for i in range(1, 11): mn = min(mn, cnt[i])
    L = []
    for i in range(1, 11):
        if cnt[i] == mn: L.append(i)
    return random.choice(L)

idx_to_major_category = [0] * 3272
for i in range(3272):
    idx_to_major_category[i] = ArgMin(idx_to_categories[i])


# bucket
bucket = [[] for _ in range(11)]
for i in range(3272):
    if idx_to_image_info.get(i) == None: continue
    bucket[idx_to_major_category[i]].append(i)
    
kfold = [[] for _ in range(4)]
for i in range(1, 11):
    random.shuffle(bucket[i])
    for j in range(len(bucket[i])):
        kfold[j % 4].append(bucket[i][j])

for i in range(4):
    kfold[i].sort()

# make json
jsons = [
    {
        'info': train_dict['info'],
        'licenses': train_dict['licenses'],
        'categories': train_dict['categories'],
        'images': [],
        'annotations': []
    }
    for _ in range(4)]


for i in range(4):
    for idx in kfold[i]:
        jsons[i]['images'].append(idx_to_image_info[idx])
        for annotation_info in idx_to_annotations_info[idx]:
            jsons[i]['annotations'].append(annotation_info)


# save jsons

for i in range(4):
    with open(save_path + str(i + 1) + ".json", 'w') as f:
        json.dump(jsons[i], f)