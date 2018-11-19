import pandas as pd
import numpy as np
from pathlib import Path
from fastai import *
from fastai.vision import *
from utils import *
import ast

NUM_SAMPLES_PER_CLASS = 100000
NUM_VAL = 50 * 340
PATH = Path('../data/quickdraw/')

bs = 150
sz = 256

dfs_combined = pd.read_csv(PATH/"train/dfs_combined.csv")
test_df = pd.read_csv(PATH/"test_simplified.csv")
dfs_combined = dfs_combined[['drawing', 'word']]

def create_func(item):
    arr = list2drawing(eval(item[0]), size=sz, lw=6, time_color=True)
    img = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    tensor = torch.from_numpy(img).float()
    return Image(tensor.permute((2,0,1)).div_(255))

item_list = ItemList.from_df(dfs_combined, cols=['drawing', 'word'], create_func=create_func)
np.random.seed(42)
idxs = np.arange(item_list.items.shape[0])
np.random.shuffle(idxs)
val_idxs = idxs[:NUM_VAL]
item_lists = item_list.split_by_idx(val_idxs)
label_lists = item_lists.label_from_df(cols=1)
test_items = ItemList.from_df(test_df, cols=['drawing', 'key_id'], create_func=create_func)
label_lists.add_test(test_items);

train_dl = DataLoader(label_lists.train, bs, True)
valid_dl = DataLoader(label_lists.valid, bs, False)
test_dl = DataLoader(label_lists.test, bs, False)
data_bunch = ImageDataBunch(train_dl, valid_dl, test_dl)


import sys
sys.path.append("pytorch-mobilenet-v2/")
from MobileNetV2 import MobileNetV2
model = MobileNetV2(340)
learn = Learner(data_bunch, model, metrics=[accuracy, map3])
learn.fit_one_cycle(1,max_lr=5e-3)

name = 'mobilenet'
learn.save(f'{name}-stage-1')