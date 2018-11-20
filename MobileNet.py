import pandas as pd
import numpy as np
from pathlib import Path
from fastai import *
from fastai.vision import *
from utils import *

NUM_VAL = 50 * 340

PATH = Path('../data/quickdraw/')

bs = 200
sz = 256
test_df = pd.read_csv(PATH/"test_simplified.csv")

def create_func(item):
    with open(item) as f: item = f.read()
    arr = list2drawing(eval(item), size=sz, lw=6, time_color=True)
    img = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    tensor = torch.from_numpy(img).float()
    return Image(tensor.permute((2,0,1)).div_(255))

item_list = ItemList.from_folder(PATH/"train_folders", create_func=create_func)
np.random.seed(42)
idxs = np.arange(item_list.items.shape[0])
np.random.shuffle(idxs)
val_idxs = idxs[:NUM_VAL]
item_lists = item_list.split_by_idx(val_idxs)
label_lists = item_lists.label_from_folder()
test_items = ItemList.from_df(test_df, cols=['drawing', 'key_id'], create_func=create_func)
label_lists.add_test(test_items);

train_dl = DataLoader(label_lists.train, bs, True, num_workers=4)
valid_dl = DataLoader(label_lists.valid, bs, False, num_workers=4)
test_dl = DataLoader(label_lists.test, bs, False, num_workers=4)

data_bunch = DataBunch(train_dl, valid_dl, test_dl)


import sys
sys.path.append("pytorch-mobilenet-v2/")
from MobileNetV2 import MobileNetV2
model = MobileNetV2(340)
learn = Learner(data_bunch, model, metrics=[accuracy, map3])
learn.fit_one_cycle(1,max_lr=5e-3)

name = 'mobilenet'
learn.save(f'{name}-stage-1')
