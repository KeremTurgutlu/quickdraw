import pandas as pd
import numpy as np
from pathlib import Path
from fastai import *
from fastai.vision import *
from utils import *

NUM_SAMPLES_PER_CLASS = 100000
NUM_VAL = 50 * 340

PATH = Path('../data/quickdraw/')

bs = 400
sz = 256

dfs_combined = pd.read_csv(PATH/"train/dfs_combined.csv")
test_df = pd.read_csv(PATH/"test_simplified.csv")
dfs_combined = dfs_combined[['drawing', 'word']]

def create_func(item):
    img = list2multichannel(eval(item[0]), size=225)
    tensor = torch.from_numpy(img).float()
    return tensor.div_(255)

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

data_bunch = DataBunch(train_dl, valid_dl, test_dl)


def conv_layer(ni:int, nf:int, ks:int=3, stride:int=1, pad=0)->nn.Sequential:
    "Create Conv2d->BatchNorm2d->LeakyReLu layer: `ni` input, `nf` out filters, `ks` kernel, `stride`:stride."
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=pad),
        nn.BatchNorm2d(nf),
        nn.ReLU())

class SketchNet(nn.Module):
    def __init__(self, num_classes=340):
        super(SketchNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = conv_layer(6, 64, 15, stride=3, pad=0)
        self.conv2 = conv_layer(64, 128, 5, stride=1, pad=0)
        self.conv3 = conv_layer(128, 256, 3, stride=1, pad=1)
        self.conv4 = conv_layer(256, 256, 3, stride=1, pad=1)
        self.conv5 = conv_layer(256, 256, 3, stride=1, pad=1)
        self.conv6 = nn.Conv2d(256, 512, 7, stride=1)
        self.conv7 = nn.Conv2d(512, 512, 1, stride=1)
        self.conv8 = nn.Conv2d(512, num_classes, 1, stride=1)        
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv5(self.conv4(self.conv3(x))))
        x = self.drop1(F.relu(self.conv6(x)))
        x = self.drop2(F.relu(self.conv7(x)))
        x = self.conv8(x)
        return x.view(x.shape[0], -1)

model = SketchNet()
learn = Learner(data_bunch, model, metrics=[accuracy, map3])
learn.fit_one_cycle(1, max_lr=1e-2)

name = "sketch-a-net"
learn.save(f'{name}-stage-1')