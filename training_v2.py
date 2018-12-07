import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from fastai import *
from fastai.vision import *

import json

from utils import *
import ast


PATH = Path('../data/quickdraw/')

from collections import defaultdict
class Splitter(object):
    def __init__(self, valid_sz=100):
        self.class_counts = defaultdict(int)
        self.valid_sz = valid_sz
        
    def split(self, item):
        c = item.parent.name
        if self.class_counts[c] < self.valid_sz:
            self.class_counts[c] += 1
            return True
        else: return False
        
itemlist = pd.read_pickle(PATH/"itemlist")
splitter = Splitter(valid_sz=100)
itemlists = itemlist.split_by_valid_func(splitter.split)              
labellists = itemlists.label_from_folder()
test_items = ImageItemList.from_folder(PATH/'test')
labellists = labellists.add_test(test_items)
tfms = get_transforms(do_flip=True, max_rotate=15, max_zoom=1.1, max_warp=0, 
                      max_lighting=0, p_affine=0.5, p_lighting=0)
labellists = labellists.transform(tfms=tfms, size=256)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class RandomSamplerWithEpochSize(Sampler):
    """Yields epochs of specified sizes. Iterates over all examples in a data_source in random
    order. Ensures (nearly) all examples have been trained on before beginning the next iteration
    over the data_source - drops the last epoch that would likely be smaller than epoch_size.
    """
    def __init__(self, data_source, epoch_size):
        self.n = len(data_source)
        self.epoch_size = epoch_size
        self._epochs = []
    def __iter__(self):
        return iter(self.next_epoch)
    @property
    def next_epoch(self):
        if len(self._epochs) == 0: self.generate_epochs()
        return self._epochs.pop()
    def generate_epochs(self):
        idxs = [i for i in range(self.n)]
        np.random.shuffle(idxs)
        self._epochs = list(chunks(idxs, self.epoch_size))[:-1]
    def __len__(self):
        return self.epoch_size
    
bs=300
epoch_size = 5_000_000

train_dl = DataLoader(labellists.train, num_workers=8,
    batch_sampler=BatchSampler(RandomSamplerWithEpochSize(labellists.train, epoch_size),
                               bs, True))

valid_dl = DataLoader(labellists.valid, bs, False, num_workers=8)
test_dl = DataLoader(labellists.test, bs, False, num_workers=8)
data_bunch = ImageDataBunch(train_dl, valid_dl, test_dl)


name = 'resnet34'
from fastai.callbacks import SaveModelCallback
import sys
learn = create_cnn(data_bunch, arch=models.resnet34, metrics=[accuracy, map3],
                  callback_fns=[partial(SaveModelCallback, every="epoch", name=f"{name}_final")])
learn.load(f'{name}-stage-2-128');
print("Loaded, training...")
learn.freeze_to(-1)
learn.fit_one_cycle(20, max_lr=slice(1e-6, 3e-3), wd=1e-4)
learn.save(f'{name}-stage-1-256')












