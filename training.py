import pandas as pd
import numpy as np
from pathlib import Path
from fastai import *
from fastai.vision import *
from utils import *
import sys

NUM_VAL = 50 * 340

PATH = Path('../data/quickdraw/')

bs = 100
sz = 256
test_df = pd.read_csv(PATH/"test_simplified.csv")

def create_func(item):
    with open(item) as f: item = f.read()
    img = list2drawing(json.loads(item)['data'], size=sz, lw=4, time_color=True)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    tensor = torch.from_numpy(img).float()
    return Image(tensor.permute((2,0,1)).div_(255))

item_list = ItemList.from_folder(PATH/"train_folders", create_func=create_func)
np.random.seed(42)
idxs = np.arange(item_list.items.shape[0])
np.random.shuffle(idxs)
val_idxs = idxs[:NUM_VAL]
item_lists = item_list.split_by_idx(val_idxs)
label_lists = item_lists.label_from_folder()

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
    
epoch_size = 1_000_000
print(f"Save model every {epoch_size} iters")

tfms = get_transforms(do_flip=True, flip_vert=False, 
                      max_rotate=10, max_zoom=0, max_lighting=None, max_warp=None)

train_dl = DataLoader(label_lists.train, num_workers=8,
    batch_sampler=BatchSampler(RandomSamplerWithEpochSize(label_lists.train, epoch_size), bs, True))
valid_dl = DataLoader(label_lists.valid, bs, False, num_workers=8)
test_dl = DataLoader(label_lists.test, bs, False, num_workers=8)
data_bunch = ImageDataBunch(train_dl, valid_dl, test_dl)

classes = data_bunch.classes
pd.to_pickle(classes, PATH/"classes.pkl")

from fastai.callbacks import SaveModelCallback, EarlyStoppingCallback
import sys
sys.path.append("./senet.pytorch/")
from se_resnet import se_resnet50
model = se_resnet50(340)
learn = Learner(data_bunch, model, metrics=[accuracy, map3],
                callback_fns=[partial(SaveModelCallback, every="epoch", name="senet-v2")])

ckpt = "final-senet-stage-1"
print(f"Loading ckpt : {ckpt}")
learn.load(ckpt)
learn.fit_one_cycle(72, max_lr=5e-4)

name = 'final-senet-stage-1'
learn.save(f'{name}')
