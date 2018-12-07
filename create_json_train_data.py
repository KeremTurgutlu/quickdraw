import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from utils import *

PATH = Path('../data/quickdraw/')
dfs_combined = pd.read_csv(PATH/"train/dfs_combined.csv")
classes = dfs_combined['word'].unique()
NUM_SAMPLES = 70000
for c in classes:
    print(c)
    folder_name = c.replace(" ", "_")
    os.makedirs(PATH/f"train_folders/{folder_name}", exist_ok=True)
    drawings = dfs_combined[dfs_combined['word'] == c]["drawing"].values
    for i, drawing in enumerate(drawings[:NUM_SAMPLES]):
        with open(PATH/f"train_folders/{folder_name}/sample_{i}.json", mode="w+") as f:
            f.write(f'{{"data":{drawing}}}')
