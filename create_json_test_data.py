import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from utils import *

PATH = Path('../data/quickdraw/')
test_df = pd.read_csv(PATH/"test_simplified.csv")

os.makedirs(PATH/f"test/", exist_ok=True)
drawings = test_df["drawing"].values
key_ids = test_df["key_id"].values

for i, (drawing, key_id) in enumerate(zip(drawings,key_ids)):
    with open(PATH/f"test/{key_id}", mode="w+") as f:
        f.write(f'{{"data":{drawing}}}')