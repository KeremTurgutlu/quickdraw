import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from fastai import *
from fastai.vision import *

import json
from utils import *
from PIL import Image

# read data
PATH = Path('../data/quickdraw/')
test_df = pd.read_csv(PATH/"test_simplified.csv")
test_df = test_df[['key_id', 'drawing']]

# utils
def get_raw_stroke_parts(raw_strokes):
    if isinstance(raw_strokes, str): raw_strokes = eval(raw_strokes)
    n = len(raw_strokes)
    if n == 1:
        return [[raw_strokes[0]], None, None]
    elif n == 2:
        return [[raw_strokes[0]], [raw_strokes[1]], None]
    else:
        div, _ = divmod(n, 3)
        return [raw_strokes[:div], raw_strokes[div:div*2], raw_strokes[div*2:]]
    
def list2drawing(raw_strokes, size=256, lw=6, time_color=False):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        return img
 
def raw_stroke_parts2drawing(raw_stroke_parts, sum_as_final=False, color=False):
    img = []
    for raw_stroke in raw_stroke_parts:
        if raw_stroke is not None:
            img.append(list2drawing(raw_stroke, 256, lw=2, time_color=color))
        else:
            img.append(np.zeros((256, 256)))
    if sum_as_final:
        img[2] = img[0] + img[1] + img[2]
    return np.array(img)

def list2drawing_3channels(raw_strokes, sum_as_final, color):
    return raw_stroke_parts2drawing(get_raw_stroke_parts(raw_strokes), sum_as_final, color)

def save_img(save_path, i, drawing):
    raw_stroke_parts = get_raw_stroke_parts(drawing)
    img = raw_stroke_parts2drawing(raw_stroke_parts, sum_as_final=True, color=True)
    pil_img = Image.fromarray(np.rollaxis(img, 0, 3).astype(dtype=np.uint8))
    pil_img.save(save_path/f"{i}.png", format='png')

    
test_drawings = test_df['drawing'].values
test_keys = test_df['key_id'].values.astype(str)
folder_path = PATH/f"test/"
# args
i_drawings = [[folder_path] + list(p) for p in list(zip(test_keys, test_drawings))]
print("Saving...")
with ThreadPoolExecutor(max_workers=4) as e:
    e.map(lambda p: save_img(*p), i_drawings)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        