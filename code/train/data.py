from pathlib import Path
import os
import numpy as np
from fastai.torch_core import parallel
from fastai.vision.all import *
from fastai.data.all import *
from PIL import Image
import shutil

data_path = Path("../tcdata/suichang_round1_train_210120")
image_path = Path("../user_data/tmp_data/or_images")
label_path = Path("../user_data/tmp_data/or_labels")
if not image_path.exists():
    image_path.mkdir(parents=True, exist_ok=True)
if not label_path.exists():
    label_path.mkdir(parents=True, exist_ok=True)

def mo(i):
    shutil.copy(i, image_path/i.name)
    label = np.array(Image.open(data_path/(i.stem+'.png')))-1
    new_label = Image.fromarray(label)
    new_label.save(label_path/(i.stem+'.png'), quality=100)
print("开始将数据集分开")
parallel(mo, [i for i in data_path.rglob("*.tif")])
print("分开完成")