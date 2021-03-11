from pathlib import Path
import os
import numpy as np
from fastai.torch_core import parallel
from fastai.vision.all import *
from fastai.data.all import *
from PIL import Image
import shutil
from fastprogress.fastprogress import progress_bar
import random

dst_root = Path("../user_data/tmp_data")

img_folder = dst_root/"need_aug_img"
label_folder = dst_root/"need_aug_lab"

save_images = dst_root/"aug_images"
save_labels = dst_root/"aug_labels"

test_img = dst_root/"test"/"images"
test_lab = dst_root/"test"/"labels"

if not test_img.exists():
    test_img.mkdir(parents=True, exist_ok=True)
if not test_lab.exists():
    test_lab.mkdir(parents=True, exist_ok=True)

if not img_folder.exists():
    img_folder.mkdir(parents=True, exist_ok=True)
if not label_folder.exists():
    label_folder.mkdir(parents=True, exist_ok=True)

if not save_images.exists():
    save_images.mkdir(parents=True, exist_ok=True)
if not save_labels.exists():
    save_labels.mkdir(parents=True, exist_ok=True)

def mo(i):
    img = Image.open(i)
    arr = np.array(img)

    if 4 in arr:
        shutil.copy(dst_root/'or_images'/(i.stem+'.tif'), img_folder/(i.stem+'.tif'))
        shutil.copy(i, label_folder/i.name)
    if 5 in arr:
        shutil.copy(dst_root/'or_images'/(i.stem+'.tif'), img_folder/(i.stem+'.tif'))
        shutil.copy(i, label_folder/i.name)
    if 7 in arr:
        shutil.copy(dst_root/'or_images'/(i.stem+'.tif'), img_folder/(i.stem+'.tif'))
        shutil.copy(i, label_folder/i.name)
    if 2 in arr:
        shutil.copy(dst_root/'or_images'/(i.stem+'.tif'), img_folder/(i.stem+'.tif'))
        shutil.copy(i, label_folder/i.name)
    if 9 in arr:
        shutil.copy(dst_root/'or_images'/(i.stem+'.tif'), img_folder/(i.stem+'.tif'))
        shutil.copy(i, label_folder/i.name)

def cp(i):
    if i.suffix == '.tif':
        shutil.copy(i, save_images/i.name)
    else:
        shutil.copy(i, save_labels/i.name)

def aug(i):
    img_w = 200
    img_h = 200  
    num = 2
    count = 0
    img = Image.open(i)
    label = Image.open(label_folder/(i.stem+'.png'))
    while count < num:
        width1 = random.randint(0, img.size[0] - img_w )
        height1 = random.randint(0, img.size[1] - img_h)
        width2 = width1 + img_w
        height2 = height1 + img_h  
        
        img_roi=img.crop((width1, height1, width2, height2))
        label_roi=label.crop((width1, height1, width2, height2))

        img_roi.save(save_images/('p_'+str(count)+'_'+i.name), quality=100)
        label_roi.save(save_labels/('p_'+str(count)+'_'+(i.stem+'.png')),quality=100)
        count += 1
print("开始提出3000张测试集")
a = [i for i in (dst_root/'or_images').rglob("*.tif")]
b = random.sample(a, 3000)
for i in b:
    shutil.move(i, test_img/i.name) 
    shutil.move(dst_root/"or_labels"/(i.stem+'.png'), test_lab/(i.stem+'.png'))
print("提出完成")
print("开始寻找需要crop的数据")
parallel(mo, [i for i in (dst_root/'or_labels').rglob("*.png")])
print("寻找完成")
print("开始crop数据")
parallel(cp, [i for i in (dst_root/'or_images').rglob("*.tif")])
parallel(cp, [i for i in (dst_root/'or_labels').rglob("*.png")])
parallel(aug, [i for i in img_folder.rglob("*.tif")])
print("crop完成")









