import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from torchvision import transforms
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import torch
import os  
from pathlib import  Path
import numpy as np
import torchvision.transforms as T
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def iou(pred, target):
    ious = []
    for cls in range(10):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious

def miou_image_seg(output, Y):
    """ miou """
    total_ious = []
    output = output.data.cpu().numpy()

    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, 10).argmax(axis=1).reshape(N, h, w)
    target = Y.reshape(N, h, w)
    for p, t in zip(pred, target):
        total_ious.append(iou(p, t))

    total_ious = np.array(total_ious).T
    ious = np.nanmean(total_ious, axis=1)
    return ious

def load_model(filename, classes, channels, mode):
    if mode == 'b7':
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights=None,
            in_channels=channels,
            classes=classes)
    else:
        model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b8",
            encoder_weights=None,
            in_channels=channels,
            classes=classes,
            decoder_attention_type='scse')

    weights = torch.load(filename,map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    
    return model
print("开始读入模型")
modelfile1 = '../user_data/model_data/stage-0_unet++_4_aug_fc_yaogan_best_model.pth'
modelfile2 = '../user_data/model_data/stage-0_unet++_4_right_fc_yaogan_best_model.pth'
modelfile3 = '../user_data/model_data/stage-0_unet++_aug_fc_yaogan_best_model.pth'
modelfile4 = '../user_data/model_data/stage-0_unet++_4_fc_new1b8_aug_attention_yaogan_best_model.pth'

learn1 = load_model(modelfile1, 10, 4, 'b7')
learn2 = load_model(modelfile2, 10, 4, 'b7')
learn3 = load_model(modelfile3, 10, 3, 'b7')
learn4 = load_model(modelfile4, 10, 4, 'b8')
print("读入模型完成")

# A榜miou
miou_or1 = [0.5613,0.8770,0.0092,0.2858,0.3686,0.4738,0.4842,0.0423,0.7901,0.0020]
miou_or2 = [0.5504,0.8751,0.0062,0.2833,0.3537,0.4567,0.4760,0.0390,0.7815,0.0048]
miou_or3 = [0.5397,0.8691,0.0075,0.2730,0.3985,0.4751,0.5038,0.0413,0.7565,0.0059]
miou_or4 = [0.5549,0.8763,0.0026,0.2878,0.3751,0.4760,0.4727,0.0336,0.7933,0.0008]

# 计算测试集的miou
res1 = []
res2 = []
res3 = []
res4 = []
print("开始计算测试集miou并和A榜miou融合")
for path in Path("../user_data/tmp_data/test/images").rglob("*.tif"):
    img = T.ToTensor()(Image.open(path)).unsqueeze(0)
    img2 = T.ToTensor()(Image.open(path).convert('RGB')).unsqueeze(0)
    label = np.array(Image.open("../user_data/tmp_data/test/labels/"+(path.stem+".png")))
    out1 = learn1.predict(img)
    out2 = learn2.predict(img)
    out3 = learn3.predict(img2)
    out4 = learn4.predict(img)
    miou1 = miou_image_seg(out1,label)
    miou2 = miou_image_seg(out2,label)
    miou3 = miou_image_seg(out3,label)
    miou4 = miou_image_seg(out4,label)
    res1.append(miou1)
    res2.append(miou2)
    res3.append(miou3)
    res4.append(miou4)

for i in res1:
    i[np.isnan(i)] = 0
for i in res2:
    i[np.isnan(i)] = 0
for i in res3:
    i[np.isnan(i)] = 0
for i in res4:
    i[np.isnan(i)] = 0

new_miou1 = sum(res1)/3000
new_miou2 = sum(res2)/3000
new_miou3 = sum(res3)/3000
new_miou4 = sum(res4)/3000

last_miou1 = []
last_miou2 = []
last_miou3 = []
last_miou4 = []

# 融合miou
for i in range(10):
    last_miou1.append(new_miou1[i]*0.4+miou_or1[i]*0.6)
for i in range(10):
    last_miou2.append(new_miou2[i]*0.4+miou_or2[i]*0.6)
for i in range(10):
    last_miou3.append(new_miou3[i]*0.4+miou_or3[i]*0.6)
for i in range(10):
    last_miou4.append(new_miou4[i]*0.4+miou_or4[i]*0.6)

# 进行加权平均
n_miou1 = []
n_miou2 = []
n_miou3 = []
n_miou4 = []
for i in range(10):
    n_miou1.append(last_miou1[i]/(last_miou1[i]+last_miou2[i]+last_miou3[i]+last_miou4[i]))
    n_miou2.append(last_miou2[i]/(last_miou1[i]+last_miou2[i]+last_miou3[i]+last_miou4[i]))
    n_miou3.append(last_miou3[i]/(last_miou1[i]+last_miou2[i]+last_miou3[i]+last_miou4[i]))
    n_miou4.append(last_miou4[i]/(last_miou1[i]+last_miou2[i]+last_miou3[i]+last_miou4[i]))

print("miou融合完成")

# 预测结果
pre_path = Path("../prediction_result")
if not pre_path.exists():
    pre_path.mkdir(parents=True, exist_ok=True)

print("开始预测结果")
for path in Path("../tcdata/suichang_round1_test_partB_210120").rglob("*.tif"):
    img = T.ToTensor()(Image.open(path)).unsqueeze(0)
    img2 = T.ToTensor()(Image.open(path).convert('RGB')).unsqueeze(0)
    arr = np.array(Image.open(path))
    out1 = learn1.predict(img)
    out2 = learn2.predict(img)
    out3 = learn3.predict(img2)
    out4 = learn4.predict(img)
    out = torch.zeros_like(out1)
    for i in range(10):
        out[:,i,:,:] = out1[:,i,:,:]*n_miou1[i]+out2[:,i,:,:]*n_miou2[i]+out3[:,i,:,:]*n_miou3[i]+out4[:,i,:,:]*n_miou4[i]

    softmax = out.squeeze(0).detach().numpy()
    res = np.argmax(softmax, axis=0).reshape((arr.shape[0], arr.shape[1]))+1
    result = Image.fromarray(np.uint8(res))
    result.save(pre_path/(path.stem+'.png'), quality=100)
print("预测完成")
