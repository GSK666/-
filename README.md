# 阿里天池2021全国数字生态创新大赛智能算法赛：生态资产智能分析
## 遥感图像分割比赛
网络主模型采用Unet++    
backbone为efficientnet-b7   
采用ImageNet预训练模型   
## 数据说明
分别采用3通道和4通道训练数据进行训练   
3通道图像直接将4通道原始数据删除最后一维进行保存   
由于fastai默认将训练数据变成RGB格式，但是原始训练数据为4通道数据，需要修改fastai源码参数，需要将fastai.vision.core下的PILBase中的参数修改为PILBase.\_open_args = {'mode':'RGBA'}   
## 训练策略
利用fastai编写   
采用fastai自带默认数据增强方式，训练尺寸为256x256，优化器为Adam   
利用fit_flat_cos训练方式训练100轮   
## 测试结果
由于训练标签从1开始，需要将原始标签值减1，最后预测结果加1    
3通道验证集结果miou：52%，测试集结果miou：37.5%    
3通道对原始数据中grass和bareland类进行随机crop扩充，最后验证集miou：56%，测试集miou：38.7%    
4通道验证集结果miou：53%，测试集结果miou：38.2%    
4通道对原始图像中的5，6，8类进行随机crop扩充，验证集miou：，测试集miou
