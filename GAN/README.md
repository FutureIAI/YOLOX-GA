## 注意生成式对抗网络的训练
---

## 所需环境
pytorch==1.2.0

 
## 训练步骤（以VOC数据集为例）
### a、训练VOC07+12数据集
1. 数据集的准备   
**本文使用VOC格式进行训练，训练前需要下载好VOC07+12的数据集，解压后放在根目录**  

2. 数据集的处理   
修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。   

3. 本文是使用yolox作为基准模型进行训练，因此使用yolox模型在VOC数据集上首先进行训练，将训练好的权重路径分别加入train.py文件和superFeature文件中的model_path参数中

4. 开始网络训练   
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。   



