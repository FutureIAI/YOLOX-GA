import torch
import torch.nn as nn

# from nets.darknet import CSPDarknet,DWConv,BaseConv
# from nets.yolo_training import weights_init,ModelEMA
import numpy as np
import torch.backends.cudnn as cudnn

from superFeatureNet.nets.darknet import CSPDarknet,DWConv,BaseConv
from superFeatureNet.nets.yolo_training import weights_init,ModelEMA
model_path = r''
local_rank = 0
device = torch.device("cuda", local_rank)
Cuda = True
distributed = False



class YOLOPAFPN(nn.Module):
    def __init__(self,depth = 1.0, width = 1.0, in_features = ("dark2","dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super(YOLOPAFPN, self).__init__()
        self.backbone = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features = in_features
        pass
    def forward(self,x):
        out_features = self.backbone.forward(x)
        [dark2,feat1, feat2, feat3] = [out_features[f] for f in self.in_features]
        return [dark2,feat1, feat2, feat3]
        pass
    pass


class YoloBodys(nn.Module):
    '''
    超分辨特征生成网络
    '''
    def __init__(self,num_classes, phi):
        super(YoloBodys, self).__init__()
        self.backbone = YOLOPAFPN(0.33, 0.25, depthwise=False)
        self.avgpool = nn.AvgPool2d((2,2))

        pass

    def forward(self,x):
        fpn_outs    = self.backbone.forward(x)
        return fpn_outs
        pass
    pass

def getSuperFeatureNet():
    '''
    获取超分辨率特征生成网络
    '''

    model = YoloBodys(20,'nano')
    # model = YoloBody(20,'s')
    weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
            pass
        pass
    model_train = model.train()

    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
            pass
        pass
    # ----------------------------#
    #   权值平滑
    # ----------------------------#
    ema = ModelEMA(model_train)
    return model_train
    pass


