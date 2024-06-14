import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch import autograd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def getMask(h,w,c,box):
    '''
    用于获取单个 mask
    feature: c*h*w
    '''

    channel = c
    heigh = h
    width = w
    mask_zeros = torch.zeros(channel, heigh, width)
    mask_ones = torch.ones(channel, heigh, width)
    for i in range(len(box)):
        xmin = int(box[i][1])
        ymin = int(box[i][0])
        xmax = int(box[i][3])
        ymax = int(box[i][2])

        #
        if xmin > xmax:
            temp = xmin
            xmin = xmax
            xmax = temp

            pass
        if ymin > ymax:
            temp = ymin
            ymin = ymax
            ymax = temp
            pass

        width = int(xmax - xmin)
        heigh = int(ymax - ymin)
        if width <= 0:
            width = 1
            pass

        if heigh <= 0:
            heigh = 1
            pass

        mask_zeros[:, ymin:ymin + heigh + 1, xmin:xmin + width + 1] = 1
        mask_ones[:, ymin:ymin + heigh + 1, xmin:xmin + width + 1] = 0
        pass
    # print(mask_zeros)
    # print(mask_ones)
    # print(mask_ones*mask_zeros)
    mask_zeros = mask_zeros.to(device)
    mask_ones = mask_ones.to(device)
    return mask_ones,mask_zeros
    pass


def getOneMask(h,w,c,box):
    '''
    获取单个图片的大目标mask
    '''
    channel = c
    heigh = h
    width = w
    mask_zeros = torch.zeros(channel, heigh, width)
    mask_ones = torch.ones(channel, heigh, width)
    flag = False # 代表是否有大尺度目标，没有则为False
    for i in range(len(box)):
        xmin = int(box[i][1])
        ymin = int(box[i][0])
        xmax = int(box[i][3])
        ymax = int(box[i][2])

        #
        if xmin > xmax:
            temp = xmin
            xmin = xmax
            xmax = temp

            pass
        if ymin > ymax:
            temp = ymin
            ymin = ymax
            ymax = temp
            pass

        width = int(xmax - xmin)
        heigh = int(ymax - ymin)
        if width <= 0:
            width = 1
            pass

        if heigh <= 0:
            heigh = 1
            pass

        scale = width * heigh
        if scale <= 10000:
            continue
        flag = True
        mask_zeros[:, ymin:ymin + heigh + 1, xmin:xmin + width + 1] = 1

        pass
    mask_zeros = mask_zeros.to(device)
    mask_ones = mask_ones.to(device)
    if flag == False:
        return mask_ones
    else:
        return mask_zeros
    pass

def getBigObjectMaskedPart(real_feature,bbox):
    '''
    获取大目标的mask
    '''

    batch_size = real_feature.shape[0]
    c = real_feature.shape[1]
    h = real_feature.shape[2]
    w = real_feature.shape[3]
    for i in range(batch_size):
        mask = getOneMask(h,w,c,bbox[i])
        real_feature[i] = real_feature[i] * mask
        pass
    return real_feature
    pass


def getMaskedPart(real_feature,fake_feature,bbox):
    '''
    用于获取被masked的部分
    '''
    batch_size = real_feature.shape[0]
    c = real_feature.shape[1]
    h = real_feature.shape[2]
    w = real_feature.shape[3]

    for i in range(batch_size):
        mask_ones, mask_zeros = getMask(h,w,c,bbox[i])
        real_feature[i] = real_feature[i] * mask_zeros
        fake_feature[i] = fake_feature[i] * mask_zeros
        pass
    return real_feature,fake_feature
    pass

def L2Loss(real,fake,counts):
    '''
    L2损失
    '''

    diff = real - fake
    diff2 = diff*diff
    diff3 = torch.sum(diff2,dim=0)
    diff4 = torch.sum(diff3,dim=0)
    diff5 = torch.sum(diff4,dim=0)
    diff6 = diff5 / counts

    # real_numpy = real.cpu().numpy()
    # fake_numpy = fake.cpu().numpy()
    # loss = np.sum(np.square(real_numpy-fake_numpy))/counts
    # loss = torch.from_numpy(loss).to(device)
    return diff6
    # return torch.from_numpy(np.sum(np.square(real-fake))/counts)
    pass
def L2LOss(real,fake):
    '''

    real:真实样本；
    fake:生成器假样本：
    '''

    pass


def reconstructLoss(b_realFeature,b_fakeFeature,b_box,h,w,c):
    '''
    重建损失
    '''
    batch_size = b_realFeature.shape[0]
    # b_realFeature = b_realFeature.numpy()
    # b_fakeFeature = b_fakeFeature.numpy()
    heigh = h
    width = w
    channel = c
    sum_L2Loss = 0
    for i in range(batch_size):
        mask_ones, mask_zeros = getMask(heigh,width,channel,b_box[i])
        masked_part_real = mask_zeros*b_realFeature[i]
        masked_part_fake = mask_zeros*b_fakeFeature[i]
        # b_realFeature[i] = mask_zeros * b_realFeature[i]
        # b_fakeFeature[i] = mask_zeros * b_fakeFeature[i]
        sum_L2Loss = sum_L2Loss + L2Loss(masked_part_real,masked_part_fake,len(b_box[i]))

        pass

    return sum_L2Loss/batch_size
    pass

def CoordinateScaling(bboxs,scale=1):

    for i in range(len(bboxs)):
        for j in range(len(bboxs[i])):
            bboxs[i][j][0] = bboxs[i][j][0] * scale
            bboxs[i][j][1] = bboxs[i][j][1] * scale
            bboxs[i][j][2] = bboxs[i][j][2] * scale
            bboxs[i][j][3] = bboxs[i][j][3] * scale
            pass

        pass
    return bboxs
    pass

def calculate_gradient_penalty(batch_size,Descrimator,real_images, fake_images,cuda=True,cuda_index=0,lambda_term=10):
        real_images = real_images.to(device)
        fake_images = fake_images.to(device)
        eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if cuda:
            eta = eta.cuda(cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if cuda:
            interpolated = interpolated.cuda(cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = Descrimator(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(cuda_index) if cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
        return grad_penalty


if __name__ == '__main__':
    sample = torch.zeros(3,6,6)
    box = [1,1,2,2]
    getMask(sample,box)
    pass
