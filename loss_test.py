# pylint: disable=W,C,R

import os
import sys
import numpy as np
import pandas as pd
from torchvision import transforms
from torch import cat
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import random
import torch.nn.functional as F


# Taken from:
# https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py
class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true, epsilon=1e-6):
        """Altered Sorensen–Dice coefficient with epsilon for smoothing."""
        y_true_flatten = np.asarray(y_true).astype(np.bool)
        y_pred_flatten = np.asarray(y_pred).astype(np.bool)

        if not np.sum(y_true_flatten) + np.sum(y_pred_flatten):
            return 1.0

        dc = (2. * np.sum(y_true_flatten * y_pred_flatten)) /\
            (np.sum(y_true_flatten) + np.sum(y_pred_flatten) + epsilon)
        return 1.0 - dc

        #assert y_pred.size() == y_true.size()
        #y_pred = y_pred[:, 0].contiguous().view(-1)
        #y_true = y_true[:, 0].contiguous().view(-1)
        #intersection = (y_pred * y_true).sum()
        #dsc = (2. * intersection + self.smooth) / (
        #    y_pred.sum() + y_true.sum() + self.smooth
        #)
        #return 1. - dsc

def dice_coef(y_pred, y_true, epsilon=1e-6):
    y_pred_flatten = y_pred.contiguous().view(-1)
    y_true_flatten = y_true.contiguous().view(-1)

    #if (not (y_true_flatten).sum() + (y_pred_flatten).sum()):
    #    return 0.0
    
    dc = (2. * (y_true_flatten * y_pred_flatten).sum()) /\
        ((y_true_flatten).sum() + (y_pred_flatten).sum() + epsilon)
    
    return 1.0 - dc
    

#def wbce(pred, true):
#    #pred = F.log_softmax(pred)
#    F.nll_loss(pred, true)


count_0 = 1
count_1 = 1596
count_2 = 1
count_3 = 1
count_4 = 1



lab_0 = np.array([[[0,0,0,0] for i in range(count_0)] for j in range(400)])
lab_1 = np.array([[[0,0,0,1] for i in range(count_1)] for j in range(400)])
lab_2 = np.array([[[0,0,1,0] for i in range(count_2)] for j in range(400)])
lab_3 = np.array([[[0,1,0,0] for i in range(count_3)] for j in range(400)])
lab_4 = np.array([[[1,0,0,0] for i in range(count_4)] for j in range(400)])

lab = torch.from_numpy(np.concatenate((lab_0,lab_1,lab_2,lab_3,lab_4),axis=1).transpose(2,0,1).reshape(1,-1)).double()
zeros = torch.from_numpy(np.zeros(shape=[400,1600,4]).transpose(2,0,1).reshape(1,-1)).double()

#print(zeros.shape)
#print(lab.shape)
dc = DiceLoss()
#loss =  F.binary_cross_entropy_with_logits(zeros,lab,pos_weight=[8])   
zeros[0][0] += 1
#print(dc(zeros,lab).item())
print(dice_coef(zeros, lab))

#print(bce(zeros, lab))




#print(lab.shape)
#
#count = [0,0,0,0,0]
#
#for row in lab:
#    for pixel in row:
#        if all(pixel == [0,0,0,0]):
#            count[0]+=1 
#        if all(pixel == [0,0,0,1]):
#            count[1]+=1
#        if all(pixel == [0,0,1,0]):
#            count[2]+=1 
#        if all(pixel == [0,1,0,0]):
#            count[3]+=1            
#        if all(pixel == [1,0,0,0]):
#            count[4]+=1 
#
#print(count[0]/400)
#print(count[1]/400)
#print(count[2]/400)
#print(count[3]/400)
#print(count[4]/400)
