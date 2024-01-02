import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F
import cv2

def SoftIoULoss(pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss

def FocalLoss(inputs, targets):
    alpha = 0.75
    gamma = 2
    num_classes = 2
    gamma = gamma
    size_average = True
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    targets = targets.type(torch.long)
    at = targets*alpha+(1-targets)*(1-alpha)
    pt = torch.exp(-BCE_loss)
    F_loss = (1 - pt) ** gamma * BCE_loss
    F_loss = at * F_loss
    return F_loss.sum()



