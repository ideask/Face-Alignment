import torch
from torch import nn

class LinearLoss(nn.Module):
    def __init__(self):
        super(LinearLoss, self).__init__()
    def forward(self, landmark_gt, landmarks,face_cls_gt, face_cls, train_batchsize):
        l2_distant = torch.sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)
        cls_loss = nn.BCELoss()
        print(cls_loss(face_cls, face_cls_gt))
        return torch.mean(l2_distant) + cls_loss(face_cls, face_cls_gt)