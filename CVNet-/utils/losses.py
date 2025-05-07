import torch
import torch.nn as nn
import torch.nn.functional as F

# 可以在一定程度上解决正负例不平衡
class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        # batch equal to True means views all batch images as an entity and calculate loss
        # batch equal to False means calculate loss of every single image in batch and get their mean
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.00001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)

        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred.to(dtype=torch.float32), y_true)


class dice_focal_loss(nn.Module):

    def __init__(self):
        super(dice_focal_loss, self).__init__()
        # 这个就是交叉熵和sigomd结合
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.binnary_dice = dice_loss()

    def __call__(self, scores, labels):
        diceloss = self.binnary_dice(torch.sigmoid(scores.clone()), labels) #这里会不会出问题？？
        foclaloss = self.focal_loss(scores.clone(), labels)
        return [diceloss, foclaloss]


def FCCDN_loss_without_seg(scores, labels):
    # scores = change_pred
    # labels = binary_cd_labels
    scores = [score.squeeze(1) if len(score.shape) > 3 else score for score in scores]
    # labels = [label.squeeze(1) if len(label.shape) > 3 else label for label in labels]
    # if len(scores.shape) > 3:
    #     scores = scores.squeeze(1)
    # if len(labels.shape) > 3:
    #     labels = labels.squeeze(1)
    """ for binary change detection task"""
    criterion_change = dice_focal_loss()

    # change loss
    loss_change = criterion_change(scores[0], labels[0])
    # loss_seg1 = criterion_change(scores[1], labels[1])
    # loss_seg2 = criterion_change(scores[2], labels[1])


    return loss_change







class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        target = target.float()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-1] != target.shape[-1]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

        loss = F.cross_entropy(input=input, target=target, weight=self.weight,
                                ignore_index=self.ignore_index, reduction=self.reduction)
        return loss



def cross_entropy_loss(scores, labels):
    """
    For binary change detection task, using only cross entropy loss.
    """
    criterion_change = CrossEntropyLoss()

    # Ensure scores and labels are properly shaped
    scores = [score.squeeze(1) if len(score.shape) > 3 else score for score in scores]
    labels = [label.squeeze(1) if len(label.shape) > 3 else label for label in labels]

    # Calculate change loss
    loss_change = criterion_change(scores[0], labels[0])
    # If you have additional segmentation tasks, you can calculate additional losses here
    # loss_seg1 = criterion_change(scores[1], labels[1])
    # loss_seg2 = criterion_change(scores[2], labels[1])

    return loss_change



# def cross_entropy_loss(scores, labels):
#     """
#     For binary change detection task, using only cross entropy loss.
#     """
#     criterion_change = CrossEntropyLoss()
#
#     # Ensure scores and labels are properly shaped
#     # scores = [score.squeeze(1) if len(score.shape) > 3 else score for score in scores]
#     # labels = [label.squeeze(1) if len(label.shape) > 3 else label for label in labels]
#
#     # Calculate change loss
#     loss_change = criterion_change(scores, labels)
#     # If you have additional segmentation tasks, you can calculate additional losses here
#     # loss_seg1 = criterion_change(scores[1], labels[1])
#     # loss_seg2 = criterion_change(scores[2], labels[1])
#
#     return loss_change







class bce_loss(nn.Module):

    def __init__(self):
        super(bce_loss, self).__init__()
        self.focal_loss = nn.BCEWithLogitsLoss()

    def __call__(self, scores, labels):
        bce_loss1 = self.focal_loss(scores.clone(), labels)
        bce_loss2 = self.focal_loss(scores.clone(), labels)
        return [bce_loss1, bce_loss2]

def BCE_loss_without_seg(scores, labels):

    scores = [score.squeeze(1) if len(score.shape) > 3 else score for score in scores]
    """ for binary change detection task"""
    criterion_change = bce_loss()

    loss_change = criterion_change(scores[0], labels[0])
    # loss_seg1 = criterion_change(scores[1], labels[1])
    # loss_seg2 = criterion_change(scores[2], labels[1])


    return loss_change