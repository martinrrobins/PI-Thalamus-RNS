#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:05:34 2021

@author: vpeterson
"""

import torch
import torch.nn as nn


# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

class TverskyLoss(nn.Module):
    r"""Criterion that computes Tversky Coeficient loss.

    According to [1], we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \ G| + \beta |G \ P|}

    where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Notes:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.TverskyLoss(alpha=0.5, beta=0.5)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(self, alpha, beta, gamma, reduction='mean', eps = 1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6
        self.reduction = reduction
        self.gamma = 1

    def forward(self, predict, target):
        if not torch.is_tensor(predict):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(predict)))
        if not predict.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    predict.device, target.device))
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        # compute the actual dice score
        intersection = torch.sum(torch.mul(predict, target), dim=1) 
        fps = torch.sum(torch.mul(predict, (1. - target)), dim=1)
        fns = torch.sum(torch.mul((1. - predict),target), dim=1)

        numerator = intersection 
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = (numerator + self.eps) / (denominator + self.eps)
        
        # focal loss (since by default gamma is equal to 1, focal loss will be
        # always applied before the return
        loss  =  (1. - tversky_loss).pow(self.gamma)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
        return loss
    
class TverskyLossStepWise(nn.Module):
    r"""Criterion that computes Tversky Coeficient loss.

    According to [1], we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \ G| + \beta |G \ P|}

    where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Notes:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.TverskyLoss(alpha=0.5, beta=0.5)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(self, alpha, beta, gamma, reduction='mean', eps = 1e-6):
        super(TverskyLossStepWise, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6
        self.reduction = reduction
        self.gamma = 1

    def forward(self, predict, target):
        if not torch.is_tensor(predict):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(predict)))
        if not predict.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    predict.device, target.device))
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        # compute the actual dice score
        intersection = torch.sum(torch.mul(predict, target), dim=1) 
        fps = torch.sum(torch.mul(predict, (1. - target)), dim=1)
        fns = torch.sum(torch.mul((1. - predict),target), dim=1)

        numerator = intersection + self.eps
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_ = numerator / (denominator + self.eps)
        
        tversky_loss = (1. - tversky_)
        energy = torch.linalg.norm(predict, dim=1)
        
        label = target.max(dim=1)[0]
        one_hot = torch.nn.functional.one_hot(label.to(torch.int64), 2)
        tversky_loss_comb = torch.stack((energy, tversky_loss), 0)
        # tversky_loss_comb = torch.stack((tversky_loss, energy), 0)

        loss_stepwise = torch.mul(one_hot,tversky_loss_comb.T).sum(dim=1)
        
        # focal loss (since by default gamma is equal to 1, focal loss will be
        # always applied before the return
        loss  =  loss_stepwise.pow(self.gamma)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
        return loss
