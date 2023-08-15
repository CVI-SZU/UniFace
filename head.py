from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import random
import numpy as np


class Normalized_Softmax_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64, r=1.0):
        super(Normalized_Softmax_Loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.r = r
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.register_buffer('weight_mom', torch.zeros_like(self.weight))

    def forward(self, input, label, partial_index):
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight[partial_index], eps=1e-5))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = torch.index_select(one_hot, 1, partial_index)

        d_theta = one_hot.to(cos_theta) * self.m
        logits = self.s * (cos_theta - d_theta)
        return F.cross_entropy(logits, label)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) \
               + ', r=' + str(self.r) + ')'


class Normalized_BCE_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64, l=1.0, r=1.0):
        super(Normalized_BCE_Loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.l = l
        self.r = r
        self.bias = Parameter(torch.FloatTensor(1, out_features))
        nn.init.constant_(self.bias, math.log(out_features*r*10))
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.register_buffer('weight_mom', torch.zeros_like(self.weight))

    def forward(self, input, label, partial_index):
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight[partial_index], eps=1e-5))

        cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
        cos_m_theta_n = self.s * cos_theta - self.bias
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s))) * self.l

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = torch.index_select(one_hot, 1, partial_index)

        loss = one_hot * p_loss + (~one_hot) * n_loss

        return loss.sum(dim=1).mean()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) \
               + ', l=' + str(self.l) \
               + ', r=' + str(self.r) + ')'


class Unified_Cross_Entropy_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64, l=1.0, r=1.0):
        super(Unified_Cross_Entropy_Loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.l = l
        self.r = r
        self.bias = Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.bias, math.log(out_features*r*10))
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.register_buffer('weight_mom', torch.zeros_like(self.weight))

    def forward(self, input, label, partial_index):
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight[partial_index], eps=1e-5))

        cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
        cos_m_theta_n = self.s * cos_theta - self.bias
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s))) * self.l

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = torch.index_select(one_hot, 1, partial_index)

        loss = one_hot * p_loss + (~one_hot) * n_loss

        return loss.sum(dim=1).mean()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) \
               + ', l=' + str(self.l) \
               + ', r=' + str(self.r) + ')'