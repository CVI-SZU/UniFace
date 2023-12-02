''' Implementation for ResNet '''

import torch
import torch.nn as nn

from head import Normalized_Softmax_Loss, Normalized_BCE_Loss, Unified_Cross_Entropy_Loss

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    if stride==2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride, bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


class IRBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(IRBlock, self).__init__()
        self.downsample = downsample
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        return out + identity


class LResNetXEIR(nn.Module):
    def __init__(self, block, layers, zero_init_residual=True, num_classes=10):
        super(LResNetXEIR, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.prelu = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.4)

        self.fc5 = nn.Conv2d(512, 512, (7, 7), bias=False)
        self.bn5 = nn.BatchNorm1d(512, affine=True)
        self.bn5.weight.requires_grad = False

        # self.loss = Normalized_Softmax_Loss(512, num_classes)
        # self.loss = Normalized_BCE_Loss(512, num_classes)
        self.loss = Unified_Cross_Entropy_Loss(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IRBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        if stride != 1:
            downsample = nn.Sequential(
                conv3x3(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes)
            )
        else:
            downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def restrict_weights(self):
        for m in self.modules():
            if isinstance(m, nn.PReLU):
                m.weight.data.clamp_(min=0.01)

    def forward(self, x, t=None, partial_index=None):
        if not self.training:
            x = torch.cat((x, x.flip(3)), 0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = torch.flatten(x, 1)
        feat = self.bn5(x)

        if not self.training:
            a, b = feat.chunk(2)
            return a + b
        else:
            return self.loss(feat, t, partial_index)


def LResNet50EIR(**kwargs):
    model = LResNetXEIR(IRBlock, [3, 4, 14, 3], **kwargs)
    return model


def LResNet100EIR(**kwargs):
    model = LResNetXEIR(IRBlock, [3, 13, 30, 3], **kwargs)
    return model


def LResNet200EIR(**kwargs):
    model = LResNetXEIR(IRBlock, [3, 33, 60, 3], **kwargs)
    return model


if __name__ == '__main__':
    model = LResNet50EIR()
    print(model)
