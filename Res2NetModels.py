import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

__all__ = ['res2net50','res2net101','res2net152']

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class Bottle2neck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, outplanes, stride=1, downsample=None, scales = 4, se=None):
        super(Bottle2neck, self).__init__()
        self.conv1 = conv1x1(inplanes, outplanes, stride)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.ModuleList([conv3x3(outplanes//scales,outplanes//scales) for _ in range(scales - 1)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(outplanes//scales) for _ in range(scales - 1)])
        self.conv3 = conv1x1(outplanes,outplanes*self.expansion)
        self.bn3 = nn.BatchNorm2d(outplanes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.scales = scales
        self.downsample = downsample
        self.se = se
        self.outplanes = outplanes

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        xs = torch.split(out,self.outplanes//4,dim=1)
        ys = []
        for i in range(self.scales):
            if i == 0:
                ys.append(xs[i])
            elif i==1:
                ys.append(self.relu(self.bn2[i-1](self.conv2[i-1](xs[i]))))
            else:
                ys.append(self.relu(self.bn2[i-1](self.conv2[i-1](xs[i]+ys[-1]))))
        out = torch.cat(ys,dim=1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out = out+ x
        out = self.relu(out)
        return out


class Res2Net(nn.Module):
    def __init__(self, num_block, num_class=200, width=16, scales=4, se=None):
        super(Res2Net, self).__init__()
        outplanes = [int(width*scales*2**i) for i in range(4)]
        self.conv1 = nn.Conv2d(3, outplanes[0],kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes[0])     
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = outplanes[0]
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(Bottle2neck, outplanes[0], num_block[0], scales=scales, se=se)
        self.layer2 = self._make_layer(Bottle2neck, outplanes[1], num_block[1], stride=2, scales=scales, se=se)
        self.layer3 = self._make_layer(Bottle2neck, outplanes[2], num_block[2], stride=2, scales=scales, se=se)
        self.layer4 = self._make_layer(Bottle2neck, outplanes[3], num_block[3], stride=2, scales=scales, se=se)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(outplanes[3] * Bottle2neck.expansion, num_class)
        self._initialize_weights()


    def _make_layer(self,block,outplanes,num_block,stride=1,scales=4,se=None):
        downsample = None
        if stride != 1 or self.inplanes != outplanes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, outplanes * block.expansion, stride),
                nn.BatchNorm2d(outplanes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, outplanes, stride, downsample, scales=scales, se=se))
        self.inplanes = outplanes * block.expansion
        for _ in range(1,num_block):
            layers.append(block(self.inplanes, outplanes, scales=scales, se=se))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
                
def res2net50(**kwargs):
    """Constructs a Res2Net-50 model.
    """
    model = Res2Net([3, 4, 6, 3], **kwargs)
    return model


def res2net101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = Res2Net([3, 4, 23, 3], **kwargs)
    return model


def res2net152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = Res2Net([3, 8, 36, 3], **kwargs)
    return model
        