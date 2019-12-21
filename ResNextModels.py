import math
import torch
import torch.nn as nn


__all__ = ['resnext18', 'resnext34', 'resnext50', 'resnext101', 'resnext152']

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, outplanes, stride=1, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes*2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes*2, outplanes * BasicBlock.expansion, kernel_size=3, padding=1, groups=num_group, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes * BasicBlock.expansion)

        self.shortcut = nn.Sequential()

        if stride != 1 or inplanes != BasicBlock.expansion * outplanes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, outplanes*BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes*BasicBlock.expansion)
            )
    
    def forward(self,x):
        s = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        return self.relu(s + x)

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, outplanes, stride=1, num_group=32):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes*2, outplanes*2, stride=stride, kernel_size=3, padding=1, groups=num_group, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes*2)
        self.conv3 = nn.Conv2d(outplanes*2, outplanes* BottleNeck.expansion , kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes* BottleNeck.expansion )

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != outplanes*4 :
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, outplanes * BottleNeck.expansion , stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplanes * BottleNeck.expansion )
            )
    def forward(self,x):
        s = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return self.relu(s + x)

class ResNext(nn.Module):
    def __init__(self, block, num_block, num_classes=1000):
        super(ResNext, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=7, stride=2, padding=3,bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
            )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def forward(self, x):
        output = self.conv1(x)
        output = self.maxpool(output)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output 

    def _make_layer(self, block, outplanes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, outplanes, stride))
            self.inplanes = outplanes * block.expansion
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


def resnext18():
    return ResNext(BasicBlock, [2, 2, 2, 2])

def resnext34():
    return ResNext(BasicBlock, [3, 4, 6, 3])

def resnext50():
    return ResNext(BottleNeck, [3, 4, 6, 3])

def resnext101():
    return ResNext(BottleNeck, [3, 4, 23, 3])

def resnext152():
    return ResNext(BottleNeck, [3, 8, 36, 3])        
        