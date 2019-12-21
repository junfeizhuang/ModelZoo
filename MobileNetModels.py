import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MobileNetV1','MobileNetV2','MobileNetV3Large','MobileNetV3Small']

class MobileNetV1Module(nn.Module):
    def __init__(self,inplanes, outplanes, stride):
        super(MobileNetV1Module, self).__init__()
        self.block =  nn.Sequential(
                            nn.Conv2d(inplanes,inplanes,3,stride,1,groups=inplanes, bias=False),
                            nn.BatchNorm2d(inplanes),
                            nn.ReLU6(inplace=True),
                            nn.Conv2d(inplanes,outplanes,1,1, bias=False),
                            nn.BatchNorm2d(outplanes),
                            nn.ReLU6(inplace=True),
                        )

    def forward(self, x):
        return self.block(x)

class MobileNetV2Module(nn.Module):
    def __init__(self, inplanes, outplanes, expand, stride):
        super(MobileNetV2Module, self).__init__()
        if stride == 1 and inplanes == outplanes: 
            self.shortcut = True 
        else:
            self.shortcut = False
        self.middleplanes = inplanes * expand
        self.block = nn.Sequential(
                            nn.Conv2d(inplanes, self.middleplanes, 1,1,bias=False),
                            nn.BatchNorm2d(self.middleplanes),
                            nn.ReLU6(inplace=True),
                            nn.Conv2d(self.middleplanes, self.middleplanes,3,stride,1,groups=self.middleplanes, bias=False),
                            nn.BatchNorm2d(self.middleplanes),
                            nn.ReLU6(inplace=True),
                            nn.Conv2d(self.middleplanes, outplanes, 1,1, bias=False),
                            nn.BatchNorm2d(outplanes),
            )
        

    def forward(self, x):
        out = self.block(x)
        if not self.shortcut:
            return out
        else:
            return out + x

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MobileNetV3Module(nn.Module):
    def __init__(self, inplanes, outplanes, expand, kernel, stride, nolieaner, se):
        super(MobileNetV3Module, self).__init__()
        self.middleplanes = round(inplanes * expand)
        self.se = se
        if se: self.se_module = SeModule(outplanes)
        self.feature = nn.Sequential(
                                nn.Conv2d(inplanes, self.middleplanes,1,1,0,bias=False),
                                nn.BatchNorm2d(self.middleplanes),
                                nolieaner,
                                nn.Conv2d(self.middleplanes,self.middleplanes,kernel,stride,kernel // 2, groups=self.middleplanes,bias=False),
                                nn.BatchNorm2d(self.middleplanes),
                                nolieaner,
                                nn.Conv2d(self.middleplanes,outplanes,1,1,0,bias=False),
                                nn.BatchNorm2d(outplanes),
            )
        self.shortcut = False
        if stride == 1 and inplanes != outplanes:
            self.shortcut = True
            self.shortcut_module = nn.Sequential(
                                        nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
                                        nn.BatchNorm2d(outplanes),
                )

    def forward(self,x):
        out = self.feature(x)
        if self.se:
            out = self.se_module(out)
        if self.shortcut:
            out = out + self.shortcut_module(x)
        return out

class MobileNetV1(nn.Module):
    def __init__(self,num_class=200):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Sequential(
                            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(32),
                            nn.ReLU6(inplace=True)
            )
        self.module1 = MobileNetV1Module(32, 64, 1)     # 112
        self.module2 = MobileNetV1Module(64, 128, 2)
        self.module3 = MobileNetV1Module(128, 128, 1)   # 56
        self.module4 = MobileNetV1Module(128, 256, 2)
        self.module5 = MobileNetV1Module(256, 256, 1)   #28
        self.module6 = MobileNetV1Module(256, 512, 2)
        self.module7 = MobileNetV1Module(512, 512, 1)   #14
        self.module8 = MobileNetV1Module(512, 512, 1)
        self.module9 = MobileNetV1Module(512, 512, 1)
        self.module10 = MobileNetV1Module(512, 512, 1)
        self.module11 = MobileNetV1Module(512, 512, 1)
        self.module12 = MobileNetV1Module(512, 1024, 2)
        self.module13 = MobileNetV1Module(1024, 1024, 1) #7
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_class)
        self._initialize_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.module1(out)
        out = self.module2(out)
        out = self.module3(out)
        out = self.module4(out)
        out = self.module5(out)
        out = self.module6(out)
        out = self.module7(out)
        out = self.module8(out)
        out = self.module9(out)
        out = self.module10(out)
        out = self.module11(out)
        out = self.module12(out)
        out = self.module13(out)
        out = self.avg_pool(out)
        out = out.view(-1,1024)
        out = self.fc(out)
        return out

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

class MobileNetV2(nn.Module):
    def __init__(self,num_class=200):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU6(inplace=True),
                        nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU6(inplace=True),
                        nn.Conv2d(32, 16, 1, 1, bias=False),
                        nn.BatchNorm2d(16),
            )
        # inplanes, outplanes, expand, stride
        self.module1 = MobileNetV2Module(16,24,6,2)
        self.module2 = MobileNetV2Module(24,24,6,1)
        self.module3 = MobileNetV2Module(24,32,6,2)
        self.module4 = MobileNetV2Module(32,32,6,1)
        self.module5 = MobileNetV2Module(32,32,6,1)
        self.module6 = MobileNetV2Module(32,64,6,2)
        self.module7 = MobileNetV2Module(64,64,6,1)
        self.module8 = MobileNetV2Module(64,64,6,1)
        self.module9 = MobileNetV2Module(64,64,6,1)
        self.module10 = MobileNetV2Module(64,96,6,1)
        self.module11 = MobileNetV2Module(96,96,6,1)
        self.module12 = MobileNetV2Module(96,96,6,1)
        self.module13 = MobileNetV2Module(96,160,6,2)
        self.module14 = MobileNetV2Module(160,160,6,1)
        self.module15 = MobileNetV2Module(160,160,6,1)
        self.module16 = MobileNetV2Module(160,320,6,1)
        self.conv2 = nn.Sequential(
                    nn.Conv2d(320, 1280, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(1280),
                    nn.ReLU6(inplace=True)
            )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_class)
        self._initialize_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.module1(out)
        out = self.module2(out)
        out = self.module3(out)
        out = self.module4(out)
        out = self.module5(out)
        out = self.module6(out)
        out = self.module7(out)
        out = self.module8(out)
        out = self.module9(out)
        out = self.module10(out)
        out = self.module11(out)
        out = self.module12(out)
        out = self.module13(out)
        out = self.module14(out)
        out = self.module15(out)
        out = self.module16(out)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = out.view(-1,1280)
        out = self.fc(out)
        return out

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

        
class MobileNetV3Large(nn.Module):
    def __init__(self,num_class=200):
        super(MobileNetV3Large, self).__init__()
        self.conv1 = nn.Sequential(
                            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(16),
                            hswish(),
            )
        self.module1 = MobileNetV3Module(16,16,1,3,1,nn.ReLU(inplace=True),False)
        self.module2 = MobileNetV3Module(16,24,4,3,2,nn.ReLU(inplace=True),False)
        self.module3 = MobileNetV3Module(24,24,3,3,1,nn.ReLU(inplace=True),False)
        self.module4 = MobileNetV3Module(24,40,3,5,2,nn.ReLU(inplace=True),True)
        self.module5 = MobileNetV3Module(40,40,3,5,1,nn.ReLU(inplace=True),True)
        self.module6 = MobileNetV3Module(40,40,3,5,1,nn.ReLU(inplace=True),True)
        self.module7 = MobileNetV3Module(40,80,6,3,2,hswish(),False)
        self.module8 = MobileNetV3Module(80,80,2.5,3,1,hswish(),False)
        self.module9 = MobileNetV3Module(80,80,2.3,3,1,hswish(),False)
        self.module10 = MobileNetV3Module(80,80,2.3,3,1,hswish(),False)
        self.module11 = MobileNetV3Module(80,112,6,3,1,hswish(),True)
        self.module12 = MobileNetV3Module(112,112,6,3,1,hswish(),True)
        self.module13 = MobileNetV3Module(112,160,6,5,1,hswish(),True)
        self.module14 = MobileNetV3Module(160,160,4.2,5,2,hswish(),True)
        self.module15 = MobileNetV3Module(160,160,6,5,1,hswish(),True)
        self.conv2 = nn.Sequential(
                            nn.Conv2d(160,960,1,1,0,bias=False),
                            nn.BatchNorm2d(960),
                            hswish()
            )
        self.fc1 = nn.Sequential(
                        nn.Linear(960,1280),
                        nn.BatchNorm1d(1280),
                        hswish()
            )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(1280, num_class)
        self._initialize_weights()

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
        out = self.conv1(x)
        out = self.module1(out)
        out = self.module2(out)
        out = self.module3(out)
        out = self.module4(out)
        out = self.module5(out)
        out = self.module6(out)
        out = self.module7(out)
        out = self.module8(out)
        out = self.module9(out)
        out = self.module10(out)
        out = self.module11(out)
        out = self.module12(out)
        out = self.module13(out)
        out = self.module14(out)
        out = self.module15(out)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class MobileNetV3Small(nn.Module):
    def __init__(self, num_class=200):
        super(MobileNetV3Small, self).__init__()
        self.conv1 = nn.Sequential(
                            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(16),
                            hswish(),
            )
        self.module1 = MobileNetV3Module(16,16,1,3,2,nn.ReLU(inplace=True),True)
        self.module2 = MobileNetV3Module(16,24,4.5,3,2,nn.ReLU(inplace=True),False)
        self.module3 = MobileNetV3Module(24,24,88/24,3,1,nn.ReLU(inplace=True),False)
        self.module4 = MobileNetV3Module(24,40,4,5,2,nn.ReLU(inplace=True),True)
        self.module5 = MobileNetV3Module(40,40,6,5,1,nn.ReLU(inplace=True),True)
        self.module6 = MobileNetV3Module(40,40,6,5,1,nn.ReLU(inplace=True),True)
        self.module7 = MobileNetV3Module(40,48,3,5,1,hswish(),True)
        self.module8 = MobileNetV3Module(48,48,3,5,1,hswish(),True)
        self.module9 = MobileNetV3Module(48,96,6,5,2,hswish(),True)
        self.module10 = MobileNetV3Module(96,96,6,5,1,hswish(),True)
        self.module11 = MobileNetV3Module(96,96,6,5,1,hswish(),True)
        self.conv2 = nn.Sequential(
                            nn.Conv2d(96,576,1,1,0,bias=False),
                            nn.BatchNorm2d(576),
                            hswish()
            )
        self.fc1 = nn.Sequential(
                        nn.Linear(576,1280),
                        nn.BatchNorm1d(1280),
                        hswish()
            )                
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(1280, num_class)
        self._initialize_weights()

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
        out = self.conv1(x)
        out = self.module1(out)
        out = self.module2(out)
        out = self.module3(out)
        out = self.module4(out)
        out = self.module5(out)
        out = self.module6(out)
        out = self.module7(out)
        out = self.module8(out)
        out = self.module9(out)
        out = self.module10(out)
        out = self.module11(out)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
        
        