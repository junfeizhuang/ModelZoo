import math
import torch
import torch.nn as nn
__all__ = ['EfficientNet_B0','EfficientNet_B1','EfficientNet_B2','EfficientNet_B3',
            'EfficientNet_B4','EfficientNet_B5','EfficientNet_B6']

class SEModule(nn.Module):
    def __init__(self, in_, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_, squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(squeeze_ch, in_, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DropConnect(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def forward(self, x):
        if not self.training:
            return x

        random_tensor = self.ratio
        random_tensor += torch.rand([x.shape[0], 1, 1, 1], dtype=torch.float, device=x.device)
        random_tensor.requires_grad_(False)
        return x / self.ratio * random_tensor.floor()

class _conv_bn(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride=1,group=1, use_act=False):
        super(_conv_bn, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size,stride=stride, padding=kernel_size // 2, groups=group, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes,eps=1e-3, momentum=0.99)
        self.use_act = use_act
        if self.use_act:
            self.act = Swish()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.use_act:
            x = self.act(x)
        return x
        
class MBConv(nn.Module):
    """docstring for MBConv"""
    def __init__(self, inplanes, outplanes, expand, kernel_size, stride, droprate):
        super(MBConv, self).__init__()
        midplanes = inplanes * expand
        self.layer1 = _conv_bn(inplanes, midplanes, 1, use_act=True) if expand != 1 else nn.Sequential()
        self.layer2 = _conv_bn(midplanes, midplanes, kernel_size, group=midplanes,stride=stride, use_act=True)
        self.layer3 = _conv_bn(midplanes, outplanes, 1)
        self.skip = True if stride == 1 and inplanes == outplanes else False
        self.se =  SEModule(midplanes, int(inplanes* 0.25))
        self.dropconnect = DropConnect(droprate)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.se(out)
        out = self.layer3(out)
        if self.skip:
            out = self.dropconnect(out) + x
        return out  

class EfficientNet(nn.Module):
    def __init__(self,width_coeff,depth_coeff,droprate, num_classes=200):
        super(EfficientNet, self).__init__()
        self.depth_coeff = depth_coeff
        self.width_coeff = width_coeff
        self.conv1 = _conv_bn(3, 32, 3, 2, use_act=True)

        self.blocks = nn.Sequential(
            self._make_layer(self._obtain_channels(32), self._obtain_channels(16), 1, 3, 1, self._obtain_repeats(1),droprate),
            self._make_layer(self._obtain_channels(16), self._obtain_channels(24), 6, 3, 2, self._obtain_repeats(2),droprate),
            self._make_layer(self._obtain_channels(24), self._obtain_channels(40), 6, 5, 2, self._obtain_repeats(2),droprate),
            self._make_layer(self._obtain_channels(40), self._obtain_channels(80), 6, 3, 2, self._obtain_repeats(3),droprate),
            self._make_layer(self._obtain_channels(80), self._obtain_channels(112), 6, 5, 1, self._obtain_repeats(3),droprate),
            self._make_layer(self._obtain_channels(112), self._obtain_channels(192), 6, 5, 2, self._obtain_repeats(4),droprate),
            self._make_layer(self._obtain_channels(192), self._obtain_channels(320), 6, 3, 1, self._obtain_repeats(1),droprate),
        )
        self.last_process = nn.Sequential(
            _conv_bn(self._obtain_channels(320), self._obtain_channels(1280), 1, use_act=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(droprate, True)
        )
        self.linear = nn.Linear(self._obtain_channels(1280), num_classes)
        self._initialize_weights()

    def _obtain_channels(self, inplanes, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        outplanes = max(min_value, int(inplanes + divisor / 2) // divisor * divisor)
        if outplanes < 0.9 * inplanes:
            outplanes += divisor
        return outplanes

    def _obtain_repeats(self, x):
        return int(math.ceil(x * self.depth_coeff))

    def _make_layer(self, inplanes, outplanes, expand, kernel_size, stride, repeats, droprate):
        layers = []
        layers.append(MBConv(inplanes, outplanes, expand, kernel_size, stride, droprate))
        for _ in range(repeats - 1):
            layers.append(MBConv(outplanes, outplanes, expand, kernel_size, 1, droprate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.last_process(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

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

def EfficientNet_B0(num_classes=200, **kwargs):
    input_size = 224
    model = EfficientNet(1., 1., 0.2, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B1(num_classes=200, **kwargs):
    input_size = 240
    model = EfficientNet(1., 1.1, 0.2, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B2(num_classes=200, **kwargs):
    input_size = 260
    model = EfficientNet(1.1, 1.2, 0.3, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B3(num_classes=200, **kwargs):
    input_size = 300
    model = EfficientNet(1.2, 1.4, 0.4, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B4(num_classes=200, **kwargs):
    input_size = 300
    model = EfficientNet(1.4, 1.8, 0.4, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B5(num_classes=200, **kwargs):
    input_size = 456
    model = EfficientNet(1.6, 2.2, 0.4, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B6(num_classes=200, **kwargs):
    input_size = 528
    model = EfficientNet(1.8, 2.6, 0.5, num_classes=num_classes, **kwargs)
    return model


        