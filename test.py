import torch
from torchsummary import summary
from torchvision.models import resnet18
from MobileNetModels import MobileNetV1,MobileNetV2,MobileNetV3Large,MobileNetV3Small
from ResNetModels import resnet18 as res18
from ResNextModels import resnext50 as resx50
from ResNextModels import resnext18 as resx18
from Res2NetModels import res2net50
from EfficientNetModels import EfficientNet_B0
model = EfficientNet_B0().cuda()
print(model)
summary(model, ( 3, 224, 224))
# model = resnet18().cuda()
# summary(model, ( 3, 224, 224))