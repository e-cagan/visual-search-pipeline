import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Backbone, self).__init__()
        
        # Hazır ağırlıkları yükle (Transfer Learning)
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
        
        # Orijinal ResNet'in amacı 1000 sınıflık sınıflandırma yapmaktır.
        # Biz sınıflandırma yapmıyoruz, bu yüzden son Fully Connected (fc) katmanını siliyoruz.
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        # x boyutu: [Batch, 3, 224, 224]
        x = self.features(x)
        # Pooling sonrası boyut [Batch, 512, 1, 1] olur, bunu [Batch, 512] yapıyoruz
        x = torch.flatten(x, 1)
        return x