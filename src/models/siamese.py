import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from src.models.backbone import ResNet18Backbone

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=512, pretrained=True):
        super(SiameseNetwork, self).__init__()
        
        # Ortak omurga (Shared Weights) - İkiz ağların ağırlıkları ortaktır
        self.backbone = ResNet18Backbone(pretrained=pretrained)
        
        # Projection Head: Özellikleri nihai vektör uzayına taşıyan MLP
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )

    def forward_once(self, x):
        # 1. Görüntüden özellik haritasını çıkar
        features = self.backbone(x)
        
        # 2. Vektör uzayına yansıt (Embeddings)
        embeddings = self.projection_head(features)
        
        # 3. L2 Normalizasyonu (Çok Kritik!)
        # Bu işlem vektörlerin uzunluğunu 1 yapar, böylece Euclidean mesafe
        # Cosine benzerliği ile orantılı hale gelir.
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def forward(self, anchor, positive=None, negative=None):
        # Eğitim sırasında (Training) üç görsel de aynı anda gelir
        if positive is not None and negative is not None:
            emb_a = self.forward_once(anchor)
            emb_p = self.forward_once(positive)
            emb_n = self.forward_once(negative)
            return emb_a, emb_p, emb_n
        
        # Çıkarım sırasında (Inference/Deployment) sadece tek görsel gelir
        return self.forward_once(anchor)

# Test Bloğu
if __name__ == "__main__":
    # 4 GB VRAM dostu, batch_size=4 olacak şekilde rastgele tensör üret
    dummy_anchor = torch.randn(4, 3, 224, 224)
    dummy_positive = torch.randn(4, 3, 224, 224)
    dummy_negative = torch.randn(4, 3, 224, 224)
    
    model = SiameseNetwork(embedding_dim=512)
    
    # 1. Eğitim simülasyonu
    out_a, out_p, out_n = model(dummy_anchor, dummy_positive, dummy_negative)
    print(f"[*] Triplet Çıktı Boyutu: Anchor -> {out_a.shape}")
    
    # L2 Norm test: Vektör uzunluğu (normu) 1.0 olmalı
    norm_value = torch.norm(out_a[0], p=2).item()
    print(f"[*] Vektör L2 Normu (1.0 olmalı): {norm_value:.4f}")