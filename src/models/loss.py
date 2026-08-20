import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletMarginLossCustom(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletMarginLossCustom, self).__init__()
        self.margin = margin
        # max(0, x) işlemi matematiksel olarak ReLU ile tamamen aynıdır
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative):
        # 1. Anchor ile Positive arasındaki Öklid mesafesi
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        
        # 2. Anchor ile Negative arasındaki Öklid mesafesi
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # 3. Triplet Loss Formülü
        losses = self.relu(distance_positive - distance_negative + self.margin)
        
        # Tüm batch'in ortalama kaybını döndür
        return losses.mean()

# Test Bloğu
if __name__ == "__main__":
    # Matematiksel testi yapalım
    loss_fn = TripletMarginLossCustom(margin=0.2)
    
    # 1. Senaryo: Model Mükemmel Öğrenmiş
    # Anchor ile Positive aynı yerde, Negative çok uzakta
    a = torch.tensor([[1.0, 0.0]])
    p = torch.tensor([[1.0, 0.0]])
    n = torch.tensor([[-1.0, 0.0]])
    
    loss_perfect = loss_fn(a, p, n)
    print(f"[*] Mükemmel Model Kaybı (0.0 olmalı): {loss_perfect.item()}")
    
    # 2. Senaryo: Model Berbat Durumda
    # Anchor ile Negative aynı yerde, Positive çok uzakta
    loss_terrible = loss_fn(a, n, p)
    print(f"[*] Berbat Model Kaybı (Çok yüksek olmalı): {loss_terrible.item()}")