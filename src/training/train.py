import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Modül yollarını ayarla
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from src.models.siamese import SiameseNetwork
from src.models.loss import TripletMarginLossCustom
from src.data_loader.dataset import TripletEcommerceDataset

def train_model():
    # 1. Hiperparametreler ve Konfigürasyon
    BATCH_SIZE = 16  # 4GB VRAM için güvenli sınır. OOM yersen 8'e düşür.
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    EMBEDDING_DIM = 512
    DB_PATH = os.path.join(root_dir, "data", "metadata.db")
    MODEL_SAVE_DIR = os.path.join(root_dir, "data", "models")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Eğitim cihazı: {device}")

    # 2. Veri Hazırlığı
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TripletEcommerceDataset(db_path=DB_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    print(f"[*] Toplam Batch Sayısı (1 Epoch için): {len(dataloader)}")

    # 3. Model, Loss ve Optimizer
    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM, pretrained=True).to(device)
    criterion = TripletMarginLossCustom(margin=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 4GB VRAM kurtarıcısı: AMP Scaler
    scaler = GradScaler()

    # 4. Eğitim Döngüsü
    print("[*] Eğitim Başlıyor...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for anchor, positive, negative in progress_bar:
            # Verileri GPU'ya taşı
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            # AMP: İleri beslemeyi (Forward Pass) 16-bit hassasiyetle yap
            with autocast():
                out_a, out_p, out_n = model(anchor, positive, negative)
                loss = criterion(out_a, out_p, out_n)
                
            # Geri yayılım (Backward) ve optimizasyon
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{(running_loss / (progress_bar.n + 1)):.4f}"})
            
        epoch_loss = running_loss / len(dataloader)
        print(f"[-] Epoch {epoch+1} Tamamlandı. Ortalama Kayıp: {epoch_loss:.4f}")
        
    # 5. Eğitilmiş Modeli Kaydet
    save_path = os.path.join(MODEL_SAVE_DIR, "visual_search_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"[+] Model başarıyla kaydedildi: {save_path}")

if __name__ == "__main__":
    train_model()