import os
import sqlite3
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TripletEcommerceDataset(Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.transform = transform
        
        # SQLite'a bağlanıp verileri RAM'e alıyoruz (I/O darboğazını önlemek için)
        self.con = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cur = self.con.cursor()
        
        self.cur.execute("SELECT product_id, image_path FROM products")
        rows = self.cur.fetchall()
        
        # Ürünleri ID'lerine göre grupla: {'dummy_1': ['path1.jpg', 'path2.jpg'], ...}
        self.product_to_images = {}
        for p_id, path in rows:
            # Eğer dosya fiziksel olarak yoksa atla
            if not os.path.exists(path):
                continue
                
            if p_id not in self.product_to_images:
                self.product_to_images[p_id] = []
            self.product_to_images[p_id].append(path)
            
        # Sadece birden fazla görseli olan ürünleri filtrele (Anchor ve Positive eşleşmesi için şart)
        self.valid_products = [p_id for p_id, paths in self.product_to_images.items() if len(paths) > 1]
        
        if len(self.valid_products) == 0:
            raise ValueError("Hata: Veritabanında birden fazla görseli olan hiçbir ürün bulunamadı!")

    def __len__(self):
        # Toplam geçerli ürün sayısı kadar epoch adımı olacak
        return len(self.valid_products)

    def __getitem__(self, idx):
        # 1. Anchor ve Positive Seçimi
        product_id = self.valid_products[idx]
        images = self.product_to_images[product_id]
        
        # Aynı üründen rastgele 2 farklı görsel seç
        anchor_path, positive_path = random.sample(images, 2)
        
        # 2. Negative Seçimi
        # Başka bir rastgele ürün seç (Anchor ile aynı ID olmamasına dikkat et)
        negative_product_id = random.choice(self.valid_products)
        while negative_product_id == product_id:
            negative_product_id = random.choice(self.valid_products)
            
        negative_path = random.choice(self.product_to_images[negative_product_id])
        
        # 3. Görselleri Belleğe Yükle ve Tensöre Çevir
        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        return anchor_img, positive_img, negative_img

# Test için küçük bir script
if __name__ == "__main__":
    import numpy as np
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Kodun nereden çalıştırıldığından bağımsız olarak kök dizini (root) bul
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        db_file = os.path.join(root_dir, "data", "metadata.db")
        
        dataset = TripletEcommerceDataset(db_path=db_file, transform=transform)
        print(f"[+] Dataset başarıyla yüklendi. Toplam geçerli ürün: {len(dataset)}")
        
        a, p, n = dataset[0]
        print(f"[*] Tensör Boyutları -> Anchor: {a.shape}, Positive: {p.shape}, Negative: {n.shape}")
    except Exception as e:
        print(f"[-] Hata: {e}")