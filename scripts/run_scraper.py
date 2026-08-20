import os
import sys
import sqlite3
import requests
from tqdm import tqdm

# Proje ana dizinini ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DIR = "data/raw"
DB_PATH = "data/metadata.db"
os.makedirs(RAW_DIR, exist_ok=True)

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT,
            category TEXT,
            brand TEXT,
            image_path TEXT UNIQUE
        )
    """)
    con.commit()
    return con

def fetch_and_save():
    con = init_db()
    cur = con.cursor()

    print("[*] E-ticaret API'lerinden ürün verileri çekiliyor...")
    
    # 1. Kaynak: DummyJSON Products (Tüm kategoriler)
    url_dummy = "https://dummyjson.com/products?limit=200"
    res = requests.get(url_dummy).json()
    products = res.get("products", [])

    # 2. Kaynak: Platzi Fake Store API (Giyim/Ayakkabı/Elektronik)
    url_platzi = "https://api.escuelajs.co/api/v1/products"
    res_platzi = requests.get(url_platzi).json()
    
    all_items = []
    
    for p in products:
        images = p.get("images", [])
        if len(images) > 1: # Siamese için en az 2 görsel şart
            all_items.append({
                "product_id": f"dummy_{p.get('id')}",
                "category": p.get("category", "General"),
                "brand": p.get("brand", "Generic"),
                "images": images
            })
            
    for p in res_platzi:
        images = p.get("images", [])
        # Platzi bazen bozuk URL formatı döner, temizle
        clean_images = [img.strip('["\']') for img in images if img.startswith("http")]
        if len(clean_images) > 1:
            all_items.append({
                "product_id": f"platzi_{p.get('id')}",
                "category": p.get("category", {}).get("name", "Fashion"),
                "brand": "FashionBrand",
                "images": clean_images
            })

    print(f"[*] Toplam {len(all_items)} uygun ürün bulundu. Görseller indiriliyor...")

    downloaded_count = 0
    for item in tqdm(all_items, desc="Downloading"):
        p_id = item["product_id"]
        category = item["category"]
        brand = item["brand"]
        
        for idx, img_url in enumerate(item["images"]):
            img_name = f"{p_id}_{idx}.jpg"
            img_path = os.path.join(RAW_DIR, img_name)
            
            if not os.path.exists(img_path):
                try:
                    r = requests.get(img_url, timeout=5)
                    if r.status_code == 200 and len(r.content) > 1000: # Boş/hatalı görselleri atla
                        with open(img_path, "wb") as f:
                            f.write(r.content)
                        
                        cur.execute("""
                            INSERT OR IGNORE INTO products (product_id, category, brand, image_path)
                            VALUES (?, ?, ?, ?)
                        """, (p_id, category, brand, img_path))
                        downloaded_count += 1
                except Exception:
                    continue
        con.commit()

    con.close()
    print(f"\n[+] İndirme tamamlandı. Toplam {downloaded_count} yeni görsel kaydedildi.")

if __name__ == "__main__":
    fetch_and_save()