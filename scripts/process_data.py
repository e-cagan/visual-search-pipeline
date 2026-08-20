import os
import sys
import sqlite3
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.image_ops import make_square_with_padding, is_image_corrupted_or_blank

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
DB_PATH = "data/metadata.db"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_dataset():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    # Veritabanındaki tüm görselleri çek
    cur.execute("SELECT id, product_id, image_path FROM products")
    rows = cur.fetchall()
    
    print(f"[*] İşlenecek toplam görsel sayısı: {len(rows)}")
    
    processed_count = 0
    deleted_count = 0
    
    for row in tqdm(rows, desc="Processing Images"):
        db_id, p_id, raw_path = row
        
        # Dosya gerçekten diskte var mı?
        if not os.path.exists(raw_path):
            cur.execute("DELETE FROM products WHERE id = ?", (db_id,))
            deleted_count += 1
            continue
            
        # Görseli oku ve kare yap
        processed_img = make_square_with_padding(raw_path, target_size=224)
        
        # Varyans kontrolünden geçemezse (veya okunamazsa) kaydı sil
        if is_image_corrupted_or_blank(processed_img):
            cur.execute("DELETE FROM products WHERE id = ?", (db_id,))
            # Raw dosyayı diskten silebiliriz (isteğe bağlı)
            try:
                os.remove(raw_path)
            except OSError:
                pass
            deleted_count += 1
            continue
            
        # Başarılı olanı processed klasörüne kaydet
        filename = os.path.basename(raw_path)
        processed_path = os.path.join(PROCESSED_DIR, filename)
        
        cv2.imwrite(processed_path, processed_img)
        
        # Veritabanındaki dosya yolunu yeni klasör olarak güncelle
        cur.execute("UPDATE products SET image_path = ? WHERE id = ?", (processed_path, db_id))
        processed_count += 1
        
    con.commit()
    con.close()
    
    print(f"\n[+] İşlem tamamlandı. Başarılı: {processed_count}, Çöp/Hatalı (Silinen): {deleted_count}")

if __name__ == "__main__":
    process_dataset()