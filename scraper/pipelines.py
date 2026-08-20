import os
import sqlite3
import requests
from itemadapter import ItemAdapter

class VisualSearchPipeline:
    def __init__(self):
        # Klasör yollarını belirle
        self.raw_dir = 'data/raw'
        os.makedirs(self.raw_dir, exist_ok=True)
        
        # SQLite veritabanını oluştur/bağlan
        self.con = sqlite3.connect('data/metadata.db')
        self.cur = self.con.cursor()
        self.create_table()

    def create_table(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT,
                category TEXT,
                brand TEXT,
                image_path TEXT UNIQUE
            )
        """)
        self.con.commit()

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        product_id = adapter.get('product_id')
        image_urls = adapter.get('image_urls', [])
        
        for idx, url in enumerate(image_urls):
            # Görseli indir (gerçek projede asenkron yapılır ama şimdilik request yeterli)
            image_name = f"{product_id}_{idx}.jpg"
            image_path = os.path.join(self.raw_dir, image_name)
            
            if not os.path.exists(image_path):
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        with open(image_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Veritabanına kaydet
                        self.cur.execute("""
                            INSERT OR IGNORE INTO products (product_id, category, brand, image_path)
                            VALUES (?, ?, ?, ?)
                        """, (product_id, adapter.get('category'), adapter.get('brand'), image_path))
                        self.con.commit()
                except Exception as e:
                    spider.logger.error(f"Görsel indirilemedi: {url} - Hata: {e}")
                    
        return item

    def close_spider(self, spider):
        self.con.close()