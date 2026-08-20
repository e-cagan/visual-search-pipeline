import cv2
import numpy as np

import cv2
import numpy as np

def make_square_with_padding(image_path, target_size=224):
    """
    Görselin aspect-ratio'sunu bozmadan kare yapar.
    Şeffaf (Alpha) kanallı görselleri tespit edip siyah arka planla harmanlar.
    """
    # cv2.IMREAD_UNCHANGED ile okuyarak Alpha kanalını da al
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Eğer görsel 4 kanallıysa (BGRA), Alpha Blending yap
    if len(img.shape) == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:] / 255.0
        
        # Siyah bir arka plan oluştur (Eğer beyaz istersen np.zeros yerine np.ones * 255 kullan)
        background = np.zeros_like(bgr, dtype=np.uint8)
        
        # Şeffaf pikselleri arka planla harmanla
        img = (bgr * alpha + background * (1 - alpha)).astype(np.uint8)
    
    # Eğer görsel siyah beyaz (Grayscale) geldiyse 3 kanala (BGR) çevir
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    max_side = max(h, w)
    
    top = (max_side - h) // 2
    bottom = max_side - h - top
    left = (max_side - w) // 2
    right = max_side - w - left
    
    color = [0, 0, 0] # Padding rengi
    squared_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    resized_img = cv2.resize(squared_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return resized_img

def is_image_corrupted_or_blank(img, variance_threshold=50.0):
    # Bu fonksiyon öncekiyle tamamen aynı kalacak
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    if variance < variance_threshold:
        return True
    return False