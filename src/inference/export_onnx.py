import os
import sys
import torch

# Proje dizinini ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from src.models.siamese import SiameseNetwork

def export_to_onnx():
    MODEL_PATH = os.path.join(root_dir, "data", "models", "visual_search_model.pt")
    ONNX_PATH = os.path.join(root_dir, "data", "models", "visual_search_model.onnx")

    print("[*] PyTorch modeli yükleniyor...")
    # ONNX dönüşümleri CPU üzerinde daha stabil ve hatasız gerçekleşir
    device = torch.device("cpu") 
    
    # Pretrained=False çünkü kendi eğittiğimiz ağırlıkları yükleyeceğiz
    model = SiameseNetwork(embedding_dim=512, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # Modeli 'eval' (çıkarım) moduna al. Dropout, BatchNorm gibi katmanları dondurur.
    model.eval()

    # Modelin içine girecek verinin boyutlarını taklit eden bir kukla tensör (Batch=1, Ch=3, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    print("[*] Modelin işlem grafiği (Computational Graph) çıkarılıyor ve ONNX'e dönüştürülüyor...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=14,               # ONNX sürümü (14 genelde çok stabildir)
        do_constant_folding=True,       # Sabit ağırlıkları birleştirerek modeli hızlandırır
        input_names=['input'],          # C++ tarafından modeli çağırırken kullanacağımız giriş adı
        output_names=['output'],        # C++ tarafından vektörü okurken kullanacağımız çıkış adı
        dynamic_axes={                  # Batch size'ı dinamik yapıyoruz (1 görsel de girebilir, 10 görsel de)
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )

    print(f"[+] Başarılı! ONNX modeli üretime hazır: {ONNX_PATH}")

if __name__ == "__main__":
    export_to_onnx()