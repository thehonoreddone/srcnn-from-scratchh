import os
import numpy as np
from src.model import SRCNN
from src.utils import load_image, preprocess_image

def create_simple_model():
    """Basit bir model oluştur ve kaydet."""
    print("Basit bir SRCNN modeli oluşturuluyor...")
    
    # Model klasörünü oluştur
    os.makedirs("models", exist_ok=True)
    
    # SRCNN modelini başlat
    model = SRCNN(input_channels=1)
    
    # Modeli kaydet
    model_path = "models/srcnn_final.npy"
    model.save_weights(model_path)
    
    print(f"Basit model kaydedildi: {model_path}")
    
    return model

def quick_test():
    """Hızlı bir test gerçekleştir."""
    # Model yoksa oluştur
    if not os.path.exists("models/srcnn_final.npy"):
        model = create_simple_model()
    else:
        print("Model zaten var, yükleniyor...")
        model = SRCNN(input_channels=1)
        model.load_weights("models/srcnn_final.npy")
    
    # Test görüntüsü yükle
    test_image_path = "Set5/butterfly.png"
    if os.path.exists(test_image_path):
        print(f"Test görüntüsü yükleniyor: {test_image_path}")
        img = load_image(test_image_path)
        
        # Görüntüyü işle
        scale_factor = 2
        bicubic_y, hr_y, _ = preprocess_image(img, scale_factor)
        
        # Model için girişi şekillendir
        bicubic_input = bicubic_y.reshape(1, 1, *bicubic_y.shape)
        
        # İleriye doğru geçiş
        print("Model çalıştırılıyor...")
        output = model.forward(bicubic_input)
        
        print("Model başarıyla çalıştı ve çıktı üretti!")
        print(f"Çıktı şekli: {output.shape}")
    else:
        print(f"Test görüntüsü bulunamadı: {test_image_path}")

if __name__ == "__main__":
    quick_test() 