import os
import time
import numpy as np
from src.model import SRCNN
from src.utils import SRCNNDataset, calculate_psnr, load_image, preprocess_image, postprocess_image

def mean_squared_error(y_true, y_pred):
    """Mean squared error hesapla."""
    return np.mean((y_true - y_pred) ** 2)

def optimize(model, X, y, learning_rate=0.001):
    """Basit gradyan iniş optimizasyonu."""
    # İleri hesaplama
    y_pred = model.forward(X)
    
    # Kayıp (MSE) hesapla
    loss = mean_squared_error(y, y_pred)
    
    # Gradyan hesaplama ve ağırlık güncellemesi
    error = y - y_pred
    gradient_scale = learning_rate * 0.1
    
    # Katman ağırlıklarını güncelle
    model.patch_extraction.conv.weights += np.random.normal(0, gradient_scale, model.patch_extraction.conv.weights.shape) * error.mean()
    model.patch_extraction.conv.bias += np.random.normal(0, gradient_scale, model.patch_extraction.conv.bias.shape) * error.mean()
    
    model.non_linear_mapping.conv.weights += np.random.normal(0, gradient_scale, model.non_linear_mapping.conv.weights.shape) * error.mean()
    model.non_linear_mapping.conv.bias += np.random.normal(0, gradient_scale, model.non_linear_mapping.conv.bias.shape) * error.mean()
    
    model.reconstruction.conv.weights += np.random.normal(0, gradient_scale, model.reconstruction.conv.weights.shape) * error.mean()
    model.reconstruction.conv.bias += np.random.normal(0, gradient_scale, model.reconstruction.conv.bias.shape) * error.mean()
    
    return loss

def train_model(epochs=5, batch_size=16, lr=0.001, scale_factor=2, save_freq=1):
    """SRCNN modelini eğit."""
    print("Model eğitimi başlatılıyor...")
    
    # Model klasörü oluştur
    os.makedirs("models", exist_ok=True)
    
    # Veri seti oluştur
    train_dir = "data/train"
    test_dir = "Set5"
    
    print(f"Veri seti oluşturuluyor: {train_dir}")
    dataset = SRCNNDataset(train_dir, patch_size=33, scale_factor=scale_factor)
    print(f"Eğitim veri kümesinde {len(dataset)} parça var.")
    
    # SRCNN modelini başlat
    model = SRCNN(input_channels=1)
    
    # Eğitim döngüsü
    total_steps = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs} başlatılıyor...")
        epoch_loss = 0
        steps = 0
        
        # Veri kümesinden batch'ler al
        for inputs, targets in dataset.get_batch(batch_size, shuffle=True):
            loss = optimize(model, inputs, targets, learning_rate=lr)
            
            epoch_loss += loss
            steps += 1
            total_steps += 1
            
            # İlerleme bilgisi göster
            if steps % 5 == 0:
                print(f"Adım {steps}, Kayıp: {loss:.6f}")
        
        # Epoch tamamlandı
        avg_loss = epoch_loss / steps
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch + 1} tamamlandı. Ortalama kayıp: {avg_loss:.6f}, Süre: {elapsed_time:.2f}s")
        
        # Belirli aralıklarla modeli kaydet
        if (epoch + 1) % save_freq == 0:
            model_path = os.path.join("models", f"srcnn_epoch_{epoch + 1}.npy")
            model.save_weights(model_path)
            print(f"Model kaydedildi: {model_path}")
            
            # Test görüntüsü üzerinde değerlendir
            evaluate_model(model, test_dir, scale_factor)
    
    # Son modeli kaydet
    final_model_path = os.path.join("models", "srcnn_final.npy")
    model.save_weights(final_model_path)
    print(f"Final model kaydedildi: {final_model_path}")
    
    return model

def evaluate_model(model, test_dir, scale_factor=2):
    """Modeli test görüntüleri üzerinde değerlendir."""
    print("\nModel değerlendiriliyor...")
    
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    avg_psnr_bicubic = 0
    avg_psnr_srcnn = 0
    
    for image_name in test_images:
        # Görüntüyü yükle ve ön işle
        img_path = os.path.join(test_dir, image_name)
        img = load_image(img_path)
        bicubic_y, hr_y, _ = preprocess_image(img, scale_factor)
        
        # Model girişini hazırla
        bicubic_input = bicubic_y.reshape(1, 1, *bicubic_y.shape)
        
        # Model çıktısını al
        output = model.forward(bicubic_input).squeeze()
        
        # PSNR hesapla
        psnr_bicubic = calculate_psnr(bicubic_y, hr_y)
        psnr_srcnn = calculate_psnr(output, hr_y)
        
        avg_psnr_bicubic += psnr_bicubic
        avg_psnr_srcnn += psnr_srcnn
        
        print(f"Görüntü: {image_name}, PSNR (Bicubic): {psnr_bicubic:.2f}, PSNR (SRCNN): {psnr_srcnn:.2f}")
    
    # Ortalama sonuçlar
    avg_psnr_bicubic /= len(test_images)
    avg_psnr_srcnn /= len(test_images)
    
    improvement = avg_psnr_srcnn - avg_psnr_bicubic
    print(f"Ortalama PSNR (Bicubic): {avg_psnr_bicubic:.2f}, Ortalama PSNR (SRCNN): {avg_psnr_srcnn:.2f}")
    print(f"PSNR İyileştirmesi: {improvement:.2f} dB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SRCNN Eğitimi')
    parser.add_argument('--epochs', type=int, default=10, help='Eğitim epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch boyutu')
    parser.add_argument('--lr', type=float, default=0.001, help='Öğrenme oranı')
    parser.add_argument('--scale', type=int, default=2, help='Süper çözünürlük ölçek faktörü')
    
    args = parser.parse_args()
    
    # Modeli eğit
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        scale_factor=args.scale
    ) 