import os
import numpy as np
from PIL import Image
import argparse

from src.model import SRCNN
from src.utils import load_image, preprocess_image, postprocess_image, calculate_psnr

def test_single_image(image_path, output_dir, scale_factor=2):
    """Tek bir görüntü üzerinde SRCNN modelini test eder."""
    # Çıktı klasörünü oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Görüntüyü yükle
    print(f"Görüntü yükleniyor: {image_path}")
    img = load_image(image_path)
    img_filename = os.path.basename(image_path)
    
    # Görüntüyü ön işle
    print(f"Ölçek faktörü {scale_factor}x ile ön işleme yapılıyor...")
    bicubic_y, hr_y, (cb, cr) = preprocess_image(img, scale_factor)
    
    # Görüntüleri kaydet
    # Orijinal görüntü
    img.save(os.path.join(output_dir, f"original_{img_filename}"))
    
    # Bicubic görüntü
    bicubic_img = postprocess_image(bicubic_y, cb, cr)
    bicubic_img.save(os.path.join(output_dir, f"bicubic_x{scale_factor}_{img_filename}"))
    
    # SRCNN görüntüsünü oluşturmak için bir model yüklemek istersen:
    try:
        # Model dosyası var mı kontrol et
        model_path = "models/srcnn_final.npy"
        if os.path.exists(model_path):
            print("Model yükleniyor...")
            model = SRCNN(input_channels=1)
            model.load_weights(model_path)
            
            # Girişi model için şekillendir
            bicubic_input = bicubic_y.reshape(1, 1, *bicubic_y.shape)
            
            # Model çıktısını al
            print("Süper çözünürlük modeli çalıştırılıyor...")
            output = model.forward(bicubic_input).squeeze()
            
            # SRCNN görüntüsü
            srcnn_img = postprocess_image(output, cb, cr)
            srcnn_img.save(os.path.join(output_dir, f"srcnn_x{scale_factor}_{img_filename}"))
            
            # PSNR değerlerini hesapla
            psnr_bicubic = calculate_psnr(bicubic_y, hr_y)
            psnr_srcnn = calculate_psnr(output, hr_y)
            
            print(f"\nPSNR (Bicubic): {psnr_bicubic:.2f} dB")
            print(f"PSNR (SRCNN): {psnr_srcnn:.2f} dB")
            print(f"İyileştirme: {psnr_srcnn - psnr_bicubic:.2f} dB")
        else:
            print(f"Model dosyası bulunamadı: {model_path}")
            print("Sadece bicubic yükseltilmiş görüntü kaydedildi.")
    except Exception as e:
        print(f"Model kullanımında hata oluştu: {e}")
        print("Sadece bicubic yükseltilmiş görüntü kaydedildi.")
    
    print(f"\nGörüntüler şu klasöre kaydedildi: {output_dir}")
    print(f"Dosyalar: {os.listdir(output_dir)}")

def main():
    parser = argparse.ArgumentParser(description='Tek görüntü SRCNN testi')
    parser.add_argument('--image', type=str, default='Set5/butterfly.png',
                        help='test edilecek görüntünün yolu')
    parser.add_argument('--output_dir', type=str, default='test_output',
                        help='çıktı görüntülerinin kaydedileceği klasör')
    parser.add_argument('--scale', type=int, default=2,
                        help='süper çözünürlük ölçek faktörü (2, 3 veya 4)')
    
    args = parser.parse_args()
    
    test_single_image(args.image, args.output_dir, args.scale)

if __name__ == '__main__':
    main() 