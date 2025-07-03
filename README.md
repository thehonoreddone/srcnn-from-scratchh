# SRCNN - Super Resolution CNN (Sıfırdan İmplementasyon)

Bu proje, ["Image Super-Resolution Using Deep Convolutional Networks"](https://arxiv.org/abs/1501.00092) (Dong et al., 2014) makalesi temel alınarak oluşturulmuş bir SRCNN (Super Resolution Convolutional Neural Network) implementasyonudur. Implementasyon tamamen **saf Python ve NumPy** kullanılarak, hiçbir derin öğrenme kütüphanesi kullanmadan sıfırdan gerçekleştirilmiştir.

## Proje Yapısı

```
srcnn-from-scratch/
├── src/
│   ├── __init__.py
│   ├── layers.py     # Sıfırdan konvolüsyon katmanları ve aktivasyon fonksiyonları
│   ├── model.py      # Ana SRCNN model mimarisi
│   ├── utils.py      # Yardımcı fonksiyonlar (veri yükleme, bicubic interpolasyon, PSNR)
│   ├── train.py      # Eğitim ve test prosedürleri
│   └── eval.py       # Farklı ölçeklerde PSNR değerlendirmesi
├── Set5/             # Test veri seti
├── models/           # Eğitilmiş modellerin kaydedileceği yer
├── results/          # Test sonuçları
└── eval_results/     # Değerlendirme sonuçları
```

## Gereksinimler

Projeyi çalıştırmak için minimum gereksinimler:

```
numpy
pillow
```

## Özellikler

- Sıfırdan konvolüsyon operasyonu implementasyonu
- Sıfırdan bicubic interpolasyon implementasyonu
- NumPy tabanlı model ve eğitim döngüsü
- Gradyan iniş optimizasyonu için basit bir implementasyon
- PSNR (Peak Signal-to-Noise Ratio) hesaplama

## Kullanım

### Modeli Eğitmek İçin

```bash
python -m src.train --mode train --train_dir data/train --test_dir Set5 --epochs 50
```

### Modeli Test Etmek İçin

```bash
python -m src.train --mode test --test_dir Set5 --model_path models/srcnn_final.npy --output_dir results
```

### Modeli Değerlendirmek ve PSNR Hesaplamak İçin

```bash
python -m src.eval --test_dir Set5 --model_path models/srcnn_final.npy --output_dir eval_results --scale_factors 2 3 4 --save_images
```

Bu komut Set5 veri seti üzerinde modeli değerlendirir, 2x, 3x ve 4x ölçeklendirme faktörleri için PSNR değerlerini hesaplar ve sonuçları görselleştirir.

## SRCNN Mimarisi

SRCNN, düşük çözünürlüklü bir görüntüyü daha yüksek çözünürlüklü bir görüntüye dönüştürmek için tasarlanmış üç katmanlı bir CNN'dir:

1. **Patch Extraction and Representation**: 9x9 filtre boyutlu konvolüsyon katmanı, 64 filtre
2. **Non-linear Mapping**: 1x1 filtre boyutlu konvolüsyon katmanı, 32 filtre
3. **Reconstruction**: 5x5 filtre boyutlu konvolüsyon katmanı, 1 filtre

## Nasıl Çalışır?

1. Düşük çözünürlüklü bir görüntü alınır ve bicubic interpolation ile orijinal boyuta getirilir.
2. Bu bicubic interpolasyonlu görüntü SRCNN modeline giriş olarak verilir.
3. Model, bicubic görüntüden daha yüksek kaliteli bir süper çözünürlüklü görüntü üretir.

## Performans Değerlendirmesi

Modelin performansı, PSNR (Peak Signal-to-Noise Ratio) metriği kullanılarak değerlendirilir. PSNR değeri ne kadar yüksekse, oluşturulan görüntü orijinal görüntüye o kadar yakındır. 
