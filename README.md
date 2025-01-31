# Akciğer MRI Görselleri Web Scraping ve Model Eğitme Projesi

Bu proje, akciğer MR (Manyetik Rezonans) görüntülerini analiz ederek kanser tespiti yapmayı hedefleyen bir makine öğrenimi uygulamasıdır. Model, görselleri kullanarak "sağlıklı" ya da "kanserli" sınıflandırması yapar. Proje, Google Colab üzerinde Python kullanılarak gerçekleştirilmiştir.

## Amaç
Akciğer kanseri, erken teşhisle tedavi edilebilen bir hastalık olmasına rağmen çoğu durumda geç fark edilmektedir. Bu projede geliştirilen model, MR görüntülerine dayalı olarak akciğer kanseri varlığını tahmin etmek için tasarlanmıştır.

## Çalışma Süreci
**Veri Toplama:**

MR görüntüleri, web scraping yöntemleriyle çeşitli kaynaklardan toplanmıştır.
Görseller, "akciğer" ve alakalı anahtar kelimeler ile filtrelenmiştir.

**Veri Ön İşleme:**

Görüntüler yeniden boyutlandırılmış ve normalize edilmiştir.
Veri artırma (augmentation) teknikleri uygulanmıştır (döndürme, parlaklık ayarı vb.).

**Model Eğitimi:**

CNN mimarisinin yetersizliği nedeniyle model, transformatör tabanlı ağa dönüştürülmüştür. Model eğitiminde ViT, Swin Transformer, BEiT ve CvT kullanılmıştır.

**BEiT Model Sonuçları:**

Epoch: 10

En iyi başarım: Accuracy: 1.0000

AUC: 1.00

Precision, Recall, F1-score: 1.00 (Sağlıklı ve Hastalı tespitlerinde %100 başarı)

**Swin Model Sonuçları:**

Epoch: 10

En iyi başarım: Loss: 0.0000

AUC: 1.00

Confusion Matrix: 249 Doğru Pozitif, 305 Doğru Negatif

**CvT Model Sonuçları:**

Epoch: 4

En iyi başarım: Loss: 0.0001

AUC: 1.00

Confusion Matrix: 249 Doğru Pozitif, 304 Doğru Negatif

**ViT Model Sonuçları:**

Epoch: 4

En iyi başarım: Loss: 0.0030

AUC: 1.00

Confusion Matrix: 249 Doğru Pozitif, 304 Doğru Negatif

**Değerlendirme:**

ROC eğrisi ve karmaşıklık matrisi gibi metriklerle model performansı analiz edilmiştir.

## Projede Kullanılar Teknolojiler ve Araçlar

Google Colab: Model geliştirme ve test.
- **Python Kütüphaneleri:** NumPy, scikit-learn, matplotlib, PyTorch.
- **Web Scraping:** Görsellerin otomatik toplanması.
  
## Sonuç ve Gelecek Çalışmalar

Bu model, başlangıç seviyesinde bir performansa sahiptir. Daha büyük ve dengeli bir veri seti ile modelin doğruluğu artırılabilir. Gelecek çalışmalar için:

- Daha Fazla Veri Toplama: MR görüntülerinin çeşitliliği artırılabilir.

- Model Optimizasyonu: Farklı mimariler ve hiperparametre ayarları denenebilir.

Bu çalışma, tıbbi teşhis süreçlerine teknolojik bir katkı sağlamayı amaçlayan önemli bir adımdır.
