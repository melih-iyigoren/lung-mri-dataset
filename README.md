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

Eğitim için scikit-learn ve TensorFlow gibi makine öğrenimi kütüphaneleri kullanılmıştır.

**Değerlendirme:**

ROC eğrisi ve karmaşıklık matrisi gibi metriklerle model performansı analiz edilmiştir.

## Modelin performansı

**ROC Eğrisi (AUC Skoru)**

AUC skoru 0.52 olarak hesaplanmıştır. Bu skor, modelin geliştirmeye açık olduğunu göstermektedir.

**Karmaşıklık Matrisi**

Doğru Pozitif: 184

Yanlış Pozitif: 62

Doğru Negatif: 35

Yanlış Negatif: 68

# Projede aşağıdaki teknolojiler ve araçlar kullanılmıştır:

Google Colab: Model geliştirme ve test.
- **Python Kütüphaneleri:** NumPy, scikit-learn, matplotlib, TensorFlow.
- **Web Scraping:** Görsellerin otomatik toplanması.
## Sonuç ve Gelecek Çalışmalar

Bu model, başlangıç seviyesinde bir performansa sahiptir. Daha büyük ve dengeli bir veri seti ile modelin doğruluğu artırılabilir. Gelecek çalışmalar için:

- Daha Fazla Veri Toplama: MR görüntülerinin çeşitliliği artırılabilir.

- Model Optimizasyonu: Farklı mimariler ve hiperparametre ayarları denenebilir.

Bu çalışma, tıbbi teşhis süreçlerine teknolojik bir katkı sağlamayı amaçlayan önemli bir adımdır.
