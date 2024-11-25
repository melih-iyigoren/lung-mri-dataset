# Akciğer MRI Görselleri Web Scraping Projesi

Bu proje, Python ve Selenium kullanılarak internetten akciğer MRI görselleri toplamayı amaçlamaktadır. Proje, makine öğrenimi modelleri için gerekli olan görüntü veri setini oluşturmayı hedefler.

## Proje Özeti

- **Amaç:** Akciğer kanseri teşhisinde kullanılmak üzere MRI görsellerini internetten otomatik olarak indirerek bir veri seti oluşturmak.
- **Araçlar:** Python, Selenium, Google Chrome ve Radiopaedia web sitesi.
- **Sonuç:** Akciğer görselleri başarıyla indirildi ve bir model eğitmek üzere hazırlandı.

---

## Geliştirme Ortamı

- **Programlama Dili:** Python 3.8+
- **Kütüphaneler:** 
  - Selenium
  - Requests (isteğe bağlı)
  - os ve time gibi Python yerleşik modülleri
- **Gereksinimler:** 
  - Google Chrome tarayıcısı
  - Chromedriver (Proje için `C:\Program Files (x86)\chromedriver.exe` yolunda konumlandırıldı)
