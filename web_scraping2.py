from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
import requests
import os

# WebDriver yolunu belirtin
driver_path = r"C:\Program Files (x86)\chromedriver.exe"  # Kendi yolunuzu girin

# Chrome seçeneklerini belirtmek
options = webdriver.ChromeOptions()

# ChromeDriver servisini başlatmak
service = Service(executable_path=driver_path)

# WebDriver'ı başlatın
driver = webdriver.Chrome(service=service, options=options)

# İndirilecek klasör
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

# URL'leri ve başlamak istediğiniz sayfayı tanımlayın
base_url = "https://openi.nlm.nih.gov/gridquery?q=axial%20normal%20lung&m={}&n={}&it=xg"
current_page_start = 1  # Başlangıçta sayfa 1'den başla

image_counter = 0  # Görsel sayacını başlat

while True:
    # Dinamik URL oluşturuluyor
    current_url = base_url.format(current_page_start, current_page_start + 99)

    # Sayfayı aç
    driver.get(current_url)
    time.sleep(5)  # Sayfanın yüklenmesini bekle

    # Sayfadaki görselleri bul
    images = driver.find_elements(By.CSS_SELECTOR, "img")
    print(f"Sayfa {current_page_start // 100 + 1} - Bulunan görsel sayısı: {len(images)}")

    if len(images) <= 7:  # Eğer 7 veya daha az görsel varsa sayfayı yeniden yükle
        print("Yeterli görsel bulunamadı, sayfa yeniden yükleniyor...")
        driver.refresh()
        time.sleep(5)
        continue  # Yeniden yükleme yapıldığında aynı sayfayı tekrar indir

    # Görselleri indir
    for img in images:
        src = img.get_attribute("src")
        if src:
            try:
                response = requests.get(src, stream=True)
                if response.status_code == 200:
                    image_counter += 1
                    with open(os.path.join(output_folder, f"image_{image_counter}.jpg"), "wb") as file:
                        file.write(response.content)
                    print(f"Görsel {image_counter} indirildi.")
            except Exception as e:
                print(f"Görsel indirilemedi: {e}")

    # Bir sonraki sayfaya geçiş yap
    current_page_start += 100  # Sayfa numarasını artır

    # Bu URL'den sonra son sayfa olduğunu kontrol et
    if current_page_start > 1500:  # 1500'e kadar olan sayfalar
        print("İşlem tamamlandı.")
        break

# Tarayıcıyı kapat
driver.quit()