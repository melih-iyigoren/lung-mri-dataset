import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from xml.etree import ElementTree as ET

# Chromedriver yolu
chrome_driver_path = r"C:\Program Files (x86)\chromedriver.exe"

def create_driver():
    """Yeni bir WebDriver oluşturur"""
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--headless")
    service = Service(chrome_driver_path)
    return webdriver.Chrome(service=service, options=options)

def wait_for_page_load(driver, timeout=35):
    """Sayfanın tamamen yüklenmesini bekler"""
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "urlset"))
        )
        print("Sayfa tamamen yüklendi.")
    except Exception as e:
        print(f"Sayfa yüklenirken hata oluştu: {e}")

def download_image(url, filename):
    """Görsel indirme işlemi"""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Görsel indirildi: {filename}")
        else:
            print(f"Görsel indirilemedi: {url}")
    except Exception as e:
        print(f"Görsel indirme hatası: {e}")

def process_sitemap(sitemap_url):
    """Belirtilen sitemap URL'sini işler"""
    global driver
    try:
        driver.get(sitemap_url)
        wait_for_page_load(driver)

        # XML içeriğini al
        sitemap_content = driver.page_source
        root = ET.fromstring(sitemap_content)

        for image in root.findall(".//{*}image"):
            caption = image.find("{*}caption")
            title = image.find("{*}title")
            loc = image.find("{*}loc")

            # "lung" kelimesi içeren görselleri indir
            if loc is not None and (
                (caption is not None and "lung" in caption.text.lower()) or
                (title is not None and "lung" in title.text.lower())
            ):
                image_url = loc.text
                image_filename = os.path.join(download_folder, image_url.split("/")[-1])

                # Görsel zaten mevcutsa indirmeden geç
                if os.path.exists(image_filename):
                    print(f"Görsel zaten mevcut (indirilmedi): {image_filename}")
                else:
                    download_image(image_url, image_filename)
                    time.sleep(0.5)

        print(f"{sitemap_url} işlem tamamlandı.")
    except Exception as e:
        print(f"Hata oluştu: {e}. Tarayıcı yeniden başlatılıyor.")
        driver.quit()
        driver = create_driver()

# İndirme klasörü oluşturma
download_folder = "downloaded_images"
os.makedirs(download_folder, exist_ok=True)

# WebDriver başlatma
driver = create_driver()

# Sitemap URL'leri
sitemap_urls = [
    "https://radiopaedia.org/sitemap-articles_1.xml",
    "https://radiopaedia.org/sitemap-articles_2.xml",
    "https://radiopaedia.org/sitemap-articles_3.xml",
    "https://radiopaedia.org/sitemap-articles_4.xml",
    "https://radiopaedia.org/sitemap-articles_5.xml",
    "https://radiopaedia.org/sitemap-cases_1.xml",
    "https://radiopaedia.org/sitemap-cases_2.xml",
    "https://radiopaedia.org/sitemap-cases_3.xml",
    "https://radiopaedia.org/sitemap-cases_4.xml",
    "https://radiopaedia.org/sitemap-cases_5.xml",
    "https://radiopaedia.org/sitemap-cases_6.xml",
    "https://radiopaedia.org/sitemap-cases_7.xml",
    "https://radiopaedia.org/sitemap-cases_8.xml",
    "https://radiopaedia.org/sitemap-cases_9.xml",
    "https://radiopaedia.org/sitemap-cases_10.xml",
    "https://radiopaedia.org/sitemap-cases_11.xml",
    "https://radiopaedia.org/sitemap-cases_12.xml",
    "https://radiopaedia.org/sitemap-cases_13.xml",
    "https://radiopaedia.org/sitemap-cases_14.xml",
    "https://radiopaedia.org/sitemap-cases_15.xml",
    "https://radiopaedia.org/sitemap-cases_16.xml"
]

# URL'leri işleme alma
for sitemap_url in sitemap_urls:
    print(f"İşlem başlatılıyor: {sitemap_url}")
    process_sitemap(sitemap_url)

driver.quit()
print("Tüm işlemler tamamlandı.")
