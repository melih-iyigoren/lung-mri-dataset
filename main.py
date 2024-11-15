import cv2
import os

# Görsellerin bulunduğu klasör ve hedef boyutlar
input_folder = r'C:\Users\melih\PycharmProjects\pythonProject\input'
output_folder = r'C:\Users\melih\PycharmProjects\pythonProject\output'
target_size = 224  # Hedef genişlik ve yükseklik

# Çıktı klasörünü oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Klasördeki her bir görseli işleme
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    if img is not None:
        # Orijinal boyutları al
        h, w = img.shape[:2]

        # Oranı koruyarak yeniden boyutlandırma
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))

        resized_img = cv2.resize(img, (new_w, new_h))

        # Dolgu ekleyerek kare hale getirme
        top = (target_size - new_h) // 2
        bottom = target_size - new_h - top
        left = (target_size - new_w) // 2
        right = target_size - new_w - left

        padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Desteklenmeyen dosya uzantılarıyla başa çıkmak için dosya uzantısını kontrol et
        valid_extensions = ['.jpg', '.jpeg', '.png']
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
        else:
            output_path = os.path.join(output_folder, filename)

        # Boyutlandırılmış görseli çıktı klasörüne kaydet
        try:
            cv2.imwrite(output_path, padded_img)
            print(f"{filename} başarıyla boyutlandırıldı ve dolgu eklendi.")
        except Exception as e:
            print(f"{filename} kaydedilirken hata oluştu: {e}")

    else:
        print(f"{filename} yüklenemedi.")

print("Tüm görseller işleme alındı.")
