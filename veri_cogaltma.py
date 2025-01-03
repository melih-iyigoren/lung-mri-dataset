import os
from PIL import Image, ImageEnhance
import random


def adjust_brightness_contrast(image):
    # Parlaklık ve kontrastı daha düşük aralıkta ayarlama
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.95, 1.05))  # %5 parlaklık artışı/azaltması

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.95, 1.05))  # %5 kontrast artışı/azaltması

    return image


def random_crop(image):
    # Daha az kesme işlemi, küçük değişiklikler
    width, height = image.size
    new_width = random.randint(int(0.95 * width), width)  # Yeni genişlik
    new_height = random.randint(int(0.95 * height), height)  # Yeni yükseklik

    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)

    return image.crop((left, top, left + new_width, top + new_height))


def random_zoom(image):
    # Yakınlaştırma işlemi daha hafif olacak şekilde ayarlandı
    width, height = image.size
    zoom_factor = random.uniform(1.0, 1.01)  # Daha düşük yakınlaştırma faktörü
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    image = image.resize((new_width, new_height), Image.LANCZOS)  # LANCZOS kullanılmalı
    image = image.crop((random.randint(0, new_width - width), random.randint(0, new_height - height),
                        random.randint(0, new_width - width) + width, random.randint(0, new_height - height) + height))
    return image


def rotate_image(image):
    # Küçük döndürme işlemi
    angle = random.uniform(-2, 2)  # -3 ile 3 derece arasında döndürme
    return image.rotate(angle)


def augment_image(image, output_dir, filename):
    # Görseli çoğalt ve kaydet
    for i in range(4):  # 4 farklı görsel üretmek için döngü
        augmented_image = image.copy()
        augmented_image = adjust_brightness_contrast(augmented_image)
        augmented_image = random_crop(augmented_image)
        augmented_image = random_zoom(augmented_image)
        augmented_image = rotate_image(augmented_image)

        # Yeni görselin kaydedileceği yolu belirle
        output_path = os.path.join(output_dir, f"aug_{i+1}_{filename}")
        augmented_image.save(output_path)


def process_images(input_dir, output_dir):
    # Giriş ve çıkış klasörlerini kontrol et
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Giriş klasöründeki tüm görselleri işle
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file_path)

            # Görseli çoğalt
            augment_image(image, output_dir, filename)


# Örnek kullanım
input_dir = 'input_images'  # Orijinal MR görsellerinin bulunduğu klasör
output_dir = 'augmented_images'  # Çoğaltılmış görsellerin kaydedileceği klasör

process_images(input_dir, output_dir)