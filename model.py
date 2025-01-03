import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Veri Yolu
dataset_path = "/Users/talhasari/PycharmProjects/modelegitim/.venv/lib/veriler"

# Hiperparametreler
image_size = (128, 128)
batch_size = 32
epochs = 10

# 1. Veri Hazırlama
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # %20 doğrulama seti
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# 2. Model Oluşturma
model = Sequential([
    tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # İkili sınıflandırma
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Model Eğitimi
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    verbose=1
)

# 4. Model Değerlendirme
validation_generator.reset()
predictions = model.predict(validation_generator, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# 5. Karmaşıklık Matrisi
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Karmaşıklık Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

# 6. ROC Eğrisi
fpr, tpr, _ = roc_curve(true_classes, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Eğrisi (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
plt.title("Receiver Operating Characteristic (ROC) Eğrisi")
plt.legend(loc="lower right")
plt.show()

# 7. Performans Özeti
print("\n--- Sınıflandırma Raporu ---\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))