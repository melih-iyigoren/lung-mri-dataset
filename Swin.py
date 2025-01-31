#Gerekli Kütüphanelerin İçe Aktarılması
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import SwinForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#Veri Yolları
dataset_path = "/content/drive/MyDrive/veriler"
train_path = f"{dataset_path}/train"
val_path = f"{dataset_path}/val"

#Veri Ön İşleme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizasyon
])

#Dataset Yükleme
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

#DataLoader (Verileri Batch Halinde Yükleme)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#Modeli Yükleme (Hugging Face)
num_classes = 2  # İki sınıf
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True  # Boyut uyuşmazlıklarını yok say
)
model.to(device)

#Eğitim Parametreleri
epochs = 10
learning_rate = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

#Modeli Eğitme
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Sıfırlama, İleri Geçiş, Geriye Yayılma ve Optimizasyon
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Eğitim sonrası loss çıktısı
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

#Modeli Değerlendirme
model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        probs = torch.nn.functional.softmax(outputs, dim=1)  # Softmax ile olasılıkları hesapla
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

#Karmaşıklık Matrisi
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.title("Karmaşıklık Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

#ROC Eğrisi
all_probs = np.array(all_probs)
fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])  # Sadece pozitif sınıfın olasılıklarını kullan
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Eğrisi (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Eğrisi")
plt.legend(loc="lower right")
plt.show()

#Performans Özeti
print("\n--- Sınıflandırma Raporu ---\n")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

#Modeli Kaydetme
torch.save(model.state_dict(), "/content/drive/MyDrive/beit_model.pth")
print("Model kaydedildi.")