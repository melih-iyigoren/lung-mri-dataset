import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from transformers import BeitForImageClassification
from google.colab import drive

drive.mount('/content/drive')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

#  Veri yolları
dataset_path = "/content/drive/MyDrive/veriler"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
#  Klasörleri kontrol et
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Hata: Train klasörü bulunamadı: {train_path}")
if not os.path.exists(val_path):
    raise FileNotFoundError(f"Hata: Validation klasörü bulunamadı: {val_path}")

#  Veri ön işleme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#  Dataset yükleme
train_dataset = ImageFolder(root=train_path, transform=transform)
val_dataset = ImageFolder(root=val_path, transform=transform)

#  Dataset boşsa hata ver
if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Hata: Train veya validation dataset boş! Lütfen verileri kontrol et.")

#  DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"✔ Veri yüklendi! {len(train_dataset)} eğitim, {len(val_dataset)} doğrulama örneği var.")

#  Beit modeli yükleme (2 sınıflı)
model = BeitForImageClassification.from_pretrained(
    "microsoft/beit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True  # Boyut uyuşmazlıklarını görmezden gel
)
model.to(device)

#  Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

#  Model Eğitimi
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits  # Model çıktısı
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {train_acc:.4f}")

    # Modeli Kaydetme
    torch.save(model.state_dict(), "beit_model.pth")
    print("Model kaydedildi.")

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

    # Modeli Değerlendirme
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

    # Karmaşıklık Matrisi
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.title("Karmaşıklık Matrisi")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.show()

    # ROC Eğrisi
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

    # Performans Özeti
    print("\n--- Sınıflandırma Raporu ---\n")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))