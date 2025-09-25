# MERT IYIBICER
# 21100011070
#DERİN OGRENME-CNN İLE KÖPEK CİNS TESPİTİ

# 2565 adet gorsel mevcut - 10 CİNS(Baslangıc olarak)
# 20560 adet gorsel mevcut - 118 CİNS
import itertools

import cv2
import os

image_directory = "D:/PHYTON/pythonProject30"

for filename in os.listdir(image_directory):  # Dizinin içindeki tüm dosya ve klasörleri listeler

    file_path = os.path.join(image_directory, filename)

    if filename.endswith(".jpg") or filename.endswith(
            ".png"):  # Dosya uzantısı kontrolü yapar --- sadece görüntü dosyalarının işlenmesi için

        image = cv2.imread(file_path)

        # Görüntü adını yazdır
        print("Görüntü:", filename)



'''
# GORUNTU_BOYUTU_AYARLAMA_KODLARI--ORNEK OLARAK KALMASI ANLAMINDA 2 CİNS TURUNUN BOYUTLANDIRILMASI KOD UZERINDE BIRAKILDI.
import cv2
import os

# Görüntülerin bulunduğu klasör
folder_path = "D:/PHYTON/pythonProject30/test/whippet_test"

# Yeni boyutlar
new_width = 224
new_height = 224

# Klasördeki tüm dosyaları al
for filename in os.listdir(folder_path):
    # Dosya yolunu oluştur
    filepath = os.path.join(folder_path, filename)

    # Dosyanın bir resim dosyası olup olmadığını kontrol et
    if os.path.isfile(filepath) and filename.endswith(('.png', '.jpg', '.jpeg')):
        # Resmi aç
        img = cv2.imread(filepath)

        # Boyutlandır
        resized_img = cv2.resize(img, (new_width, new_height))

        # Yeniden boyutlandırılmış görüntüyü üzerine yaz
        cv2.imwrite(filepath, resized_img)

print("Tüm görüntüler başarıyla boyutlandırıldı.")

# Görüntülerin bulunduğu klasör
folder_path2 = "D:/PHYTON/pythonProject30/n02085620-Chihuahua"

# Yeni boyutlar
new_width = 224
new_height = 224

# Klasördeki tüm dosyaları aldı
for filename in os.listdir(folder_path2):
    # Dosya yolunu oluştur
    filepath = os.path.join(folder_path2, filename)

    # Dosyanın bir resim dosyası olup olmadığını kontrol etti
    if os.path.isfile(filepath) and filename.endswith(('.png', '.jpg', '.jpeg')):
        # Resmi aç
        img = cv2.imread(filepath)

        # Boyutlandırıldı
        resized_img = cv2.resize(img, (new_width, new_height))

        # Yeniden boyutlandırılmış görüntüyü tekrar aynı görüntünün üzerine yazdı
        cv2.imwrite(filepath, resized_img)


# Görüntülerin bulunduğu klasörün yolu :

folder_path2 = "D:/PHYTON/pythonProject30/n02106382-Bouvier_des_Flandres"

# Yeni boyutların degerleri :
new_width = 224
new_height = 224

# Klasördeki tüm dosyaları al
for filename in os.listdir(folder_path2):
    # Dosya yolunu oluştur
    filepath = os.path.join(folder_path2, filename)

    # Dosyanın bir resim dosyası olup olmadığı kontrol edildi
    if os.path.isfile(filepath) and filename.endswith(('.png', '.jpg', '.jpeg')):
        # Resmi aç
        img = cv2.imread(filepath)

        # Boyutlandırıldı
        resized_img = cv2.resize(img, (new_width, new_height))

        # Yeniden boyutlandırılmış görüntü aynı görüntünün üzerine yazıldı.
        cv2.imwrite(filepath, resized_img)


#ALEXNET MODELİ RGB İLE ÇALIŞABİLDİĞİ İÇİN VERİLERDE RENK DÜZENLEMESİNE İHTİYAC DUYULMAMISTIR.
print("Tüm görüntüler başarıyla boyutlandırıldı.")


'''


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# Veri setinin bulunduğu dizinleri belirtin ve işleyin
train_data_path = "D:/PHYTON/pythonProject30/Images"
test_data_path = "D:/PHYTON/pythonProject30/test"

# Veri dönüşümleri tanımlayın
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resimleri AlexNet giriş boyutuna yeniden boyutlandırın
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #RAPORA DETAYINI YAZ.
])

# Veri setlerini yükleyin
train_dataset = datasets.ImageFolder(train_data_path, transform=transform)
test_dataset = datasets.ImageFolder(test_data_path, transform=transform)

# Veri yükleyicilerini tanımlayın
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modeli yükleyin
alexnet = models.alexnet(pretrained=True)
num_classes = 118
alexnet.classifier[6] = nn.Linear(4096, num_classes)

#CNN KATMANLARI ALEXNET MODELİ İÇERİSİNDE AYARLANILIP KODA O ŞEKİLDE ENTEGRE EDİLMİŞTİR.

# Cihazı belirleyin (GPU kullanılabilirse)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet = alexnet.to(device)

# Kayıp fonksiyonunu ve optimizasyon algoritmasını tanımlayın
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)

# Doğruluk ve kayıp değerlerini saklamak için listeler oluşturun
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


# Eğitim ve test için yardımcı fonksiyon
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# Eğitim ve test sırasında doğruluk ve kayıp değerlerini saklamak için listeler oluşturulmuştur.
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


# Eğitim döngüsü
num_epochs = 13
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = alexnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Her epoch sonunda eğitim ve test veri setleri üzerinde modelin doğruluğunu ve kaybını değerlendirin
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    test_loss, test_accuracy = evaluate_model(alexnet, test_loader, criterion, device)

    # Değerleri listelere kaydedin
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # Her 1 epoch sonunda bilgi yazdır
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Eğitim ve test accuracy ile loss grafiğini çizin
plt.figure(figsize=(12, 5))

# Train ve test loss grafiği
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses, label='Test Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

# Train ve test accuracy grafiği
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(test_accuracies, label='Test Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Classification report'u oluşturun
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Modelin test veri seti üzerinde tahminlerini alınmıştır bunun için liste tutulmuştur.
predicted_labels = []
true_labels = []

# Modeli değerlendirme modunda ayarlanmıştır.
alexnet.eval()

# Test veri seti üzerinde modelin performansını değerlendirilmiştir.
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = alexnet(images)
        _, predicted = torch.max(outputs, 1)
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Classification report'u oluşturun
class_names = test_dataset.classes
report = classification_report(true_labels, predicted_labels, target_names=class_names)
print(report)

# Confusion matrix hesaplanılmıştır.
cm = confusion_matrix(true_labels, predicted_labels)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalization

# Confusion matrix grafiğini çizilmiştir.
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Normalized Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

fmt = '.2f'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Eğitilmiş modelin ağırlıkları kaydededilmistir.
model_path = "D:/PHYTON/pythonProject30"  # Kaydedilecek dosya yolunu belirleyelim.
torch.save(alexnet.state_dict(), 'model_weights_118_classes.pth')


loaded_model = models.alexnet(pretrained=False)  # Önceden eğitilmiş modeli yükleyelim.
loaded_model.classifier[6] = nn.Linear(4096, num_classes)  # Sınıflandırıcı katmanını yeniden tanımlayalım.
loaded_model.load_state_dict(torch.load(model_weights_118_classes.pth))  # Eğitilmiş ağırlıkları yükleyelim.
