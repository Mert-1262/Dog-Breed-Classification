import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(42)

# Veri dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


image_path = "D:/PHYTON/pythonProject30/test/n02086079_186.jpg"
image = Image.open(image_path)
image_transformed = transform(image).unsqueeze(0)  # Modelin beklediği şekle getirin

model_path = "model_weights_118_classes.pth"
num_classes = 8  # Test edeceğiniz sınıf sayısı

alexnet = models.alexnet(pretrained=True)
alexnet.classifier[6] = nn.Linear(4096, 118)  # Modelin orijinal 118 sınıflı ağırlıklarını yüklemek için
alexnet.load_state_dict(torch.load(model_path))  # Eğitilmiş model ağırlıkları yüklenildi.

# Son katmanı 8 sınıf için yeniden tanımlayın
alexnet.classifier[6] = nn.Linear(4096, num_classes)

# Cihazı belirleyin (GPU kullanılabilirse)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet = alexnet.to(device)

# Modeli değerlendirme moduna geçir
alexnet.eval()

with torch.no_grad():
    image_transformed = image_transformed.to(device)
    outputs = alexnet(image_transformed)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

# Manuel sınıf adları
class_names = ['WHİPPET', 'PEKİNESE', 'GOLDEN ROTREİVER', 'DİNGO', 'LHASA', 'CLUMBER', 'BOXER', 'CARTİGAN']  # Bu listeyi kendi sınıf adlarınıza göre güncelleyin


predicted_class = class_names[predicted.item()]
confidence_score = confidence.item() * 100
print(f'KÖPEK TÜRÜ: {predicted_class}')
print(f'KESİNLİK ORANI: {confidence_score:.2f}%')


plt.imshow(image)
plt.title(f'TÜR: {predicted_class}\nAccuracy: {confidence_score:.2f}%')
plt.axis('off')  # Ekseni kapat
plt.show()
