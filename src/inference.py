import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL ----------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(128, 2)
)

model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=device))
model.to(device)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- PREDICT ----------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)[0]

    fake_prob = probs[0].item()
    real_prob = probs[1].item()

    if abs(fake_prob - real_prob) < 0.1:
        label = "Not Sure"
        confidence = max(fake_prob, real_prob)
    elif fake_prob > real_prob:
        label = "Fake"
        confidence = fake_prob
    else:
        label = "Real"
        confidence = real_prob

    # ✅ Console logs
    print("\n===== IMAGE ANALYSIS =====")
    print(f"Image Path   : {image_path}")
    print(f"Label        : {label}")
    print(f"Confidence   : {confidence * 100:.2f}%")
    print(f"Real Prob    : {real_prob * 100:.2f}%")
    print(f"Fake Prob    : {fake_prob * 100:.2f}%")
    print("===========================\n")

    return label, confidence, fake_prob, real_prob