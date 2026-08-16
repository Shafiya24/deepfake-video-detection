# ================= INSTALL =================
!pip install timm einops opencv-python

# ================= IMPORTS =================
import os
import cv2
import torch
import timm
import numpy as np
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ================= PATH FIX (YOUR DATASET) =================
#DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/FF++"
DATA_PATH = "/content/dataset/FF++"

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# ================= DATASET =================
class VideoDataset(Dataset):
    def __init__(self, root, seq_len=8):
        self.data = []
        self.seq_len = seq_len

        for label in ["real", "fake"]:
            folder = os.path.join(root, label)

            for video in os.listdir(folder):
                path = os.path.join(folder, video)

                if path.endswith(".mp4"):   # avoid junk files
                    self.data.append((path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        cap = cv2.VideoCapture(path)

        frames = []

        while len(frames) < self.seq_len:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (224,224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)

            frames.append(frame)

        cap.release()

        # ⚠️ FIX: handle empty video
        if len(frames) == 0:
            frames = [torch.zeros(3,224,224) for _ in range(self.seq_len)]

        # pad frames
        while len(frames) < self.seq_len:
            frames.append(frames[-1])

        frames = torch.stack(frames)

        label = 1 if label=="fake" else 0

        return frames, label

# ================= LOAD DATA =================
dataset = VideoDataset(DATA_PATH, seq_len=8)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

print("Total Videos:", len(dataset))

# ================= MODEL =================
class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)

        features = self.backbone(x)
        features = features.view(B, T, -1)

        features = features.mean(dim=1)

        return self.classifier(features)

# ================= INIT =================
model = DeepfakeModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ================= TRAIN =================
epochs = 50

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for videos, labels in loader:

        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    # ✅ SAVE EVERY EPOCH
    torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

# ================= FINAL SAVE =================
torch.save(model.state_dict(), "deepfake_model_final.pth")
print("Model saved successfully")

# ================= ACCURACY =================
correct = 0
total = 0

model.eval()

with torch.no_grad():
    for videos, labels in loader:

        videos = videos.to(device)

        outputs = model(videos)
        _, pred = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (pred.cpu() == labels).sum().item()

accuracy = 100 * correct / total
print("Final Accuracy:", accuracy, "%")

# ================= PREDICT =================
def predict_video(video_path, seq_len=8):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while len(frames) < seq_len:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224,224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print("Error: No frames read")
        return

    while len(frames) < seq_len:
        frames.append(frames[-1])

    frames = torch.stack(frames).unsqueeze(0).to(device)

    model.eval()
    output = model(frames)

    _, pred = torch.max(output, 1)

    print("FAKE VIDEO" if pred.item()==1 else "REAL VIDEO")

