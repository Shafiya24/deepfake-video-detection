import streamlit as st
import torch
import cv2
import timm
import torch.nn as nn
from torchvision import transforms

# ================= PAGE =================
st.set_page_config(page_title="Deepfake Detector", layout="wide")

# ================= ADVANCED CSS =================
st.markdown("""
<style>

/* Animated gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

/* Animation */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Hide default header */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Main container */
.main-box {
    max-width: 850px;
    margin: auto;
    margin-top: 50px;
    padding: 40px;
    border-radius: 20px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    box-shadow: 0px 10px 40px rgba(0,0,0,0.6);
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #bbb;
    margin-bottom: 25px;
}

/* Upload box */
section[data-testid="stFileUploader"] {
    border: 2px dashed #888;
    border-radius: 12px;
    padding: 20px;
    background: rgba(255,255,255,0.05);
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #ff6a00, #ee0979);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 12px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #ff6a00;
}

/* Result styles */
.result {
    margin-top: 25px;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    animation: fadeIn 1s ease-in-out;
}

.fake {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
}

.real {
    background: linear-gradient(90deg, #00c851, #007e33);
    color: white;
}

/* Fade animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# ================= UI BOX =================
st.markdown('<div class="main-box">', unsafe_allow_html=True)

st.markdown('<div class="title">Deepfake Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered fake video detection system</div>', unsafe_allow_html=True)

# ================= MODEL =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
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

model = DeepfakeModel()
model.load_state_dict(torch.load("deepfake_model_finalcode.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ================= PREDICTION =================
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < 8:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(224,224))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return "Error"

    while len(frames) < 8:
        frames.append(frames[-1])

    frames = torch.stack(frames).unsqueeze(0).to(device)

    output = model(frames)
    _, pred = torch.max(output,1)

    return "FAKE" if pred.item()==1 else "REAL"

# ================= UI =================
uploaded_file = st.file_uploader("Upload your video", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)

    with open("temp.mp4", "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Analyze Now"):
        result = predict_video("temp.mp4")

        if result == "FAKE":
            st.markdown('<div class="result fake">FAKE VIDEO DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result real">REAL VIDEO</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)