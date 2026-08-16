# Deepfake Video Detection Using EfficientNet-B0

## Project Overview

A deepfake video detection system that classifies videos as **REAL** or **FAKE** using an EfficientNet-B0 based deep learning model.

The system processes video frames and uses a trained deep learning model to determine whether an uploaded video is real or fake.

## Technologies Used

- Python
- PyTorch
- EfficientNet-B0
- OpenCV
- Torchvision
- Streamlit
- timm

## Project Structure

```text
deepfake-video-detection/
│
├── backend/
│   └── train.py
│
├── frontend/
│   └── app.py
│
├── .gitignore
└── README.md
```

## Backend

The backend contains the deep learning model and training code.

The system:

- Loads real and fake videos
- Extracts video frames
- Resizes frames to 224 × 224
- Processes 8 frames from each video
- Uses EfficientNet-B0 for feature extraction
- Uses a custom classifier for binary classification
- Trains the model using PyTorch
- Saves the trained model

## Frontend

The frontend is developed using Streamlit.

Users can:

- Upload an MP4 video
- Preview the uploaded video
- Click the **Analyze Now** button
- Receive a **REAL** or **FAKE** prediction

## Dataset

The model was trained using the **FF++ dataset** containing real and fake video samples.

The dataset is not included in this repository.

## Model

The project uses **EfficientNet-B0** as the feature extraction backbone with a custom classifier.

The model processes a sequence of 8 video frames before classification.

Classification:

- `0` → REAL
- `1` → FAKE

## How to Run

### Install Dependencies

```bash
pip install torch torchvision timm opencv-python streamlit
```

### Run the Application

From the project directory:

```bash
streamlit run frontend/app.py
```

The trained model file is required by the frontend application for making predictions.

## Features

- Deepfake video detection
- REAL/FAKE classification
- MP4 video upload
- Video preview
- EfficientNet-B0 based detection
- 8-frame video processing
- Streamlit web interface

## Author

**Shafiya24**
