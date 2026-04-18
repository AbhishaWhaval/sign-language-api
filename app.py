from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import joblib
import mediapipe as mp
from transformers import pipeline
import os
import requests

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 Download model if not present
MODEL_PATH = "sign_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=1jdLRf3rgKjPQ7tshChRtxqlAe8kElAWF"
    r = requests.get(url)

    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

    print("Model downloaded!")

# ✅ Load model
model = joblib.load(MODEL_PATH)

# ✅ Lazy sentiment model
sentiment_model = None

def get_sentiment_model():
    global sentiment_model
    if sentiment_model is None:
        sentiment_model = pipeline("sentiment-analysis")
    return sentiment_model

# ✅ Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ✅ Mapping
SIGN_TO_TEXT = {
    "happy": "I am very happy",
    "yes": "I feel good",
    "no": "I am not happy",
    "hello": "Hello",
    "please": "Please"
}

def extract_hand_features(landmarks):
    wrist = landmarks[0]
    features = []
    for lm in landmarks:
        features.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])
    return features

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return {"sign": "Invalid", "confidence": 0}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return {"sign": "No hand", "confidence": 0}

        hand_landmarks = results.multi_hand_landmarks[0]
        features = extract_hand_features(hand_landmarks.landmark)

        if len(features) != 63:
            return {"sign": "Invalid", "confidence": 0}

        X = np.array(features).reshape(1, -1)

        # Prediction
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            idx = np.argmax(probs)
            prediction = model.classes_[idx]
            confidence = float(probs[idx])
        else:
            prediction = model.predict(X)[0]
            confidence = 1.0

        # Lower threshold
        if confidence < 0.3:
            prediction = "UNKNOWN"

        print(f"[DEBUG] {prediction} ({confidence:.2f})")

        text = SIGN_TO_TEXT.get(prediction, "")

        sentiment = "Neutral"
        score = 0.0

        if text:
            model_sent = get_sentiment_model()
            result = model_sent(text)[0]
            sentiment = result["label"]
            score = float(result["score"])

        return {
            "sign": prediction,
            "confidence": confidence,
            "text": text,
            "sentiment": sentiment,
            "score": score
        }

    except Exception as e:
        return {"error": str(e)}