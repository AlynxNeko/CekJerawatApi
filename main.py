from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import os
import pandas as pd
import requests
import base64
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "app", "model", "model.h5")
IMAGE_DIR = os.path.join(BASE_DIR, "app", "obat_totol", "extracted_images")

MODEL = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI()

CLASS_NAMES = ['blackheads', 'kistik', 'nodul', 'nodulakistik', 'papula', 'pustula', 'whitehead']
SYMPTOM_KEYS = [
    'komedo_hitam', 'titik_putih', 'berisi_nanah',
    'benjolan_merah', 'benjolan_besar', 'nyeri',
    'merah', 'menyatu', 'tekstur_keras'
]

class RuleInput(BaseModel):
    rules: list[int]

def rule_based_diagnosis(answers):
    scores = dict.fromkeys(CLASS_NAMES, 0.0)

    if answers['komedo_hitam']: scores['blackheads'] += 1.0
    if answers['titik_putih'] and not answers['nyeri']: scores['whitehead'] += 1.0
    if answers['benjolan_merah'] and not answers['berisi_nanah']: scores['papula'] += 1.0
    if answers['berisi_nanah'] and answers['merah']: scores['pustula'] += 1.0
    if answers['benjolan_besar'] and answers['nyeri']:
        scores['nodul'] += 1.0
        scores['kistik'] += 0.5
    if answers['benjolan_besar'] and answers['berisi_nanah']: scores['kistik'] += 1.0
    if answers['menyatu'] and answers['berisi_nanah']: scores['nodulakistik'] += 1.0
    if answers['tekstur_keras']:
        scores['papula'] += 0.3
        scores['nodul'] += 0.2
    if answers['nyeri']:
        scores['kistik'] += 0.2
        scores['nodul'] += 0.2

    arr = np.array([scores[c] for c in CLASS_NAMES])
    prob = arr / arr.sum() if arr.sum() > 0 else np.zeros_like(arr)
    return CLASS_NAMES[np.argmax(prob)], prob

def is_blurry(img, threshold=50.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def crop_to_face(img):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    return img if len(faces) == 0 else img[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]

def combined_prediction(img_array, answers, model, alpha=0.6):
    cnn_pred = model.predict(img_array)[0]
    _, rule_prob = rule_based_diagnosis(answers)
    combined = alpha * cnn_pred + (1 - alpha) * rule_prob
    return CLASS_NAMES[np.argmax(combined)], combined

def recommend_medicine(pred_class):
    sheet_id = "1BBOagNlsbqy_QUVVnksivu0NOtfS9tOCGjWQe4LcZPQ"
    gid = "0"
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    try:
        response = requests.get(csv_url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        return [{"error": f"Gagal ambil data dari Google Sheets: {str(e)}"}]

    df["acne_type_cleaned"] = df["acne_type"].str.lower().fillna("")
    match = df[df["acne_type_cleaned"].str.contains(pred_class.lower())]

    results = []
    for _, row in match.iterrows():
        nama = row.get("obat_totol", "").strip()
        ingredients = row.get("ingredients", "").strip()
        safe_name = nama.replace(" ", "_")
        img_path = os.path.join(IMAGE_DIR, f"{safe_name}.png")

        image_base64 = None
        if os.path.exists(img_path):
            with open(img_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        results.append({
            "nama": nama,
            "ingredients": ingredients,
            "image_base64": image_base64
        })
    return results

from fastapi import Form

@app.post("/predict/")
async def predict(image_file: UploadFile = File(...), rule: str = Form(...)):
    import json
    try:
        rule_data = json.loads(rule)
        if "rules" not in rule_data or len(rule_data["rules"]) != len(SYMPTOM_KEYS):
            return JSONResponse(status_code=422, content={"error": "rules must be a list of 9 binary values."})

        answers = {k: bool(v) for k, v in zip(SYMPTOM_KEYS, rule_data["rules"])}

        img_bytes = await image_file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        cropped = crop_to_face(img)
        if is_blurry(cropped):
            return JSONResponse(status_code=400, content={"error": "Image too blurry."})

        resized = cv2.resize(cropped, (128, 128))
        pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        img_array = image.img_to_array(pil_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        pred_class, scores = combined_prediction(img_array, answers, MODEL)
        recommendations = recommend_medicine(pred_class)

        return {
            "diagnosis": pred_class,
            "probabilities": {cls: float(f"{p:.4f}") for cls, p in zip(CLASS_NAMES, scores)},
            "recommendations": recommendations
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
