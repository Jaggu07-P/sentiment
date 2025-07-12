from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

MODEL_DIR = "./model"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

app = FastAPI()

class PredictRequest(BaseModel):
    text: str

# Load model and tokenizer
def load_model():
    if os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

nlp = load_model()

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        result = nlp(request.text)[0]
        return {"label": result['label'].lower(), "score": float(result['score'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
