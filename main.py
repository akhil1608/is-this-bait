from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

text_classifier = pipeline("text-classification", model="distilbert-base-uncased")

class TextInput(BaseModel):
    text: str

@app.post("/analyze_text")
async def analyze_text(data: TextInput):
    result = text_classifier(data.text)
    return {"label": result[0]["label"], "score": result[0]["score"]}