from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

from app.model import load_model, predict
from app.schemas import DiseaseResponse

app = FastAPI(title="Smart Krishi Disease Detection API")

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/")
def health():
    return {"status": "ok", "service": "disease-api"}

@app.post("/predict", response_model=DiseaseResponse)
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = predict(image)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
