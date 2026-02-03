from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

MODEL_NAME = "wambugu71/crop_leaf_diseases_vit"

# These are loaded ONCE when the server starts
processor = None
model = None

def load_model():
    global processor, model
    if model is None or processor is None:
        print("[INFO] Loading disease detection model...")
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        model.eval()
        print("[INFO] Model loaded successfully")

def predict(image):
    if model is None or processor is None:
        raise RuntimeError("Model not loaded")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()

    return model.config.id2label[predicted_class]
