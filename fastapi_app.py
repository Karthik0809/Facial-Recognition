import base64
import io
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from sklearn.datasets import fetch_lfw_people

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "model.h5"
TARGET_NAMES = None
IMG_H, IMG_W = 0, 0


def crop_face(image: Image.Image) -> Image.Image:
    """Detect and crop the first face in the image if OpenCV is available."""
    if cv2 is None:
        return image
    try:
        cv_img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return image.crop((x, y, x + w, y + h))
    except Exception:
        pass
    return image


def load_model_and_meta():
    """Load pre-trained weights and dataset meta information."""
    global TARGET_NAMES, IMG_H, IMG_W
    try:
        faces = fetch_lfw_people(min_faces_per_person=100, download_if_missing=False)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load LFW dataset for preprocessing."
        ) from exc
    TARGET_NAMES = faces.target_names
    IMG_H, IMG_W = faces.images.shape[1:3]
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Pre-trained weights not found. Run facial_recognition.py to create them."
        )
    return load_model(MODEL_PATH)


model = load_model_and_meta()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = crop_face(image).convert("L").resize((IMG_W, IMG_H))
    arr = np.array(image).astype("float32") / 255.0
    arr = arr.flatten().reshape(1, -1)
    pred = model.predict(arr)
    label = TARGET_NAMES[int(np.argmax(pred))]
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "label": label, "image": encoded},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
