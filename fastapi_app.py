import base64
import io
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "model.h5"
TARGET_NAMES = None
IMG_H, IMG_W = 0, 0


def train_model():
    """Train the CNN model if weights are not available."""
    global TARGET_NAMES, IMG_H, IMG_W
    try:
        faces = fetch_lfw_people(min_faces_per_person=100)
    except Exception as exc:
        raise RuntimeError("Unable to load LFW dataset. Ensure network access or pre-downloaded data.") from exc
    TARGET_NAMES = faces.target_names
    IMG_H, IMG_W = faces.images.shape[1:3]

    mask = np.zeros(faces.target.shape, dtype=bool)
    for target in np.unique(faces.target):
        mask[np.where(faces.target == target)[0][:150]] = True

    x = faces.data[mask] / 255.0
    y = to_categorical(faces.target[mask])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, stratify=y, random_state=0
    )

    model = Sequential([
        Dense(512, activation="relu", input_shape=(IMG_H * IMG_W,)),
        Dense(len(TARGET_NAMES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10, batch_size=20, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)
    return model


def load_or_train():
    """Load existing model or train a new one."""
    global TARGET_NAMES, IMG_H, IMG_W
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        # fetch dataset meta information for preprocessing
        try:
            faces = fetch_lfw_people(min_faces_per_person=100, download_if_missing=False)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load LFW dataset for preprocessing."
            ) from exc
        TARGET_NAMES = faces.target_names
        IMG_H, IMG_W = faces.images.shape[1:3]
    else:
        model = train_model()
    return model


model = load_or_train()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L").resize((IMG_W, IMG_H))
    arr = np.array(image).astype("float32") / 255.0
    arr = arr.flatten().reshape(1, -1)
    pred = model.predict(arr)
    label = TARGET_NAMES[int(np.argmax(pred))]
    encoded = base64.b64encode(contents).decode()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "label": label, "image": encoded},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
