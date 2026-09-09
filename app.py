import base64
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from text_recognition import TEXT_MODEL_PATH, recognize_text_image, train_text_model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MODEL_PATH = Path("modelo/mnist_model.keras")
model = None
text_model = None
training = False
text_training = False
training_lock = threading.Lock()
text_training_lock = threading.Lock()


def load_digit_model():
    global model
    if model is None and MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def create_digit_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def preprocess_digit(image_bytes):
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("No se pudo leer la imagen.")

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) >= 20]
    if not contours:
        raise ValueError("No se encontró un número.")

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    margin = int(max(w, h) * 0.20)
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(binary.shape[1], x + w + margin), min(binary.shape[0], y + h + margin)
    crop = binary[y1:y2, x1:x2]

    ch, cw = crop.shape
    scale = 20.0 / max(ch, cw)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    ox, oy = (28 - nw) // 2, (28 - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas.astype("float32") / 255.0, canvas


def data_url_from_gray(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/status")
def status():
    return jsonify({
        "digit_model": load_digit_model() is not None,
        "text_model": Path(TEXT_MODEL_PATH).exists(),
        "training": training,
        "text_training": text_training,
    })


@app.post("/api/train")
def train_digits():
    global model, training
    if training:
        return jsonify({"ok": False, "message": "El entrenamiento de números ya está en curso."}), 409

    with training_lock:
        training = True
        try:
            print("Cargando el dataset MNIST...", flush=True)
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            print("Dataset MNIST cargado", flush=True)
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0

            model = create_digit_model()
            print("Entrenando la red neuronal...", flush=True)
            model.fit(
                x_train, y_train, epochs=10, validation_split=0.1,
                batch_size=128, verbose=1,
            )
            print("Entrenamiento terminado", flush=True)
            print("Evaluar la red neuronal", flush=True)
            print("Comprobando qué tan bien funciona utilizando imágenes que no vio durante el entrenamiento...", flush=True)
            loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            model.save(MODEL_PATH)
            print(f"Pérdida: {loss:.4f}", flush=True)
            print(f"Precisión: {accuracy * 100:.2f}%", flush=True)
            return jsonify({"ok": True, "accuracy": round(float(accuracy) * 100, 2), "loss": round(float(loss), 4), "message": "Modelo de números entrenado y guardado."})
        except Exception as error:
            return jsonify({"ok": False, "message": str(error)}), 500
        finally:
            training = False


@app.post("/api/train-text")
def train_text():
    global text_model, text_training
    if text_training:
        return jsonify({"ok": False, "message": "El entrenamiento de letras ya está en curso."}), 409

    with text_training_lock:
        text_training = True
        try:
            print("Cargando EMNIST Letters para entrenar el detector de letras...", flush=True)
            accuracy = train_text_model()
            text_model = tf.keras.models.load_model(TEXT_MODEL_PATH)
            return jsonify({"ok": True, "accuracy": round(float(accuracy) * 100, 2), "message": "Modelo de letras entrenado y guardado."})
        except Exception as error:
            return jsonify({"ok": False, "message": str(error)}), 500
        finally:
            text_training = False


@app.post("/api/predict")
def predict_digit():
    digit_model = load_digit_model()
    if digit_model is None:
        return jsonify({"ok": False, "message": "Primero entrena el modelo de números."}), 400
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"ok": False, "message": "No se recibió ninguna imagen."}), 400
    try:
        image, canvas = preprocess_digit(image_file.read())
        probabilities = digit_model.predict(image.reshape(1, 28, 28), verbose=0)[0]
        digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities) * 100)
        return jsonify({"ok": True, "type": "numero", "result": str(digit), "confidence": round(confidence, 2), "processed": data_url_from_gray(canvas)})
    except Exception as error:
        return jsonify({"ok": False, "message": str(error)}), 400


@app.post("/api/predict-text")
def predict_text():
    global text_model
    if text_model is None and Path(TEXT_MODEL_PATH).exists():
        text_model = tf.keras.models.load_model(TEXT_MODEL_PATH)
    if text_model is None:
        return jsonify({"ok": False, "message": "Primero entrena el modelo de letras."}), 400

    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"ok": False, "message": "No se recibió ninguna imagen."}), 400
    try:
        result = recognize_text_image(image_file.read(), text_model)
        return jsonify({"ok": True, **result})
    except Exception as error:
        return jsonify({"ok": False, "message": str(error)}), 400


@app.errorhandler(413)
def too_large(_):
    return jsonify({"ok": False, "message": "La imagen es demasiado grande."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
