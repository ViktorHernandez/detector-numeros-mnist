import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = Path("modelo/mnist_model.keras")
EPOCHS = 5


def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, y_train, x_test, y_test


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
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


def prepare_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"No se pudo abrir la imagen: {path}")

    _, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
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
    return canvas.astype("float32") / 255.0


def train():
    x_train, y_train, x_test, y_test = load_dataset()
    model = create_model()
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128, verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Precisión: {accuracy * 100:.2f}%")
    print(f"Modelo guardado en: {MODEL_PATH}")
    return model


def predict_image(model, path):
    image = prepare_image(path)
    probabilities = model.predict(image.reshape(1, 28, 28), verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities) * 100)
    print(f"Número reconocido: {digit}")
    print(f"Confianza: {confidence:.2f}%")
    return digit, confidence


def main():
    model = train()
    path = input("Escribe la ruta de una imagen de un número o presiona Enter para terminar: ").strip()
    if path and os.path.exists(path):
        predict_image(model, path)


if __name__ == "__main__":
    main()
