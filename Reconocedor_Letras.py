from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = Path("modelo/emnist_letters_model.keras")
IMAGE_SIZE = 28
EPOCHS = 5


def fix_orientation(image):
    return np.flipud(np.transpose(image))


def load_dataset():
    import tensorflow_datasets as tfds

    (train, test), _ = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
    )

    train = train.map(lambda image, label: (
        tf.cast(fix_orientation(image), tf.float32) / 255.0,
        label - 1,
    ))
    test = test.map(lambda image, label: (
        tf.cast(fix_orientation(image), tf.float32) / 255.0,
        label - 1,
    ))

    train = train.batch(128).prefetch(tf.data.AUTOTUNE)
    test = test.batch(128).prefetch(tf.data.AUTOTUNE)
    return train, test


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(26, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_character(binary, contour):
    x, y, w, h = cv2.boundingRect(contour)
    margin = max(2, int(max(w, h) * 0.15))
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


def recognize(model, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) >= 8]
    contours.sort(key=lambda c: cv2.boundingRect(c)[0])

    results = []
    for contour in contours:
        prepared = prepare_character(binary, contour)
        probabilities = model.predict(
            prepared.reshape(1, 28, 28), verbose=0
        )[0]
        index = int(np.argmax(probabilities))
        results.append((
            chr(ord("A") + index),
            float(np.max(probabilities) * 100),
        ))
    return results


def train():
    train_data, test_data = load_dataset()
    model = create_model()
    model.fit(train_data, epochs=EPOCHS, verbose=1)
    loss, accuracy = model.evaluate(test_data, verbose=0)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Precisión: {accuracy * 100:.2f}%")
    print(f"Modelo guardado en: {MODEL_PATH}")
    return model


def camera_mode(model):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        height, width = frame.shape[:2]
        size = min(300, height - 40, width - 40)
        x1 = (width - size) // 2
        y1 = (height - size) // 2
        x2, y2 = x1 + size, y1 + size

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, "Escribe una letra | ESPACIO = reconocer | ESC = salir",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )
        cv2.imshow("Reconocedor de letras", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            crop = frame[y1:y2, x1:x2].copy()
            results = recognize(model, crop)
            if results:
                print(f"Letra reconocida: {results[0][0]}")
                print(f"Confianza: {results[0][1]:.2f}%")
            else:
                print("No se detecto ninguna letra.")

    camera.release()
    cv2.destroyAllWindows()


def main():
    model = train()
    camera_mode(model)


if __name__ == "__main__":
    main()
