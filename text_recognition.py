from pathlib import Path
import base64
import os
import shutil

import cv2
import numpy as np
import tensorflow as tf

TEXT_MODEL_PATH = Path("modelo/emnist_letters_model.keras")
IMG_SIZE = 28


def create_text_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(26, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _fix_emnist_orientation(image):
    return np.fliplr(np.rot90(image, 1))


def _get_tfds_data_dir():
    return Path(os.environ.get("TFDS_DATA_DIR", str(Path.home() / "tensorflow_datasets")))


def _prepare_emnist_builder(tfds):
    data_dir = _get_tfds_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    builder = tfds.builder("emnist/letters", data_dir=str(data_dir))
    builder_dir = Path(builder.data_dir)
    dataset_info = builder_dir / "dataset_info.json"
    if builder_dir.exists() and not dataset_info.exists():
        print(f"Detectada una descarga incompleta de EMNIST. Eliminando: {builder_dir}", flush=True)
        shutil.rmtree(builder_dir, ignore_errors=True)
        builder = tfds.builder("emnist/letters", data_dir=str(data_dir))
    return builder


def train_text_model(epochs=8):
    import tensorflow_datasets as tfds

    print("Preparando EMNIST Letters...", flush=True)
    builder = _prepare_emnist_builder(tfds)
    builder.download_and_prepare()
    print("Cargando EMNIST Letters...", flush=True)

    ds_train = builder.as_dataset(split="train", as_supervised=True, batch_size=-1)
    ds_test = builder.as_dataset(split="test", as_supervised=True, batch_size=-1)
    x_train, y_train = tfds.as_numpy(ds_train)
    x_test, y_test = tfds.as_numpy(ds_test)

    x_train = np.asarray([_fix_emnist_orientation(x) for x in x_train], dtype=np.float32) / 255.0
    x_test = np.asarray([_fix_emnist_orientation(x) for x in x_test], dtype=np.float32) / 255.0
    y_train = np.asarray(y_train, dtype=np.int64) - 1
    y_test = np.asarray(y_test, dtype=np.int64) - 1

    model = create_text_model()
    print("Entrenando el modelo de letras...", flush=True)
    model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, batch_size=256, verbose=1)
    print("Evaluando el modelo de letras...", flush=True)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    TEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(TEXT_MODEL_PATH)
    print(f"Modelo de letras guardado en: {TEXT_MODEL_PATH}", flush=True)
    print(f"Pérdida de letras: {loss:.4f}", flush=True)
    print(f"Precisión de letras: {accuracy * 100:.2f}%", flush=True)
    return accuracy


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
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    ox, oy = (IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas.astype(np.float32) / 255.0


def _segment_lines(binary):
    rows = np.where(np.sum(binary > 0, axis=1) > 0)[0]
    if len(rows) == 0:
        return []
    groups = []
    start = prev = int(rows[0])
    for row in rows[1:]:
        row = int(row)
        if row > prev + 2:
            groups.append((start, prev))
            start = row
        prev = row
    groups.append((start, prev))
    return [(max(0, a - 5), min(binary.shape[0], b + 6)) for a, b in groups if b - a >= 3]


def _segment_characters(line):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    line_binary = cv2.morphologyEx(line, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(line_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, _ = line_binary.shape
    items = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 8 or ch < max(6, int(h * 0.15)):
            continue
        items.append((x, y, cw, ch, contour))
    return sorted(items, key=lambda item: item[0])


def _split_words(chars, gap_factor=1.6):
    if not chars:
        return []
    median_w = max(3.0, float(np.median([c[2] for c in chars])))
    words = [[chars[0]]]
    for prev, cur in zip(chars, chars[1:]):
        gap = cur[0] - (prev[0] + prev[2])
        if gap > median_w * gap_factor:
            words.append([cur])
        else:
            words[-1].append(cur)
    return words


def recognize_text_image(image_bytes, model):
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    gray = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("No se pudo leer la imagen.")
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lines = _segment_lines(binary)
    if not lines:
        raise ValueError("No se encontró texto manuscrito.")

    output_lines = []
    character_results = []
    for y1, y2 in lines:
        line = binary[y1:y2, :]
        chars = _segment_characters(line)
        if not chars:
            continue
        line_words = []
        for word_chars in _split_words(chars):
            letters = []
            for x, y, w, h, contour in word_chars:
                canvas = prepare_character(line, contour)
                probs = model.predict(canvas.reshape(1, IMG_SIZE, IMG_SIZE), verbose=0)[0]
                index = int(np.argmax(probs))
                confidence = float(np.max(probs) * 100)
                letter = chr(ord("A") + index)
                letters.append(letter)
                character_results.append({"character": letter, "confidence": round(confidence, 2), "box": [int(x), int(y + y1), int(w), int(h)]})
            line_words.append("".join(letters))
        if line_words:
            output_lines.append(" ".join(line_words))

    text = "\n".join(output_lines).strip()
    if not text:
        raise ValueError("No fue posible separar letras en la imagen.")
    ok, encoded = cv2.imencode(".png", binary)
    processed = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode() if ok else None
    return {
        "type": "texto",
        "text": text,
        "processed": processed,
        "characters": character_results,
        "message": "Texto reconstruido a partir de letras manuscritas. El modelo trabaja con A-Z; la escritura y separación entre caracteres influyen en el resultado.",
    }
