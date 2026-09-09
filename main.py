import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


MODELO_PATH = Path("modelo/mnist_model.keras")
EPOCHS = 10


def cargar_dataset():
    print("Cargando el dataset MNIST...")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("Dataset MNIST cargado")
    print(f"Imágenes de entrenamiento: {x_train.shape[0]}")
    print(f"Imágenes de prueba: {x_test.shape[0]}")
    print("Dimensión de las imágenes: 28x28 píxeles")

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return x_train, y_train, x_test, y_test


def crear_red_neuronal():
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


def entrenar(modelo, x_train, y_train):
    print("\nEntrenando la red neuronal...")

    modelo.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        validation_split=0.1,
        batch_size=128,
        verbose=1
    )

    print("Entrenamiento terminado")


def evaluar(modelo, x_test, y_test):
    print("\nEvaluar la red neuronal")
    print("Comprobando qué tan bien funciona utilizando imágenes que no vio durante el entrenamiento...")

    resultado = modelo.evaluate(x_test, y_test, verbose=0)

    print(f"Pérdida: {resultado[0]:.4f}")
    print(f"Precisión: {resultado[1] * 100:.2f}%")

    return resultado


def guardar_modelo(modelo):
    MODELO_PATH.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(MODELO_PATH)
    print(f"\nModelo guardado en: {MODELO_PATH}")


def mostrar_predicciones(modelo, x_test, y_test, cantidad=10):
    print("\nPredicciones sobre imágenes que el modelo no vio durante el entrenamiento:")

    predicciones = modelo.predict(x_test[:cantidad], verbose=0)

    for indice, probabilidades in enumerate(predicciones):
        numero_predicho = int(np.argmax(probabilidades))
        confianza = float(np.max(probabilidades) * 100)
        numero_real = int(y_test[indice])

        print(
            f"Imagen {indice + 1}: "
            f"esperado={numero_real}, "
            f"predicho={numero_predicho}, "
            f"confianza={confianza:.2f}%"
        )


def preparar_imagen_opencv(ruta):
    imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        raise ValueError(f"No se pudo abrir la imagen: {ruta}")

    _, imagen = cv2.threshold(
        imagen,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contornos, _ = cv2.findContours(
        imagen,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        raise ValueError("No se encontró un número en la imagen.")

    contorno = max(contornos, key=cv2.contourArea)
    x, y, ancho, alto = cv2.boundingRect(contorno)

    margen = int(max(ancho, alto) * 0.20)

    x1 = max(0, x - margen)
    y1 = max(0, y - margen)
    x2 = min(imagen.shape[1], x + ancho + margen)
    y2 = min(imagen.shape[0], y + alto + margen)

    recorte = imagen[y1:y2, x1:x2]

    alto_recorte, ancho_recorte = recorte.shape
    escala = 20.0 / max(alto_recorte, ancho_recorte)

    nuevo_ancho = max(1, int(ancho_recorte * escala))
    nuevo_alto = max(1, int(alto_recorte * escala))

    redimensionada = cv2.resize(
        recorte,
        (nuevo_ancho, nuevo_alto),
        interpolation=cv2.INTER_AREA
    )

    imagen_28 = np.zeros((28, 28), dtype=np.uint8)

    offset_x = (28 - nuevo_ancho) // 2
    offset_y = (28 - nuevo_alto) // 2

    imagen_28[
        offset_y:offset_y + nuevo_alto,
        offset_x:offset_x + nuevo_ancho
    ] = redimensionada

    return imagen_28.astype("float32") / 255.0


def probar_imagen_propia(modelo):
    ruta = input(
        "\nEscribe la ruta de una imagen de un número manuscrito "
        "para probarla con OpenCV (Enter para omitir): "
    ).strip()

    if not ruta:
        return

    if not os.path.exists(ruta):
        print("La imagen no existe.")
        return

    try:
        imagen = preparar_imagen_opencv(ruta)
        entrada = imagen.reshape(1, 28, 28)

        probabilidades = modelo.predict(entrada, verbose=0)[0]
        numero = int(np.argmax(probabilidades))
        confianza = float(np.max(probabilidades) * 100)

        print("\nResultado de la imagen propia")
        print(f"Número detectado: {numero}")
        print(f"Confianza: {confianza:.2f}%")

    except Exception as error:
        print(f"No se pudo procesar la imagen: {error}")


def main():
    print("=" * 60)
    print("DETECTOR DE NÚMEROS MANUSCRITOS - MNIST")
    print("TensorFlow + Keras + OpenCV")
    print("=" * 60)

    x_train, y_train, x_test, y_test = cargar_dataset()

    modelo = crear_red_neuronal()

    print("\nArquitectura de la red neuronal:")
    modelo.summary()

    entrenar(modelo, x_train, y_train)

    evaluar(modelo, x_test, y_test)

    guardar_modelo(modelo)

    mostrar_predicciones(modelo, x_test, y_test)

    probar_imagen_propia(modelo)

    print("\nProceso terminado.")


if __name__ == "__main__":
    main()
