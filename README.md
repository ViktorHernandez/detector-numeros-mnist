# Detector de números, letras y texto manuscrito

Proyecto académico con TensorFlow/Keras, OpenCV y Flask.

## Funciones

- **Números:** conserva el ejercicio MNIST original en `main.py`.
- **Letras:** entrena un clasificador de 26 clases con EMNIST Letters.
- **Palabras y frases:** OpenCV segmenta líneas y caracteres; el modelo clasifica cada carácter y reconstruye el texto.
- **Web:** permite cargar imágenes y utilizar la cámara del PC o del teléfono.
- **Render:** incluye `render.yaml` y Python 3.12.

## Ejecutar el ejercicio original

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

`main.py` se conserva como ejercicio independiente y no es sustituido por la aplicación web.

## Ejecutar la aplicación web

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Abrir `http://127.0.0.1:5000`.

## Entrenamiento de letras

La primera vez que se pulse **Entrenar letras (EMNIST)** se descargará EMNIST Letters mediante TensorFlow Datasets. `importlib-resources` se incluye explícitamente para evitar errores de importación en instalaciones donde esa dependencia no llegue transitivamente.

Si una descarga anterior de EMNIST quedó incompleta y aparece una carpeta sin `dataset_info.json`, el código detecta esa situación, elimina únicamente esa versión incompleta y vuelve a descargarla.

## Texto manuscrito

El reconocimiento de palabras y frases se implementa como un pipeline académico:

```text
imagen → escala de grises → umbral → líneas → caracteres → 28x28 → EMNIST → texto
```

El modelo de letras reconoce A-Z. No pretende sustituir un OCR general; la separación entre caracteres, la calidad de la fotografía y el estilo de escritura influyen en el resultado.

## Render

Build:

```text
pip install -r requirements.txt
```

Start:

```text
gunicorn app:app --workers 1 --threads 2 --timeout 300
```

La aplicación desplegada tendrá HTTPS, lo que permite solicitar acceso a la cámara del teléfono en navegadores compatibles.

## Modelos

- `modelo/mnist_model.keras`: modelo de números.
- `modelo/emnist_letters_model.keras`: se genera después de entrenar letras.

El sistema de archivos normal de Render es efímero. Para el proyecto académico conviene entrenar los modelos localmente y conservar los archivos finales en el repositorio, o usar almacenamiento persistente si se necesita conservar cambios generados en ejecución.

## Archivos de referencia

- `CNN_Numeros.py`: entrenamiento y prueba local del reconocedor de números con MNIST.
- `Reconocedor_Letras.py`: entrenamiento y prueba local del reconocedor de letras con EMNIST y cámara del PC.
- `app.py`: aplicación web para números y texto, compatible con cámara del navegador y despliegue en Render.

