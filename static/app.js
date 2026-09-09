const $ = (id) => document.getElementById(id);

let stream = null;
let selectedFile = null;

async function refreshStatus() {
  const r = await fetch("/api/status");
  const s = await r.json();
  $("trainingStatus").textContent =
    `Números: ${s.digit_model ? "listo" : "sin entrenar"} · ` +
    `Letras: ${s.text_model ? "listo" : "sin entrenar"}`;
}

$("openCamera").onclick = async () => {
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("El navegador no permite acceso a cámara en este contexto.");
    }
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    $("video").srcObject = stream;
    $("capture").disabled = false;
    $("closeCamera").disabled = false;
    $("message").textContent = "Cámara activa.";
  } catch (e) {
    $("message").textContent = "No se pudo abrir la cámara: " + e.message;
  }
};

$("closeCamera").onclick = () => {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  $("video").srcObject = null;
  $("capture").disabled = true;
  $("closeCamera").disabled = true;
};

$("capture").onclick = () => {
  const video = $("video");
  if (!video.videoWidth || !video.videoHeight) {
    $("message").textContent = "La cámara todavía no está lista.";
    return;
  }
  const canvas = $("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    if (!blob) return;
    selectedFile = new File([blob], "captura.jpg", { type: "image/jpeg" });
    $("preview").src = URL.createObjectURL(selectedFile);
    $("message").textContent = "Foto capturada.";
  }, "image/jpeg", 0.92);
};

$("fileInput").onchange = (event) => {
  selectedFile = event.target.files[0] || null;
  if (selectedFile) {
    $("preview").src = URL.createObjectURL(selectedFile);
    $("message").textContent = "Imagen cargada.";
  }
};

async function train(url, button) {
  button.disabled = true;
  $("message").textContent = "Entrenamiento iniciado. Puede tardar varios minutos.";
  try {
    const r = await fetch(url, { method: "POST" });
    const data = await r.json();
    if (!r.ok || !data.ok) throw new Error(data.message || `HTTP ${r.status}`);
    $("message").textContent = `${data.message} Precisión: ${data.accuracy}%`;
    await refreshStatus();
  } catch (e) {
    $("message").textContent = "Error de entrenamiento: " + e.message;
  } finally {
    button.disabled = false;
  }
}

$("trainDigits").onclick = () => train("/api/train", $("trainDigits"));
$("trainText").onclick = () => train("/api/train-text", $("trainText"));

$("analyze").onclick = async () => {
  if (!selectedFile) {
    $("message").textContent = "Primero toma una foto o carga una imagen.";
    return;
  }

  const mode = document.querySelector('input[name="mode"]:checked').value;
  const form = new FormData();
  form.append("image", selectedFile);
  $("analyze").disabled = true;
  $("message").textContent = "Procesando...";

  try {
    const url = mode === "number" ? "/api/predict" : "/api/predict-text";
    const r = await fetch(url, { method: "POST", body: form });
    const data = await r.json();
    if (!r.ok || !data.ok) throw new Error(data.message || `HTTP ${r.status}`);

    if (mode === "number") {
      $("digitResult").textContent = data.result;
      $("confidence").textContent = `Confianza: ${data.confidence}%`;
      $("textResult").textContent = "";
    } else {
      $("digitResult").textContent = "Texto detectado";
      $("confidence").textContent = data.message || "";
      $("textResult").textContent = data.text;
    }
    $("processed").src = data.processed || "";
    $("message").textContent = "Análisis terminado.";
  } catch (e) {
    $("message").textContent = "Error: " + e.message;
  } finally {
    $("analyze").disabled = false;
  }
};

refreshStatus().catch((e) => {
  $("trainingStatus").textContent = "No se pudo consultar el estado: " + e.message;
});
