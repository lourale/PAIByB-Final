"""
logic/cnn_logic/cnn_worker.py
-------------------------------
Worker QThread para ejecutar la predicción de clasificación CNN
en un hilo separado, evitando bloquear la UI durante la inferencia.

La CNN predice la clase global de la imagen (Normal / Bleeding / Ischemia)
sin localización de regiones.

Señales emitidas:
    progress (str)              → Texto de estado para mostrar en la UI.
    finished (str, list[float]) → (predicted_class, [prob_normal, prob_bleeding, prob_ischemia])
    error    (str)              → Mensaje de error si algo sale mal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Clases en el mismo orden que el modelo fue entrenado
CNN_CLASS_NAMES = ["Normal", "Bleeding", "Ischemia"]

# Ruta al modelo CNN
_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "cnn_model.pth"


class CnnWorker(QThread):
    """
    Hilo de predicción del modelo CNN.

    Uso:
        worker = CnnWorker(image_path)
        worker.finished.connect(on_result)   # (class_name, list[float])
        worker.error.connect(on_error)       # str
        worker.progress.connect(update_ui)   # str
        worker.start()
    """

    progress: pyqtSignal = pyqtSignal(str)
    finished: pyqtSignal = pyqtSignal(str, list)   # (class_name, probabilities)
    error:    pyqtSignal = pyqtSignal(str)

    def __init__(self, image_path: str) -> None:
        super().__init__()
        self._image_path = image_path

    def run(self) -> None:
        """Ejecutado en el hilo secundario."""
        try:
            if not _MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Modelo CNN no encontrado en: {_MODEL_PATH}"
                )

            self.progress.emit("⏳ Cargando modelo CNN y prediciendo…")

            from logic.cnn_logic.cnn_logic_predict_with_model import predict_with_model

            predicted_class, probabilities_tensor = predict_with_model(
                image_path=self._image_path,
                model_path=str(_MODEL_PATH),
                class_names=CNN_CLASS_NAMES,
            )

            # Convertir tensor de PyTorch a lista Python plana
            probs: list[float] = probabilities_tensor.squeeze().tolist()
            if isinstance(probs, float):
                probs = [probs]

            self.finished.emit(predicted_class, probs)

        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
