"""
logic/RF_logic/prediction_worker.py
--------------------------------------
Worker QThread para ejecutar la sliding-window prediction del Random Forest
en un hilo separado, evitando congelar la UI durante la inferencia.

Señales emitidas:
    started      → La inferencia comenzó.
    progress     → Texto de estado para mostrar en la UI.
    finished     → Emite la máscara de predicción final (np.ndarray uint8).
    error        → Emite el mensaje de error si algo sale mal.
"""

from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Constantes definidas en random_forest.py
from logic.RF_logic.random_forest import (
    predict_image_rf_sliding_window,
    BEST_FEATURES,
    PATCH_SIZE,
    STRIDE,
)

# Ruta al modelo — relativa a este mismo archivo
_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "rf_model.pkl"


def load_rf_model():
    """Carga el modelo RF desde disco con joblib (serialización nativa de sklearn)."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado en: {_MODEL_PATH}")
    return joblib.load(_MODEL_PATH)


class PredictionWorker(QThread):
    """
    Hilo de predicción del modelo Random Forest.

    Uso:
        worker = PredictionWorker(model, dicom_image, brain_mask)
        worker.finished.connect(on_result)
        worker.error.connect(on_error)
        worker.progress.connect(update_status_label)
        worker.start()
    """

    progress: pyqtSignal = pyqtSignal(str)
    finished: pyqtSignal = pyqtSignal(object)   # np.ndarray
    error:    pyqtSignal = pyqtSignal(str)

    def __init__(
        self,
        model,
        dicom_image: np.ndarray,
        brain_mask: np.ndarray,
    ) -> None:
        super().__init__()
        self._model      = model
        self._dicom      = dicom_image
        self._brain_mask = brain_mask

    def run(self) -> None:
        """Ejecutado en el hilo secundario."""
        try:
            self.progress.emit("⏳ Extrayendo características y prediciendo…")
            pred_mask = predict_image_rf_sliding_window(
                self._dicom,
                self._brain_mask,
                self._model,
                BEST_FEATURES,
                PATCH_SIZE,
                STRIDE,
            )
            self.finished.emit(pred_mask)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
