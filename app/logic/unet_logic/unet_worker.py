"""
logic/unet_logic/unet_worker.py
----------------------------------
Worker QThread para ejecutar la segmentación U-NET en background.

Según la clase predicha por la CNN escoge el modelo correcto:
  - Bleeding → unet_bleeding.keras
  - Ischemia → unet_ischemic.keras

La U-NET espera un PNG grayscale como entrada, así que la imagen CT
(numpy float32, HU) se convierte a PNG en un archivo temporal, se llama
a la función de segmentación y luego se borra el temporal.

Señales emitidas:
    progress (str)       → Texto de estado para la UI.
    finished (np.ndarray)→ Máscara binaria float32 (H, W) con valores 0/1.
    error    (str)       → Mensaje de error si algo sale mal.
"""

from __future__ import annotations

import tempfile
import os
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

_MODELS_DIR = Path(__file__).parent.parent.parent / "models"
_BLEEDING_MODEL = _MODELS_DIR / "unet_bleeding.keras"
_ISCHEMIA_MODEL = _MODELS_DIR / "unet_ischemic.keras"


class UnetWorker(QThread):
    """
    Hilo de segmentación U-NET.

    Uso:
        worker = UnetWorker(image_array, predicted_class)
        worker.finished.connect(on_mask)    # np.ndarray float32
        worker.error.connect(on_error)      # str
        worker.progress.connect(update_ui)  # str
        worker.start()
    """

    progress: pyqtSignal = pyqtSignal(str)
    finished: pyqtSignal = pyqtSignal(object)   # np.ndarray float32
    error:    pyqtSignal = pyqtSignal(str)

    def __init__(
        self,
        image_array: np.ndarray,
        predicted_class: str,
    ) -> None:
        """
        Args:
            image_array:     Array 2D float32 en HU (clean_image del CTProcessor).
            predicted_class: 'Bleeding' o 'Ischemia' (elige el modelo correcto).
        """
        super().__init__()
        self._image = image_array
        self._predicted_class = predicted_class

    def run(self) -> None:
        """Ejecutado en el hilo secundario."""
        tmp_path: str | None = None
        try:
            # ── Seleccionar modelo ─────────────────────────────────────────
            if self._predicted_class == "Bleeding":
                model_path = str(_BLEEDING_MODEL)
                if not _BLEEDING_MODEL.exists():
                    raise FileNotFoundError(
                        f"Modelo Bleeding no encontrado: {_BLEEDING_MODEL}"
                    )
                segment_fn_name = "segment_bleeding_with_unet"
            elif self._predicted_class == "Ischemia":
                model_path = str(_ISCHEMIA_MODEL)
                if not _ISCHEMIA_MODEL.exists():
                    raise FileNotFoundError(
                        f"Modelo Ischemia no encontrado: {_ISCHEMIA_MODEL}"
                    )
                segment_fn_name = "segment_ischemic_with_unet"
            else:
                raise ValueError(
                    f"Clase '{self._predicted_class}' no tiene modelo U-NET."
                )

            self.progress.emit(
                f"⏳ Segmentando con U-NET ({self._predicted_class})…"
            )

            # ── Convertir CT array→ PNG temporal ──────────────────────────
            # La U-NET lee un PNG grayscale [0,1].
            # Usamos ventana cerebral 0-80 HU como en la vista.
            import imageio.v3 as iio

            arr = np.clip(self._image, 0, 80).astype(np.float32)
            span = 80.0
            arr_u8 = (arr / span * 255).clip(0, 255).astype(np.uint8)

            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            iio.imwrite(tmp_path, arr_u8)

            # ── Llamar a la función de segmentación ───────────────────────
            from logic.unet_logic.segment_with_u_net import (
                segment_bleeding_with_unet,
                segment_ischemic_with_unet,
            )

            if self._predicted_class == "Bleeding":
                mask = segment_bleeding_with_unet(tmp_path, model_path)
            else:
                mask = segment_ischemic_with_unet(tmp_path, model_path)

            self.finished.emit(mask)

        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
