"""
logic/image_adapter.py
-----------------------
Adaptador intermediario: convierte archivos de imagen al formato numpy
que requiere CTProcessor.

Sólo soporta DICOM (.dcm).
Si se intenta cargar un PNG u otro formato, se lanza un ValueError
con un mensaje claro para el usuario.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


class ImageAdapter:
    """
    Adaptador entre archivos de imagen del disco y CTProcessor.

    Uso:
        array = ImageAdapter.from_file("/ruta/imagen.dcm")
        processor = CTProcessor(array)
    """

    @classmethod
    def from_file(cls, file_path: str) -> np.ndarray:
        """
        Carga un archivo de imagen y lo convierte a un numpy array 2D.

        Sólo se admiten archivos DICOM (.dcm).
        Cualquier otro formato lanza ValueError.

        Args:
            file_path: Ruta absoluta al archivo de imagen.

        Returns:
            Array numpy 2D de tipo float32 listo para CTProcessor.

        Raises:
            ValueError: Si el formato no es DICOM.
            RuntimeError: Si la lectura del archivo falla.
        """
        suffix = Path(file_path).suffix.lower()

        if suffix == ".dcm":
            return cls._load_dicom(file_path)

        # Cualquier otro formato (PNG, JPG, etc.) no es compatible con CTProcessor
        raise ValueError(
            f"Formato no compatible: '{suffix}'.\n"
            "CTProcessor sólo acepta archivos DICOM (.dcm).\n"
            "Por favor selecciona un archivo DICOM."
        )

    # ------------------------------------------------------------------
    # Loaders privados
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dicom(file_path: str) -> np.ndarray:
        """
        Lee un archivo DICOM y devuelve su pixel_array como float32 2D.

        Si el dataset tiene múltiples frames (3D), se usa el primer frame.
        Se intenta rescalar a unidades Hounsfield (HU) aplicando
        RescaleSlope y RescaleIntercept cuando estén disponibles.

        Args:
            file_path: Ruta al archivo .dcm.

        Returns:
            Array 2D float32.

        Raises:
            RuntimeError: Si pydicom no está instalado o el archivo es inválido.
        """
        try:
            import pydicom
        except ImportError as exc:
            raise RuntimeError(
                "pydicom no está instalado. Instálalo con:  pip install pydicom"
            ) from exc

        try:
            ds = pydicom.dcmread(file_path)
            pixel_array: np.ndarray = ds.pixel_array.astype(np.float32)

            # Si el volumen es 3D (múltiples frames), tomamos el primer corte
            if pixel_array.ndim == 3:
                pixel_array = pixel_array[0]

            # Aplicar rescalado HU si los atributos DICOM están presentes
            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            pixel_array = pixel_array * slope + intercept

            return pixel_array

        except Exception as exc:
            raise RuntimeError(
                f"No se pudo leer el archivo DICOM:\n{file_path}\n\nDetalle: {exc}"
            ) from exc
