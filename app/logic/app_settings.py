"""
logic/app_settings.py
-----------------------
Persistencia liviana de configuración de usuario entre sesiones.

Almacena y recupera el último directorio de apertura de archivos en un
archivo JSON ubicado junto al main.py (raíz de la aplicación).
"""

from __future__ import annotations
import json
from pathlib import Path

# Ruta del archivo de configuración: app/app_settings.json
_SETTINGS_FILE = Path(__file__).parent.parent / "app_settings.json"


class AppSettings:
    """Acceso estático a la configuración persistente de la aplicación."""

    @staticmethod
    def get_last_dir() -> str:
        """
        Retorna el último directorio usado al cargar una imagen.
        Si no existe entrada previa, retorna cadena vacía (QFileDialog usará CWD).
        """
        try:
            data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
            return data.get("last_dir", "")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return ""

    @staticmethod
    def save_last_dir(file_path: str) -> None:
        """
        Guarda el directorio del archivo seleccionado para la próxima sesión.

        Args:
            file_path: Ruta completa al archivo (se guarda su directorio padre).
        """
        try:
            directory = str(Path(file_path).parent)
            data: dict = {}
            if _SETTINGS_FILE.exists():
                try:
                    data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    data = {}
            data["last_dir"] = directory
            _SETTINGS_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"  [AVISO] No se pudo guardar la configuración: {exc}")

    # ── Directorio de máscara de referencia (comparación U-NET) ────────

    @staticmethod
    def get_last_mask_dir() -> str:
        """
        Retorna el último directorio usado al cargar una máscara de referencia.
        Si no existe entrada previa, cae al último directorio de imagen.
        """
        try:
            data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
            return data.get("last_mask_dir", data.get("last_dir", ""))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return ""

    @staticmethod
    def save_last_mask_dir(file_path: str) -> None:
        """
        Guarda el directorio de la máscara seleccionada para la próxima sesión.

        Args:
            file_path: Ruta completa al archivo de máscara.
        """
        try:
            directory = str(Path(file_path).parent)
            data: dict = {}
            if _SETTINGS_FILE.exists():
                try:
                    data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    data = {}
            data["last_mask_dir"] = directory
            _SETTINGS_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"  [AVISO] No se pudo guardar configuración de máscara: {exc}")
