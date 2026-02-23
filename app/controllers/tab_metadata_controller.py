"""
controllers/tab_metadata_controller.py
----------------------------------------
Controlador de la Pestaña 3 – Metadata del Modelo.

Gestiona la lógica asociada a la vista TabMetadataView:
  - Cargar y parsear el archivo de metadata del modelo.
  - (Futuro) Poblar la tabla y el área de texto de la vista con los datos.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from views.tab_metadata_view import TabMetadataView


class TabMetadataController:
    """Controlador para la pestaña de metadata del modelo."""

    def __init__(self) -> None:
        self._view: TabMetadataView | None = None

    def set_view(self, view: "TabMetadataView") -> None:
        """
        Vincula la vista a este controlador.

        Args:
            view: Instancia de TabMetadataView.
        """
        self._view = view

    # ------------------------------------------------------------------
    # Manejadores de eventos (slots)
    # ------------------------------------------------------------------

    def on_load_metadata(self) -> None:
        """
        Se ejecuta al hacer clic en el botón 'Cargar Metadata'.

        TODO: Implementar diálogo de selección de archivo y parsing de metadata.
        """
        print("Seleccionaste el botón [Cargar Metadata]")
