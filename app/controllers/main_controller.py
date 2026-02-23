"""
controllers/main_controller.py
--------------------------------
Controlador principal (raíz) de la aplicación.

Responsabilidades:
  1. Instanciar los sub-controladores de cada pestaña.
  2. Instanciar la vista principal (MainWindowView).
  3. Instanciar cada vista de pestaña y asociarla a su contenedor y controlador.
  4. Almacenar el CTProcessor activo y distribuirlo entre sub-controladores.
  5. Exponer el método show() para que main.py pueda mostrar la ventana.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from views.main_window_view import MainWindowView
from views.tab_classification_view import TabClassificationView
from views.tab_noise_view import TabNoiseView
from views.tab_metadata_view import TabMetadataView

from controllers.tab_classification_controller import TabClassificationController
from controllers.tab_noise_controller import TabNoiseController
from controllers.tab_metadata_controller import TabMetadataController

if TYPE_CHECKING:
    from logic.preprocessor import CTProcessor


class MainController:
    """
    Controlador raíz de la aplicación MVC.

    Orquesta la creación y vinculación de todas las vistas y controladores.
    También actúa como repositorio del CTProcessor activo, que es compartido
    entre los sub-controladores de Clasificación y Ruido.
    """

    def __init__(self) -> None:
        # ── Estado compartido ──────────────────────────────────────────
        self._ct_processor: CTProcessor | None = None
        self._brain_mask: "np.ndarray | None" = None

        # ── 1. Crear sub-controladores (se inyecta self como referencia al hub) ──
        self._ctrl_classification = TabClassificationController(main_ctrl=self)
        self._ctrl_noise = TabNoiseController(main_ctrl=self)
        self._ctrl_metadata = TabMetadataController()

        # ── 2. Crear la ventana principal ──────────────────────────────
        self._main_window = MainWindowView()

        # ── 3. Crear vistas de cada pestaña y vincularlas ──────────────
        self._setup_classification_tab()
        self._setup_noise_tab()
        self._setup_metadata_tab()

    # ------------------------------------------------------------------
    # API de estado compartido — CTProcessor
    # ------------------------------------------------------------------

    def get_ct_processor(self) -> CTProcessor | None:
        """Retorna el CTProcessor activo, o None si aún no se cargó imagen."""
        return self._ct_processor

    def get_brain_mask(self) -> "np.ndarray | None":
        """Retorna la máscara de cerebro activa (uint8 [0,255]) o None."""
        return self._brain_mask

    def set_brain_mask(self, mask: "np.ndarray") -> None:
        """Almacena la máscara de cerebro generada por BrainExtractor."""
        self._brain_mask = mask

    def notify_brain_mask_ready(self) -> None:
        """Notifica al controlador de ruido que la máscara ya está disponible."""
        self._ctrl_noise.on_brain_mask_ready()

    def set_ct_processor(self, processor: CTProcessor) -> None:
        """
        Almacena un nuevo CTProcessor tras cargar una imagen
        y dispara automáticamente el análisis de ruido.

        Args:
            processor: Instancia de CTProcessor ya inicializada.
        """
        self._ct_processor = processor
        # Auto-disparo: actualiza la pestaña de ruido sin que el usuario pulse nada
        self._ctrl_noise.refresh(processor)

    def refresh_noise(self, processor: CTProcessor) -> None:
        """Delega el refresco de la pestaña de ruido al sub-controlador correspondiente."""
        self._ctrl_noise.refresh(processor)

    # ------------------------------------------------------------------
    # Configuración de pestañas
    # ------------------------------------------------------------------

    def _setup_classification_tab(self) -> None:
        """Instancia la vista de clasificación y la inserta en su contenedor."""
        view = TabClassificationView(self._ctrl_classification)
        self._ctrl_classification.set_view(view)

        container = self._main_window.get_classification_container()
        container.layout().addWidget(view)

    def _setup_noise_tab(self) -> None:
        """Instancia la vista de ruido y la inserta en su contenedor."""
        view = TabNoiseView(self._ctrl_noise)
        self._ctrl_noise.set_view(view)

        container = self._main_window.get_noise_container()
        container.layout().addWidget(view)

    def _setup_metadata_tab(self) -> None:
        """Instancia la vista de metadata y la inserta en su contenedor."""
        view = TabMetadataView(self._ctrl_metadata)
        self._ctrl_metadata.set_view(view)

        container = self._main_window.get_metadata_container()
        container.layout().addWidget(view)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Muestra la ventana principal de la aplicación en tamaño normal."""
        self._main_window.show()

    def show_maximized(self) -> None:
        """Muestra la ventana principal maximizada (pantalla completa)."""
        self._main_window.showMaximized()
