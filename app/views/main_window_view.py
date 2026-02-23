"""
views/main_window_view.py
--------------------------
Vista principal de la aplicación.
Contiene el QTabWidget que integra las tres pestañas.
Su responsabilidad es EXCLUSIVAMENTE configurar el layout visual.
"""

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout


class MainWindowView(QMainWindow):
    """Ventana principal: aloja el QTabWidget con las tres pestañas."""

    def __init__(self) -> None:
        super().__init__()
        self._setup_window()
        self._setup_tabs()

    # ------------------------------------------------------------------
    # Configuración de la ventana
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """Configura propiedades básicas de la ventana principal."""
        self.setWindowTitle("Clasificador de Imágenes - MVC")
        self.setMinimumSize(900, 650)

    def _setup_tabs(self) -> None:
        """Crea el QTabWidget y los contenedores de cada pestaña."""
        self.tab_widget = QTabWidget()

        # Contenedores vacíos; el controlador inyectará la vista real de cada pestaña
        self.classification_container = QWidget()
        self.noise_container = QWidget()
        self.metadata_container = QWidget()

        # Se asignan layouts vacíos para que las vistas hijas puedan ser añadidas
        self.classification_container.setLayout(QVBoxLayout())
        self.noise_container.setLayout(QVBoxLayout())
        self.metadata_container.setLayout(QVBoxLayout())

        # Registro de pestañas en el QTabWidget
        self.tab_widget.addTab(self.classification_container, "🖼️  Clasificación")
        self.tab_widget.addTab(self.noise_container, "📊  Ruido")
        self.tab_widget.addTab(self.metadata_container, "📋  Metadata")

        self.setCentralWidget(self.tab_widget)

    # ------------------------------------------------------------------
    # Métodos de acceso (getters) para los contenedores de pestaña
    # ------------------------------------------------------------------

    def get_classification_container(self) -> QWidget:
        """Retorna el contenedor de la pestaña Clasificación."""
        return self.classification_container

    def get_noise_container(self) -> QWidget:
        """Retorna el contenedor de la pestaña Ruido."""
        return self.noise_container

    def get_metadata_container(self) -> QWidget:
        """Retorna el contenedor de la pestaña Metadata."""
        return self.metadata_container
