"""
views/tab_metadata_view.py
---------------------------
Vista de la Pestaña 3 – Metadata del Modelo.

Responsabilidades:
  - Botón para cargar los metadatos del modelo.
  - Área de texto/tabla para visualizar los datos del modelo utilizado.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QSplitter,
    QHeaderView,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

if TYPE_CHECKING:
    from controllers.tab_metadata_controller import TabMetadataController


class TabMetadataView(QWidget):
    """
    Vista de metadata del modelo.

    Muestra un botón para cargar el archivo de metadata y una tabla + área
    de texto para inspeccionar los datos del modelo de clasificación.
    """

    def __init__(self, controller: "TabMetadataController") -> None:
        super().__init__()
        self._controller = controller
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Construye y organiza todos los widgets de la pestaña."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # ── Barra de acciones ──────────────────────────────────────────
        action_bar = QHBoxLayout()
        self.btn_load_metadata = QPushButton("📋  Cargar Metadata")
        self.btn_load_metadata.setFixedHeight(36)
        self.btn_load_metadata.setFont(QFont("Arial", 10))
        self.btn_load_metadata.setToolTip(
            "Carga el archivo de metadata del modelo (JSON / YAML / CSV)"
        )
        action_bar.addWidget(self.btn_load_metadata)
        action_bar.addStretch()

        # ── Splitter: tabla superior + texto inferior ──────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Tabla de metadatos
        self.table_metadata = QTableWidget(0, 2)
        self.table_metadata.setHorizontalHeaderLabels(["Campo", "Valor"])
        self.table_metadata.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table_metadata.setAlternatingRowColors(True)
        self.table_metadata.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.table_metadata.setFont(QFont("Consolas", 9))

        # Área de texto con la representación raw de la metadata
        raw_label = QLabel("Representación Raw:")
        raw_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        self.text_raw_metadata = QTextEdit()
        self.text_raw_metadata.setReadOnly(True)
        self.text_raw_metadata.setFont(QFont("Consolas", 9))
        self.text_raw_metadata.setPlaceholderText(
            "Aquí aparecerá el contenido raw del archivo de metadata…"
        )

        raw_container = QWidget()
        raw_layout = QVBoxLayout(raw_container)
        raw_layout.setContentsMargins(0, 0, 0, 0)
        raw_layout.addWidget(raw_label)
        raw_layout.addWidget(self.text_raw_metadata)

        splitter.addWidget(self.table_metadata)
        splitter.addWidget(raw_container)
        splitter.setSizes([300, 200])

        # ── Ensamblado final ───────────────────────────────────────────
        main_layout.addLayout(action_bar)
        main_layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Conexión de señales → controlador
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Conecta cada señal de botón al método correspondiente del controlador."""
        self.btn_load_metadata.clicked.connect(self._controller.on_load_metadata)

    # ------------------------------------------------------------------
    # Métodos públicos de actualización (llamados por el controlador)
    # ------------------------------------------------------------------

    def populate_table(self, data: dict) -> None:
        """
        Rellena la tabla con los pares clave-valor del diccionario de metadata.

        Args:
            data: Diccionario {campo: valor} con los metadatos del modelo.
        """
        self.table_metadata.setRowCount(len(data))
        for row, (key, value) in enumerate(data.items()):
            self.table_metadata.setItem(row, 0, QTableWidgetItem(str(key)))
            self.table_metadata.setItem(row, 1, QTableWidgetItem(str(value)))

    def set_raw_text(self, raw: str) -> None:
        """
        Muestra el contenido raw de la metadata en el área de texto.

        Args:
            raw: Cadena con el contenido del archivo de metadata.
        """
        self.text_raw_metadata.setPlainText(raw)
