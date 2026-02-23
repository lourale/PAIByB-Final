"""
views/tab_classification_view.py
---------------------------------
Vista de la Pestaña 1 – Clasificación de Imágenes.

Responsabilidades:
  - Botón "Cargar Imagen" (DICOM).
  - Botón "Evaluar" para disparar la inferencia del modelo.
  - Selector de vista: "Imagen Completa" / "Cerebro Extraído".
  - Controles opcionales de ventana HU.
  - Área visual para la imagen (original o cerebro).
  - Área visual para el resultado de clasificación del modelo de IA.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
    QSizePolicy,
    QGroupBox,
    QFormLayout,
    QDoubleSpinBox,
    QButtonGroup,
    QRadioButton,
    QComboBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap

if TYPE_CHECKING:
    from controllers.tab_classification_controller import TabClassificationController


class TabClassificationView(QWidget):
    """
    Vista de clasificación de imágenes.

    Recibe su controlador en el constructor y conecta las señales de sus
    botones a los métodos del controlador.
    """

    def __init__(self, controller: "TabClassificationController") -> None:
        super().__init__()
        self._controller = controller
        self._current_pixmap: QPixmap | None = None
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
        action_bar.setSpacing(8)

        self.btn_load_image = QPushButton("📂  Cargar Imagen")
        self.btn_load_image.setFixedHeight(36)
        self.btn_load_image.setFont(QFont("Arial", 10))
        self.btn_load_image.setToolTip("Selecciona una imagen DICOM (.dcm)")

        self.btn_evaluate = QPushButton("🧠  Evaluar")
        self.btn_evaluate.setFixedHeight(36)
        self.btn_evaluate.setFixedWidth(110)
        self.btn_evaluate.setFont(QFont("Arial", 10))
        self.btn_evaluate.setToolTip(
            "Evalúa la imagen con el modelo seleccionado"
        )
        self.btn_evaluate.setEnabled(False)

        self.combo_model = QComboBox()
        self.combo_model.addItems(["Random Forest", "CNN"])
        self.combo_model.setFixedHeight(36)
        self.combo_model.setMinimumWidth(140)
        self.combo_model.setFont(QFont("Arial", 10))
        self.combo_model.setToolTip("Selecciona el modelo de clasificación a utilizar")

        action_bar.addWidget(self.btn_load_image)
        action_bar.addWidget(self.combo_model)
        action_bar.addWidget(self.btn_evaluate)

        self.btn_compare_mask = QPushButton("📊  Comparar con máscara original")
        self.btn_compare_mask.setFixedHeight(36)
        self.btn_compare_mask.setFont(QFont("Arial", 10))
        self.btn_compare_mask.setToolTip(
            "Carga una máscara de referencia y la muestra en el panel izquierdo"
        )
        self.btn_compare_mask.setVisible(False)   # aparece tras segmentación U-NET

        action_bar.addWidget(self.btn_compare_mask)
        action_bar.addStretch()

        # ── Etiqueta de estado del CTProcessor ────────────────────────
        self.lbl_status = QLabel("Sin imagen cargada")
        self.lbl_status.setStyleSheet(
            "color: #6c7086; font-style: italic; font-size: 11px;"
        )

        # ── Selector de vista ──────────────────────────────────────────
        view_selector_group = QGroupBox("Vista")
        view_selector_group.setFont(QFont("Arial", 9))
        view_selector_layout = QHBoxLayout(view_selector_group)
        view_selector_layout.setSpacing(12)

        self.radio_full   = QRadioButton("🖼  Imagen Completa")
        self.radio_brain  = QRadioButton("🧠  Cerebro Extraído")
        self.radio_full.setChecked(True)   # Por defecto: imagen completa
        self.radio_brain.setEnabled(False) # Se habilita cuando hay imagen

        self._view_group = QButtonGroup(self)
        self._view_group.addButton(self.radio_full,  id=0)
        self._view_group.addButton(self.radio_brain, id=1)

        self.lbl_brain_status = QLabel("")
        self.lbl_brain_status.setStyleSheet(
            "color: #89b4fa; font-size: 10px; font-style: italic;"
        )

        view_selector_layout.addWidget(self.radio_full)
        view_selector_layout.addWidget(self.radio_brain)
        view_selector_layout.addWidget(self.lbl_brain_status)
        view_selector_layout.addStretch()

        # ── Área principal: imagen + panel derecho ─────────────────────
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # Panel izquierdo: imagen
        self.label_original_image = self._make_display_label("Imagen Original")

        # Panel derecho: controles HU + resultado IA
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)

        # ── Controles opcionales de ventana HU ────────────────────────
        self.group_hu_filter = QGroupBox("Ventana HU (Filtrado Opcional)")
        self.group_hu_filter.setFont(QFont("Arial", 9))
        self.group_hu_filter.setEnabled(False)
        hu_layout = QFormLayout(self.group_hu_filter)
        hu_layout.setSpacing(6)

        self.spin_vmin = QDoubleSpinBox()
        self.spin_vmin.setRange(-2000, 4000)
        self.spin_vmin.setDecimals(1)
        self.spin_vmin.setValue(0.0)
        self.spin_vmin.setSuffix(" HU")

        self.spin_vmax = QDoubleSpinBox()
        self.spin_vmax.setRange(-2000, 4000)
        self.spin_vmax.setDecimals(1)
        self.spin_vmax.setValue(100.0)
        self.spin_vmax.setSuffix(" HU")

        self.btn_apply_filter = QPushButton("✅  Aplicar Filtro")
        self.btn_apply_filter.setFixedHeight(30)
        self.btn_reset_filter = QPushButton("↩  Sin Filtro")
        self.btn_reset_filter.setFixedHeight(30)

        hu_layout.addRow("Mínimo:", self.spin_vmin)
        hu_layout.addRow("Máximo:", self.spin_vmax)
        filter_btns = QHBoxLayout()
        filter_btns.addWidget(self.btn_apply_filter)
        filter_btns.addWidget(self.btn_reset_filter)
        hu_layout.addRow(filter_btns)

        # ── Área de resultado de clasificación ────────────────────────
        self.label_classification_result = self._make_display_label(
            "Resultado de Clasificación (IA)"
        )

        right_panel.addWidget(self.group_hu_filter)
        right_panel.addWidget(self.label_classification_result, stretch=1)

        content_layout.addWidget(self.label_original_image, stretch=1)
        content_layout.addLayout(right_panel, stretch=1)

        # ── Ensamblado final ───────────────────────────────────────────
        main_layout.addLayout(action_bar)
        main_layout.addWidget(self.lbl_status)
        main_layout.addWidget(view_selector_group)
        main_layout.addLayout(content_layout)

    @staticmethod
    def _make_display_label(placeholder_text: str) -> QLabel:
        """Crea un QLabel estilizado como área de visualización de imagen."""
        label = QLabel(placeholder_text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFrameShape(QFrame.Shape.Box)
        label.setFrameShadow(QFrame.Shadow.Sunken)
        label.setMinimumSize(300, 320)
        label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        label.setScaledContents(False)
        label.setStyleSheet(
            "QLabel {"
            "  background-color: #16161e;"
            "  color: #6c7086;"
            "  border: 2px dashed #45475a;"
            "  border-radius: 6px;"
            "  font-size: 13px;"
            "}"
        )
        return label

    # ------------------------------------------------------------------
    # Conexión de señales → controlador
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self.btn_load_image.clicked.connect(self._controller.on_load_image)
        self.btn_evaluate.clicked.connect(self._controller.on_evaluate)
        self.btn_compare_mask.clicked.connect(self._controller.on_compare_mask)
        self.btn_apply_filter.clicked.connect(self._controller.on_apply_filter)
        self.btn_reset_filter.clicked.connect(self._controller.on_reset_filter)
        self._view_group.idToggled.connect(self._on_view_toggled)

    def _on_view_toggled(self, btn_id: int, checked: bool) -> None:
        """Notifica al controlador cuando el usuario cambia de vista."""
        if checked:
            self._controller.on_view_mode_changed(btn_id)

    # ------------------------------------------------------------------
    # Evento de redimensionado
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._current_pixmap is not None:
            self._update_displayed_pixmap()

    # ------------------------------------------------------------------
    # Métodos públicos de actualización (llamados por el controlador)
    # ------------------------------------------------------------------

    def set_original_image(self, pixmap: QPixmap) -> None:
        """Almacena y muestra un QPixmap en el área de imagen."""
        self._current_pixmap = pixmap
        self._update_displayed_pixmap()
        self.btn_evaluate.setEnabled(True)
        self.group_hu_filter.setEnabled(True)

    def _update_displayed_pixmap(self) -> None:
        """Reescala el pixmap almacenado al tamaño actual del label."""
        if self._current_pixmap is None:
            return
        scaled = self._current_pixmap.scaled(
            self.label_original_image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label_original_image.setPixmap(scaled)

    def set_classification_result(self, result_text: str) -> None:
        """Muestra el resultado de clasificación como texto."""
        self.label_classification_result.setText(result_text)
        self.label_classification_result.setPixmap(QPixmap())

    def set_status(self, message: str) -> None:
        """Actualiza el label de estado con el mensaje de CTProcessor."""
        self.lbl_status.setText(f"Estado: {message}")

    def set_brain_ready(self, ready: bool, info: str = "") -> None:
        """
        Habilita/deshabilita el radio button de cerebro y muestra info.

        Args:
            ready: True si la extracción fue exitosa.
            info:  Mensaje descriptivo (ej. "Extrayendo…" o "Listo").
        """
        self.radio_brain.setEnabled(ready)
        self.lbl_brain_status.setText(info)

    def get_hu_range(self) -> tuple[float, float]:
        """Retorna (vmin, vmax) de los spinboxes HU."""
        return self.spin_vmin.value(), self.spin_vmax.value()

    def get_view_mode(self) -> int:
        """Retorna el id del radio button activo (0=full, 1=brain)."""
        return self._view_group.checkedId()

    def get_selected_model(self) -> str:
        """Retorna el nombre del modelo seleccionado en el combo ('Random Forest' o 'CNN')."""
        return self.combo_model.currentText()

    def show_compare_button(self) -> None:
        """Hace visible el botón de comparación con máscara original."""
        self.btn_compare_mask.setVisible(True)

    def hide_compare_button(self) -> None:
        """Oculta el botón de comparación con máscara original."""
        self.btn_compare_mask.setVisible(False)
