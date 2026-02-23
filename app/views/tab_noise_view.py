"""
views/tab_noise_view.py
------------------------
Vista de la Pestaña 2 – Análisis de Ruido en Imágenes.

Layout de dos filas:
  Fila 1 (análisis técnico): 4 paneles matplotlib generados directamente desde
          los datos de CTProcessor — imagen cruda, histograma crudo, ruido de
          esquina, histograma corregido.
  Fila 2 (clínica): imagen limpia en rango completo | imagen limpia filtrada +
          controles para seleccionar ventana HU (vmin/vmax) y botón Aplicar.

El análisis se dispara automáticamente al cargar una imagen (sin botón).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QGroupBox,
    QFormLayout,
    QSizePolicy,
    QPushButton,
    QDoubleSpinBox,
    QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap

if TYPE_CHECKING:
    from controllers.tab_noise_controller import TabNoiseController

# ──────────────────────────────────────────────────────────────────────────────
# Constantes de estilo
# ──────────────────────────────────────────────────────────────────────────────

_PANEL_STYLE = (
    "QLabel {"
    "  background-color: #16161e;"
    "  color: #6c7086;"
    "  border: 2px dashed #45475a;"
    "  border-radius: 6px;"
    "  font-size: 12px;"
    "}"
)

_DEFAULT_VMIN = 0.0
_DEFAULT_VMAX = 80.0


class TabNoiseView(QWidget):
    """
    Vista de análisis de ruido con layout de dos filas.

    El controlador llama a refresh(ct) para actualizar todos los paneles.
    Los controles de ventana HU emiten la señal al controlador cuando el
    usuario pulsa 'Aplicar'.
    """

    def __init__(self, controller: "TabNoiseController") -> None:
        super().__init__()
        self._controller = controller
        # Pixmaps activos — guardados para reescalado en resizeEvent
        self._px_raw: QPixmap | None = None
        self._px_hist_raw: QPixmap | None = None
        self._px_corner: QPixmap | None = None
        self._px_hist_clean: QPixmap | None = None
        self._px_clean_full: QPixmap | None = None
        self._px_clean_filtered: QPixmap | None = None

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Construye el layout de dos filas."""
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── Barra de controles globales ─────────────────────────────
        ctrl_bar = QHBoxLayout()
        self.chk_brain_overlay = QCheckBox("🧠  Superponer máscara cerebral")
        self.chk_brain_overlay.setFont(QFont("Arial", 9))
        self.chk_brain_overlay.setChecked(False)
        self.chk_brain_overlay.setEnabled(False)   # Se activa cuando hay máscara
        ctrl_bar.addWidget(self.chk_brain_overlay)
        ctrl_bar.addStretch()

        # ── Fila 1: 4 paneles de análisis técnico ─────────────────────
        row1_group = QGroupBox("Análisis Técnico")
        row1_group.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        row1_layout = QHBoxLayout(row1_group)
        row1_layout.setSpacing(6)

        self.lbl_raw        = self._make_panel("Imagen Cruda")
        self.lbl_hist_raw   = self._make_panel("Histograma Crudo")
        self.lbl_corner     = self._make_panel("Ruido Esquina")
        self.lbl_hist_clean = self._make_panel("Histograma Corregido")

        for lbl in (self.lbl_raw, self.lbl_hist_raw,
                    self.lbl_corner, self.lbl_hist_clean):
            row1_layout.addWidget(lbl)

        # ── Fila 2: imagen limpia full + filtrada + controles HU ───────
        row2_group = QGroupBox("Visualización Clínica")
        row2_group.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        row2_layout = QHBoxLayout(row2_group)
        row2_layout.setSpacing(8)

        # Panel imagen limpia rango completo
        self.lbl_clean_full = self._make_panel("Corregida\n(Rango Completo)")
        self.lbl_clean_full.setMinimumWidth(200)

        # Panel imagen limpia filtrada
        self.lbl_clean_filtered = self._make_panel(
            f"Filtrada [{int(_DEFAULT_VMIN)}–{int(_DEFAULT_VMAX)} HU]"
        )
        self.lbl_clean_filtered.setMinimumWidth(200)

        # ── Panel de controles HU ──────────────────────────────────────
        ctrl_box = QGroupBox("Ventana HU")
        ctrl_box.setFont(QFont("Arial", 9))
        ctrl_box.setFixedWidth(170)
        ctrl_layout = QFormLayout(ctrl_box)
        ctrl_layout.setSpacing(8)

        self.spin_vmin = QDoubleSpinBox()
        self.spin_vmin.setRange(-2000, 4000)
        self.spin_vmin.setDecimals(1)
        self.spin_vmin.setValue(_DEFAULT_VMIN)
        self.spin_vmin.setSuffix(" HU")

        self.spin_vmax = QDoubleSpinBox()
        self.spin_vmax.setRange(-2000, 4000)
        self.spin_vmax.setDecimals(1)
        self.spin_vmax.setValue(_DEFAULT_VMAX)
        self.spin_vmax.setSuffix(" HU")

        self.btn_apply_filter = QPushButton("✅  Aplicar")
        self.btn_apply_filter.setFixedHeight(32)

        ctrl_layout.addRow("Mínimo:", self.spin_vmin)
        ctrl_layout.addRow("Máximo:", self.spin_vmax)
        ctrl_layout.addRow(self.btn_apply_filter)

        row2_layout.addWidget(self.lbl_clean_full, stretch=1)
        row2_layout.addWidget(self.lbl_clean_filtered, stretch=1)
        row2_layout.addWidget(ctrl_box, stretch=0)

        # ── Ensamblado ─────────────────────────────────────────────────
        root.addLayout(ctrl_bar)
        root.addWidget(row1_group, stretch=1)
        root.addWidget(row2_group, stretch=1)

    @staticmethod
    def _make_panel(text: str) -> QLabel:
        """Crea un QLabel estilizado como panel de visualización."""
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFrameShape(QFrame.Shape.Box)
        lbl.setFrameShadow(QFrame.Shadow.Sunken)
        lbl.setMinimumSize(160, 180)
        lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        # setScaledContents(False): escalamos manualmente con KeepAspectRatio
        # para preservar la proporción real de la imagen.
        lbl.setScaledContents(False)
        lbl.setStyleSheet(_PANEL_STYLE)
        return lbl


    # ------------------------------------------------------------------
    # Señales → Controlador
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self.btn_apply_filter.clicked.connect(self._controller.on_apply_filter)
        self.chk_brain_overlay.stateChanged.connect(
            lambda state: self._controller.on_overlay_toggled(
                state == Qt.CheckState.Checked.value
            )
        )

    # ------------------------------------------------------------------
    # Reescalado dinámico
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        """Reescala todos los paneles manteniendo proporción al cambiar tamaño."""
        super().resizeEvent(event)
        self._rescale_all()

    def showEvent(self, event) -> None:
        """Al mostrarse por primera vez, rescala con tamaños ya definitivos."""
        super().showEvent(event)
        QTimer.singleShot(50, self._rescale_all)  # 50 ms garantiza layout terminado

    def _rescale_all(self) -> None:
        """Reescala todos los paneles con KeepAspectRatio."""
        pairs = [
            (self.lbl_raw,            self._px_raw),
            (self.lbl_hist_raw,       self._px_hist_raw),
            (self.lbl_corner,         self._px_corner),
            (self.lbl_hist_clean,     self._px_hist_clean),
            (self.lbl_clean_full,     self._px_clean_full),
            (self.lbl_clean_filtered, self._px_clean_filtered),
        ]
        for lbl, px in pairs:
            if px is not None and lbl.size().width() > 1:
                lbl.setPixmap(
                    px.scaled(
                        lbl.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

    def _set_panel(self, lbl: QLabel, px: QPixmap) -> None:
        """Asigna y escala un pixmap al panel con KeepAspectRatio."""
        if lbl.size().width() > 1:
            lbl.setPixmap(
                px.scaled(
                    lbl.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            # Widget aún sin tamaño definitivo; QTimer reescalará tras el primer show
            lbl.setPixmap(px)


    # ------------------------------------------------------------------
    # API pública — llamada por el controlador
    # ------------------------------------------------------------------

    def set_analysis_panels(
        self,
        px_raw: QPixmap,
        px_hist_raw: QPixmap,
        px_corner: QPixmap,
        px_hist_clean: QPixmap,
    ) -> None:
        """Actualiza los 4 paneles de la fila de análisis técnico."""
        self._px_raw        = px_raw
        self._px_hist_raw   = px_hist_raw
        self._px_corner     = px_corner
        self._px_hist_clean = px_hist_clean

        self._set_panel(self.lbl_raw,        px_raw)
        self._set_panel(self.lbl_hist_raw,   px_hist_raw)
        self._set_panel(self.lbl_corner,     px_corner)
        self._set_panel(self.lbl_hist_clean, px_hist_clean)
        # Rescalar después del primer show cuando los tamaños ya son definitivos
        QTimer.singleShot(0, self._rescale_all)

    def set_clean_full(self, px: QPixmap) -> None:
        """Muestra la imagen limpia en rango completo (sin ventana)."""
        self._px_clean_full = px
        self._set_panel(self.lbl_clean_full, px)
        QTimer.singleShot(0, self._rescale_all)

    def set_clean_filtered(self, px: QPixmap, vmin: float, vmax: float) -> None:
        """Muestra la imagen limpia filtrada."""
        self._px_clean_filtered = px
        self.lbl_clean_filtered.setText("")
        self._set_panel(self.lbl_clean_filtered, px)
        QTimer.singleShot(0, self._rescale_all)

    def get_hu_range(self) -> tuple[float, float]:
        """Retorna (vmin, vmax) de los spinboxes HU."""
        return self.spin_vmin.value(), self.spin_vmax.value()

    def get_overlay_enabled(self) -> bool:
        """Retorna True si el checkbox de máscara cerebral está activo."""
        return self.chk_brain_overlay.isChecked()

    def set_brain_mask_available(self, available: bool) -> None:
        """Habilita o deshabilita el checkbox según si hay máscara lista."""
        self.chk_brain_overlay.setEnabled(available)
