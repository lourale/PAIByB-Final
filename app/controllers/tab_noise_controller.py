"""
controllers/tab_noise_controller.py
-------------------------------------
Controlador de la Pestaña 2 – Análisis de Ruido.

Flujo:
  refresh(ct)  ← llamado automáticamente desde TabClassificationController
    ├── Genera 4 figuras matplotlib directamente desde los datos de CTProcessor
    │   (raw_image, raw_hist_data, corner_hist_data, clean_hist_data).
    ├── Genera QPixmap de clean_image en rango completo (numpy directo).
    ├── Genera QPixmap de clean_image filtrada [vmin=0, vmax=80 HU] (numpy directo).
    └── Envía todos los pixmaps a TabNoiseView.

  on_apply_filter()  ← señal del botón "Aplicar" de la vista
    └── Lee vmin/vmax de la vista y regenera el panel filtrado.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import io

import numpy as np
from PyQt6.QtGui import QPixmap, QImage

if TYPE_CHECKING:
    from controllers.main_controller import MainController
    from views.tab_noise_view import TabNoiseView
    from logic.preprocessor import CTProcessor


class TabNoiseController:
    """Controlador para la pestaña de análisis de ruido."""

    def __init__(self, main_ctrl: MainController) -> None:
        self._main_ctrl = main_ctrl
        self._view: TabNoiseView | None = None

    def set_view(self, view: TabNoiseView) -> None:
        """Vincula la vista a este controlador."""
        self._view = view

    def on_brain_mask_ready(self) -> None:
        """
        Habilita el checkbox de overlay cuando la máscara cerebral está disponible.
        Llamado por MainController.notify_brain_mask_ready() tras extract_brain().
        """
        if self._view is not None:
            self._view.set_brain_mask_available(True)

    # ------------------------------------------------------------------
    # API pública — disparo automático desde el controlador de clasificación
    # ------------------------------------------------------------------

    def refresh(self, ct: CTProcessor) -> None:
        """
        Regenera todos los paneles de la pestaña de ruido a partir de ct.

        Llamado automáticamente por TabClassificationController después de
        procesar una imagen. No requiere interacción del usuario.

        Args:
            ct: Instancia de CTProcessor ya inicializada con la imagen actual.
        """
        if self._view is None:
            return

        print("  [NoiseController] Generando paneles de análisis...")

        # ── Fila 1: 4 paneles matplotlib (alta calidad, figura independiente) ──
        px_raw        = self._render_raw_image(ct)
        px_hist_raw   = self._render_histogram(
            ct.raw_hist_data,
            title="Histograma Crudo",
            color="gray",
            log=True,
        )
        px_corner     = self._render_histogram(
            ct.corner_hist_data,
            title=f"Ruido Esquina\n(Media: {np.mean(ct.corner_img):.1f})",
            color="#e06c75",
            log=False,
        )
        px_hist_clean = self._render_histogram(
            ct.clean_hist_data,
            title="Histograma Corregido",
            color="teal",
            log=True,
            vlines=[(-1000, "red", "Aire Std"), (0, "orange", "Tejido")],
        )

        self._view.set_analysis_panels(px_raw, px_hist_raw, px_corner, px_hist_clean)

        # ── Fila 2: imágenes directas numpy → QPixmap (máxima calidad) ─
        mask = self._main_ctrl.get_brain_mask()
        overlay = self._view.get_overlay_enabled() if self._view else False

        px_full = self._make_image_pixmap(ct.clean_image, None, None,
                                          mask if overlay else None)
        px_filtered = self._make_image_pixmap(ct.clean_image, 0.0, 80.0,
                                              mask if overlay else None)

        self._view.set_clean_full(px_full)
        self._view.set_clean_filtered(px_filtered, 0.0, 80.0)
        self._view.set_brain_mask_available(mask is not None)

        print("  [NoiseController] Paneles listos.")

    # ------------------------------------------------------------------
    # Slot — botón "Aplicar" de la vista
    # ------------------------------------------------------------------

    def on_apply_filter(self) -> None:
        """Regenera la imagen filtrada con el rango HU seleccionado."""
        if self._view is None:
            return

        ct = self._main_ctrl.get_ct_processor()
        if ct is None:
            return

        vmin, vmax = self._view.get_hu_range()
        if vmin >= vmax:
            print(f"  [AVISO] vmin ({vmin}) debe ser menor que vmax ({vmax}).")
            return

        mask = self._main_ctrl.get_brain_mask()
        overlay = self._view.get_overlay_enabled()

        px = self._make_image_pixmap(ct.clean_image, vmin, vmax,
                                     mask if overlay else None)
        self._view.set_clean_filtered(px, vmin, vmax)
        print(f"  [NoiseController] Filtro aplicado: [{vmin:.0f}, {vmax:.0f}] HU")

    def on_overlay_toggled(self, enabled: bool) -> None:
        """
        Refresca los paneles de imagen (raw, full, filtered) cuando el
        usuario activa/desactiva la superposición de máscara cerebral.
        """
        if self._view is None:
            return

        ct = self._main_ctrl.get_ct_processor()
        mask = self._main_ctrl.get_brain_mask()
        if ct is None:
            return

        m = mask if (enabled and mask is not None) else None

        # Imagen cruda con recuadro de esquina
        px_raw = self._render_raw_image(ct, overlay_mask=m)
        # Imagen limpia rango completo
        px_full = self._make_image_pixmap(ct.clean_image, None, None, m)
        # Imagen limpia con el filtro activo en la vista
        vmin, vmax = self._view.get_hu_range()
        px_filtered = self._make_image_pixmap(ct.clean_image, vmin, vmax, m)

        # Solo actualizar los pixmaps de imagen (no los histogramas)
        self._view._px_raw = px_raw
        self._view._set_panel(self._view.lbl_raw, px_raw)
        self._view.set_clean_full(px_full)
        self._view.set_clean_filtered(px_filtered, vmin, vmax)
        print(f"  [NoiseController] Overlay {'activado' if enabled else 'desactivado'}")

    # ------------------------------------------------------------------
    # Helpers: generadores de QPixmap independientes (alta calidad)
    # ------------------------------------------------------------------

    @staticmethod
    def _figure_to_pixmap(fig, dpi: int = 150) -> QPixmap:
        """
        Convierte una Figure matplotlib a QPixmap via buffer PNG en memoria.

        Args:
            fig: matplotlib Figure ya configurada.
            dpi: Resolución de exportación (mayor = más calidad).

        Returns:
            QPixmap resultante.
        """
        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        png_data = buf.read()
        plt.close(fig)

        q_image = QImage.fromData(png_data)   # type: ignore[arg-type]
        return QPixmap.fromImage(q_image)

    @staticmethod
    def _render_raw_image(
        ct: "CTProcessor",
        overlay_mask: "np.ndarray | None" = None,
    ) -> QPixmap:
        """
        Genera un QPixmap con la imagen cruda, recuadro rojo en la esquina
        y (opcionalmente) la máscara cerebral superpuesta con color #34477a,
        transparencia 30 %.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Normalizar la imagen cruda a [0, 1] para imshow
        raw = ct.raw_image.astype(np.float32)
        r_min, r_max = raw.min(), raw.max()
        if r_max > r_min:
            raw_norm = (raw - r_min) / (r_max - r_min)
        else:
            raw_norm = np.zeros_like(raw)

        # Convertir a RGB para poder superponer colores
        raw_rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

        if overlay_mask is not None:
            raw_rgb = TabNoiseController._apply_overlay(
                raw_rgb, overlay_mask, alpha=0.6
            )

        fig, ax = plt.subplots(figsize=(4, 4), facecolor="#1e1e2e")
        ax.imshow(raw_rgb)
        ax.set_title("Imagen Cruda", color="white", fontsize=10)
        ax.axis("off")
        rect = patches.Rectangle(
            (0, 0), ct.corner_size, ct.corner_size,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        fig.tight_layout(pad=0.3)
        return TabNoiseController._figure_to_pixmap(fig)

    @staticmethod
    def _render_histogram(
        data: np.ndarray,
        title: str,
        color: str,
        log: bool,
        vlines: list | None = None,
    ) -> QPixmap:
        """
        Genera un QPixmap con un histograma de los datos proporcionados.

        Args:
            data:   Array 1D de valores.
            title:  Título del panel.
            color:  Color de las barras.
            log:    Si True, escala logarítmica en eje Y.
            vlines: Lista de (valor, color, label) para líneas verticales.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 3), facecolor="#1e1e2e")
        ax.set_facecolor("#181825")

        ax.hist(data, bins=100, color=color, log=log, alpha=0.85)
        ax.set_title(title, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
        ax.grid(True, alpha=0.2, color="#45475a")

        if vlines:
            for x_val, vcolor, label in vlines:
                ax.axvline(x_val, color=vcolor, linestyle="--",
                           alpha=0.75, linewidth=1.2, label=label)
            ax.legend(fontsize=7, labelcolor="white",
                      facecolor="#313244", edgecolor="#45475a")

        fig.tight_layout(pad=0.4)
        return TabNoiseController._figure_to_pixmap(fig)

    @staticmethod
    def _make_image_pixmap(
        image: np.ndarray,
        vmin: "float | None",
        vmax: "float | None",
        overlay_mask: "np.ndarray | None" = None,
    ) -> QPixmap:
        """
        Convierte un array 2D float a QPixmap.

        Si vmin/vmax son None, normaliza al rango completo del array.
        Si overlay_mask se provee, superpone el color cerebral (30 % alpha).
        Trabaja directamente en numpy (máxima calidad, sin matplotlib).
        """
        arr = image.astype(np.float32)
        if vmin is not None and vmax is not None:
            arr = np.clip(arr, vmin, vmax)
            span = vmax - vmin
            gray = ((arr - vmin) / span * 255).astype(np.uint8) if span > 0 \
                else np.zeros_like(arr, dtype=np.uint8)
        else:
            a_min, a_max = arr.min(), arr.max()
            gray = ((arr - a_min) / (a_max - a_min) * 255).astype(np.uint8) \
                if a_max > a_min else np.zeros_like(arr, dtype=np.uint8)

        gray = np.ascontiguousarray(gray)

        if overlay_mask is None:
            # Escalar de grises directo
            h, w = gray.shape
            q_img = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            # Convertir a RGB y aplicar overlay
            rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
            rgb_out = TabNoiseController._apply_overlay(
                rgb / 255.0, overlay_mask, alpha=0.6
            )  # [0,1] float
            rgb_out = (rgb_out * 255).clip(0, 255).astype(np.uint8)
            rgb_out = np.ascontiguousarray(rgb_out)
            h, w, _ = rgb_out.shape
            q_img = QImage(rgb_out.data, w, h, w * 3,
                           QImage.Format.Format_RGB888)

        return QPixmap.fromImage(q_img)

    @staticmethod
    def _apply_overlay(
        rgb_float: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """
        Mezcla un color de overlay sobre las regiones activas de la máscara.

        Args:
            rgb_float: Array (H, W, 3) con valores en [0, 1].
            mask:      Array uint8 (H, W) con valores 0 o 255 del BrainExtractor.
            alpha:     Transparencia del overlay (0 = invisible, 1 = sólido).

        Returns:
            Array (H, W, 3) float en [0, 1] con el overlay aplicado.
        """
        # Color overlay #34477a en [0, 1]
        overlay_color = np.array([0x34 / 255.0, 0x47 / 255.0, 0x7a / 255.0],
                                 dtype=np.float32)
        result = rgb_float.copy()
        brain_region = mask > 128
        result[brain_region] = (
            (1 - alpha) * result[brain_region] + alpha * overlay_color
        ).clip(0, 1)
        return result

    @staticmethod
    def _array_to_pixmap_full_range(image: np.ndarray) -> QPixmap:
        """Convierte un array 2D float a QPixmap normalizando al rango completo."""
        return TabNoiseController._make_image_pixmap(image, None, None, None)

    @staticmethod
    def _array_to_pixmap_windowed(
        image: np.ndarray, vmin: float, vmax: float
    ) -> QPixmap:
        """Convierte un array 2D float a QPixmap aplicando ventana HU."""
        return TabNoiseController._make_image_pixmap(image, vmin, vmax, None)

