"""
controllers/tab_classification_controller.py
---------------------------------------------
Controlador de la Pestaña 1 – Clasificación de Imágenes.

Flujo principal:
  1. on_load_image(): abre diálogo DICOM desde el último directorio usado.
  2. ImageAdapter convierte el .dcm a numpy array.
  3. CTProcessor procesa la imagen → clean_image (float32, HU).
  4. BrainExtractor corre sobre clean_image → skull_stripped_image.
  5. Se muestra la imagen seleccionada (completa o cerebro) en la vista.

Selector de vista (radio buttons):
  - "Imagen Completa" (id=0): muestra clean_image en rango completo.
  - "Cerebro Extraído" (id=1): muestra skull_stripped_image (0–80 HU).

Filtro HU opcional:
  - on_apply_filter(): aplica ventana [vmin, vmax] a la imagen activa.
  - on_reset_filter(): restaura la imagen activa al rango completo.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap, QImage, QPainter, QFont, QColor, QBrush, QPen
from PyQt6.QtCore import Qt, QRectF

from logic.image_adapter import ImageAdapter
from logic.preprocessor import CTProcessor
from logic.app_settings import AppSettings
from logic.RF_logic.prediction_worker import PredictionWorker, load_rf_model
from logic.cnn_logic.cnn_worker import CnnWorker
from logic.unet_logic.unet_worker import UnetWorker

if TYPE_CHECKING:
    from controllers.main_controller import MainController
    from views.tab_classification_view import TabClassificationView


# IDs de los radio buttons de modo de vista
_MODE_FULL  = 0   # Imagen completa
_MODE_BRAIN = 1   # Cerebro extraído


class TabClassificationController:
    """Controlador para la pestaña de clasificación de imágenes CT."""

    def __init__(self, main_ctrl: MainController) -> None:
        self._main_ctrl = main_ctrl
        self._view: TabClassificationView | None = None

        # Estado de las imágenes activas
        self._ct: CTProcessor | None = None
        self._brain_image: np.ndarray | None = None
        self._view_mode: int = _MODE_FULL
        self._hu_filter: tuple[float, float] | None = None
        self._pred_mask: np.ndarray | None = None
        self._last_image_path: str | None = None
        # Estado persistente de la predicción CNN (para refrescar con filtro HU)
        self._cnn_predicted_class: str | None = None
        self._cnn_probs: list | None = None

        # Workers (referencias para evitar GC durante ejecución)
        self._rf_model = None
        self._worker: PredictionWorker | None = None
        self._cnn_worker: CnnWorker | None = None
        self._unet_worker: UnetWorker | None = None
        # Estado de la última segmentación U-NET
        self._unet_seg_rgb: np.ndarray | None = None   # float32 (H,W,3) base para comparar
        self._unet_seg_mask: np.ndarray | None = None  # float32 (H,W) máscara cruda 0/1
        self._unet_fallback_tried: bool = False        # evita loops de fallback

    def set_view(self, view: TabClassificationView) -> None:
        """Vincula la vista a este controlador."""
        self._view = view

    # ------------------------------------------------------------------
    # Slots / manejadores de eventos
    # ------------------------------------------------------------------

    def on_load_image(self) -> None:
        """
        Abre diálogo DICOM, procesa con CTProcessor y extrae el cerebro.
        Guarda el directorio seleccionado para la próxima sesión.
        """
        print("Seleccionaste el botón [Cargar Imagen]")

        if self._view is None:
            return

        # ── 1. Diálogo desde el último directorio guardado ─────────────
        last_dir = AppSettings.get_last_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self._view,
            "Seleccionar imagen DICOM",
            last_dir,
            "DICOM (*.dcm);;Todos los archivos (*)",
        )
        if not file_path:
            return

        AppSettings.save_last_dir(file_path)

        # ── 2. Convertir a numpy array vía adaptador ───────────────────
        try:
            image_array: np.ndarray = ImageAdapter.from_file(file_path)
        except ValueError as exc:
            QMessageBox.warning(self._view, "Formato no compatible", str(exc))
            return
        except RuntimeError as exc:
            QMessageBox.critical(self._view, "Error al cargar imagen", str(exc))
            return

        # ── 3. Procesar con CTProcessor ────────────────────────────────
        try:
            ct = CTProcessor(image_array)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self._view, "Error en CTProcessor",
                f"No se pudo procesar la imagen:\n{exc}",
            )
            return

        self._ct = ct
        self._brain_image = None
        self._hu_filter = None
        self._view_mode = _MODE_FULL
        self._pred_mask = None
        self._last_image_path = file_path
        self._cnn_predicted_class = None
        self._cnn_probs = None
        self._unet_seg_rgb = None
        self._unet_seg_mask = None
        self._unet_fallback_tried = False
        if self._view is not None:
            self._view.hide_compare_button()

        # Guardar en el hub compartido → dispara auto-refresh de pestaña Ruido
        self._main_ctrl.set_ct_processor(ct)
        print(f"  → CTProcessor listo | Estado: {ct.status_msg}")

        # ── 4. Mostrar imagen completa en rango completo ───────────────
        self._view.set_status(ct.status_msg)
        pixmap = self._array_to_pixmap_full_range(ct.clean_image)
        if pixmap:
            self._view.set_original_image(pixmap)
        print("  → Imagen completa mostrada")

        # ── 5. Extraer cerebro en segundo plano (bloqueante pero rápido)
        self._view.set_brain_ready(False, "⏳ Extrayendo cerebro…")
        try:
            from logic.brain_extraction.brain_extractor import BrainExtractor
            extractor = BrainExtractor(ct.clean_image)
            skull_stripped, brain_mask = extractor.extract_brain(plot=False)
            self._brain_image = skull_stripped
            self._main_ctrl.set_brain_mask(brain_mask)
            self._main_ctrl.notify_brain_mask_ready()   # habilita checkbox en pestaña Ruido
            self._view.set_brain_ready(True, "✅ Cerebro listo")
            print("  → Extracción de cerebro completada")
        except ImportError:
            self._view.set_brain_ready(False, "⚠ cv2 no instalado (pip install opencv-python)")
        except Exception as exc:  # noqa: BLE001
            self._view.set_brain_ready(False, f"⚠ Error en extracción: {exc}")
            print(f"  [ERROR BrainExtractor] {exc}")

    def on_view_mode_changed(self, mode_id: int) -> None:
        """
        Conmuta entre imagen completa y cerebro extraído.

        Args:
            mode_id: 0 = imagen completa, 1 = cerebro extraído.
        """
        self._view_mode = mode_id
        self._refresh_display()

    def on_apply_filter(self) -> None:
        """Aplica ventana HU a ambos paneles con el mismo rango."""
        if self._view is None:
            return
        vmin, vmax = self._view.get_hu_range()
        if vmin >= vmax:
            print(f"  [AVISO] vmin ({vmin}) debe ser menor que vmax ({vmax})")
            return
        self._hu_filter = (vmin, vmax)
        self._refresh_display()
        print(f"  → Filtro [{vmin:.0f}, {vmax:.0f}] HU aplicado")

    def on_reset_filter(self) -> None:
        """Elimina el filtro HU y muestra ambos paneles en rango completo."""
        self._hu_filter = None
        self._refresh_display()
        print("  → Filtro eliminado: rango completo")

    def on_evaluate(self) -> None:
        """
        Despacha la evaluación al método correspondiente según el modelo
        seleccionado en el combo de la vista.

        Para agregar un nuevo modelo:
          1. Agregar su nombre al combo en tab_classification_view.py.
          2. Implementar un método _evaluate_<nombre>(self) -> None aquí.
          3. Agregar el elif correspondiente en este método.
        """
        if self._view is None or self._ct is None:
            return

        model_name = self._view.get_selected_model()

        if model_name == "Random Forest":
            self._evaluate_rf()
        elif model_name == "CNN":
            self._evaluate_cnn()
        else:
            self._view.set_classification_result(
                f"⚠ Modelo '{model_name}' no implementado."
            )

    # ------------------------------------------------------------------
    # Evaluadores por modelo
    # ------------------------------------------------------------------

    def _evaluate_rf(self) -> None:
        """
        Lanza la predicción con el modelo Random Forest usando sliding window.
        Carga el modelo lazy (solo la primera vez) y corre en QThread.
        """
        brain_mask = self._main_ctrl.get_brain_mask()
        if brain_mask is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self._view,
                "Máscara no disponible",
                "La extracción de cerebro aún no terminó.\n"
                "Espere unos segundos e intente de nuevo.",
            )
            return

        # Cargar modelo (lazy, una sola vez)
        if self._rf_model is None:
            self._view.set_status("Cargando modelo Random Forest…")
            try:
                self._rf_model = load_rf_model()
            except FileNotFoundError as exc:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self._view, "Modelo no encontrado", str(exc))
                return

        self._view.btn_evaluate.setEnabled(False)
        self._view.set_status("⏳ Prediciendo con Random Forest…")
        self._view.set_classification_result("⏳ Analizando imagen…")

        self._worker = PredictionWorker(
            model=self._rf_model,
            dicom_image=self._ct.clean_image,
            brain_mask=brain_mask,
        )
        self._worker.progress.connect(self._view.set_status)
        self._worker.finished.connect(self._on_prediction_ready)
        self._worker.error.connect(self._on_prediction_error)
        self._worker.start()
        print("  → Worker RF iniciado")

    def _evaluate_cnn(self) -> None:
        """
        Lanza la predicción con la CNN en un hilo separado.
        La CNN clasifica la imagen completa (Normal / Bleeding / Ischemia)
        sin localizar regiones.
        """
        if self._last_image_path is None:
            self._view.set_classification_result(
                "⚠ No hay imagen cargada para evaluar."
            )
            return

        self._view.btn_evaluate.setEnabled(False)
        self._view.set_status("⏳ Cargando CNN y realizando inferencia…")
        self._view.set_classification_result("⏳ Analizando con CNN…")

        self._cnn_worker = CnnWorker(image_path=self._last_image_path)
        self._cnn_worker.progress.connect(self._view.set_status)
        self._cnn_worker.finished.connect(self._on_cnn_ready)
        self._cnn_worker.error.connect(self._on_cnn_error)
        self._cnn_worker.start()
        print("  → Worker CNN iniciado")

    # ------------------------------------------------------------------
    # Callbacks del worker CNN
    # ------------------------------------------------------------------

    def _on_cnn_ready(self, predicted_class: str, probs: list) -> None:
        """
        Procesa el resultado de la CNN.

        - Normal:             muestra resultado directo (no hay U-NET).
        - Bleeding/Ischemia:  solo guarda estado y lanza U-NET automáticamente;
                              el display final ocurre cuando termina U-NET.
        """
        if self._view is None or self._ct is None:
            return

        from logic.cnn_logic.cnn_worker import CNN_CLASS_NAMES
        confidence = probs[CNN_CLASS_NAMES.index(predicted_class)] * 100.0

        # Persistir siempre (necesario para badge y filtro HU)
        self._cnn_predicted_class = predicted_class
        self._cnn_probs = probs

        prob_lines = " | ".join(
            f"{name}: {p*100:.1f}%"
            for name, p in zip(CNN_CLASS_NAMES, probs)
        )
        icons = {"Normal": "✅", "Bleeding": "🔴", "Ischemia": "🔵"}
        icon = icons.get(predicted_class, "🧠")
        print(f"  → CNN: {icon} {predicted_class} ({confidence:.1f}%) | {prob_lines}")

        if predicted_class in ("Bleeding", "Ischemia"):
            # No mostrar resultado parcial — esperar a U-NET
            self._view.set_status(
                f"{icon} CNN → {predicted_class} | ⏳ Segmentando con U-NET…"
            )
            self.on_segment_unet()   # disparo automático
        else:
            # Normal: no hay U-NET, mostrar resultado final ahora
            self._view.btn_evaluate.setEnabled(True)
            self._view.set_status(f"✅ {icon} CNN → Normal ({confidence:.1f}%)")
            self._refresh_cnn_panel()

    @staticmethod
    def _draw_prediction_badge(
        pixmap: QPixmap,
        predicted_class: str,
        confidence: float,
    ) -> QPixmap:
        """
        Dibuja un badge de predicción en la esquina superior-derecha del pixmap.

        Args:
            pixmap:          QPixmap sobre el que se dibuja (se modifica una copia).
            predicted_class: Nombre de la clase predicha.
            confidence:      Probabilidad en tanto por ciento (0-100).

        Returns:
            Nuevo QPixmap con el badge dibujado.
        """
        _BADGE_BG = {
            "Bleeding": QColor("#370f1a"),
            "Ischemia": QColor("#20515f"),
            "Normal":   QColor("#634c52"),
        }
        bg_color = _BADGE_BG.get(predicted_class, QColor("#45475a"))

        # Copiar para no mutar el origen
        result = pixmap.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Solo el nombre de la clase — sin porcentaje
        label_text = predicted_class
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(label_text)
        text_h = fm.height()

        padding_x = 10
        padding_y = 6
        radius    = 6
        badge_w   = text_w + padding_x * 2
        badge_h   = text_h + padding_y * 2
        margin    = 8

        badge_x = result.width()  - badge_w - margin
        badge_y = margin

        rect = QRectF(badge_x, badge_y, badge_w, badge_h)

        # Fondo semitransparente
        bg = QColor(bg_color)
        bg.setAlpha(220)
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(Qt.GlobalColor.transparent))
        painter.drawRoundedRect(rect, radius, radius)

        # Texto blanco
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignCenter,
            label_text,
        )

        painter.end()
        return result

    def _on_cnn_error(self, message: str) -> None:
        """Maneja errores del worker CNN."""
        if self._view is None:
            return
        self._view.btn_evaluate.setEnabled(True)
        self._view.set_status("❌ Error en CNN")
        self._view.set_classification_result(f"❌ Error CNN:\n{message}")
        print(f"  [ERROR CNN] {message}")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self._view, "Error en predicción CNN", message)

    # ------------------------------------------------------------------
    # U-NET: segmentación post-CNN
    # ------------------------------------------------------------------

    def on_segment_unet(self) -> None:
        """
        Lanza la segmentación U-NET en background según la clase predicha
        por la CNN (Bleeding o Ischemia). Se llama automáticamente desde
        _on_cnn_ready() y también puede ser invocado manualmente.
        """
        if self._view is None or self._ct is None:
            return
        if self._cnn_predicted_class not in ("Bleeding", "Ischemia"):
            return

        # Resetear el flag de fallback al inicio de una nueva segmentación
        self._unet_fallback_tried = False

        self._view.btn_evaluate.setEnabled(False)
        self._view.set_status(
            f"⏳ Segmentando con U-NET ({self._cnn_predicted_class})…"
        )

        self._unet_worker = UnetWorker(
            image_array=self._ct.clean_image,
            predicted_class=self._cnn_predicted_class,
        )
        self._unet_worker.progress.connect(self._view.set_status)
        self._unet_worker.finished.connect(self._on_unet_ready)
        self._unet_worker.error.connect(self._on_unet_error)
        self._unet_worker.start()
        print(f"  → Worker U-NET iniciado ({self._cnn_predicted_class})")

    def _on_unet_ready(self, seg_mask: np.ndarray) -> None:
        """
        Recibe la máscara de segmentación y la superpone sobre la imagen.

        Fallback automático:
          Si la cobertura de la clase predicha es < 5%, se relanza con la
          clase opuesta (Bleeding ↔ Ischemia) y se adopta ese resultado.
          Esto sucede solo una vez por segmentación manual.
        """
        if self._view is None or self._ct is None:
            return

        # ── Reescalar máscara al tamaño de la imagen CT ───────────────────
        image = self._active_image()
        if image is None:
            return
        h, w = image.shape
        if seg_mask.shape[:2] != (h, w):
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(
                (seg_mask.squeeze() * 255).clip(0, 255).astype(np.uint8)
            )
            mask_pil = mask_pil.resize((w, h), PILImage.NEAREST)
            seg_mask = np.asarray(mask_pil).astype(np.float32) / 255.0
        else:
            seg_mask = seg_mask.squeeze()

        region = seg_mask > 0.5
        area_pct = 100.0 * float(np.sum(region)) / max(region.size, 1)
        predicted_class = self._cnn_predicted_class or ""

        # ── Fallback: si cobertura < 5% y aún no lo intentamos ────────────
        _FALLBACK_THRESHOLD = 5.0
        if area_pct < _FALLBACK_THRESHOLD and not self._unet_fallback_tried:
            fallback_class = "Ischemia" if predicted_class == "Bleeding" else "Bleeding"
            print(
                f"  → U-NET {predicted_class}: {area_pct:.1f}% < {_FALLBACK_THRESHOLD}% "
                f"| Reintentando con {fallback_class}…"
            )
            self._unet_fallback_tried = True
            self._view.set_status(
                f"⏳ Baja cobertura {predicted_class} ({area_pct:.1f}%) "
                f"| Analizando con {fallback_class}…"
            )
            # Lanzar segundo worker con clase opuesta (sin modificar _cnn_predicted_class)
            # El resultado volverá a _on_unet_ready y será procesado normalmente.
            self._cnn_predicted_class = fallback_class
            self._unet_worker = UnetWorker(
                image_array=self._ct.clean_image,
                predicted_class=fallback_class,
            )
            self._unet_worker.progress.connect(self._view.set_status)
            self._unet_worker.finished.connect(self._on_unet_ready)
            self._unet_worker.error.connect(self._on_unet_error)
            self._unet_worker.start()
            return   # esperar resultado del fallback

        # ── Segundo fallback: si el fallback también da 0% → Normal ──────────
        # Esto cubre: CNN(Bleeding) → U-NET(Bleeding)<5% → U-NET(Ischemia)=0%
        _ZERO_THRESHOLD = 0.0
        if self._unet_fallback_tried and area_pct <= _ZERO_THRESHOLD:
            print(f"  → U-NET fallback: {area_pct:.1f}% | Sin hallazgos → Normal")
            self._cnn_predicted_class = "Normal"
            self._cnn_probs = self._cnn_probs  # mantener probs originales de la CNN
            self._unet_seg_mask = None
            self._unet_seg_rgb = None
            self._view.btn_evaluate.setEnabled(True)
            self._view.hide_compare_button()
            self._view.set_status("✅ ✅ CNN+U-NET → Normal (sin regiones detectadas)")
            self._refresh_cnn_panel()
            return

        # ── Componer overlay con la clase final ────────────────────────────
        # (predicted_class puede ser el original o el del fallback)
        predicted_class = self._cnn_predicted_class or ""

        self._view.btn_evaluate.setEnabled(True)

        overlay_colors = {
            "Bleeding": np.array([0xe0 / 255.0, 0x6c / 255.0, 0x75 / 255.0],
                                 dtype=np.float32),
            "Ischemia": np.array([0x89 / 255.0, 0xb4 / 255.0, 0xfa / 255.0],
                                 dtype=np.float32),
        }
        color = overlay_colors.get(
            predicted_class,
            np.array([1.0, 1.0, 0.0], dtype=np.float32),
        )

        if self._hu_filter is not None:
            vmin, vmax = self._hu_filter
        else:
            vmin, vmax = 0.0, 80.0

        span = vmax - vmin
        arr = np.clip(image, vmin, vmax).astype(np.float32)
        arr = (arr - vmin) / span if span > 0 else np.zeros_like(arr)
        rgb = np.stack([arr, arr, arr], axis=-1)

        if region.any():
            rgb[region] = (0.45 * rgb[region] + 0.55 * color).clip(0, 1)

        # ── Convertir a QPixmap ──────────────────────────────────────────
        rgb_u8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        rgb_u8 = np.ascontiguousarray(rgb_u8)
        h2, w2, _ = rgb_u8.shape
        q_img = QImage(rgb_u8.data, w2, h2, w2 * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self._view.label_classification_result.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            from logic.cnn_logic.cnn_worker import CNN_CLASS_NAMES
            confidence = (
                (self._cnn_probs or [0.0])[CNN_CLASS_NAMES.index(predicted_class)] * 100.0
                if self._cnn_probs and predicted_class in CNN_CLASS_NAMES else 0.0
            )
            final = self._draw_prediction_badge(scaled, predicted_class, confidence)
            self._view.label_classification_result.setStyleSheet(
                "QLabel { background-color: #16161e; border: 2px solid #45475a;"
                " border-radius: 6px; }"
            )
            self._view.label_classification_result.setPixmap(final)

        # ── Status ──────────────────────────────────────────────────
        icons = {"Bleeding": "🔴", "Ischemia": "🔵"}
        icon = icons.get(predicted_class, "🔌")
        fallback_note = " (fallback automático)" if self._unet_fallback_tried else ""
        self._view.set_status(
            f"✅ U-NET {icon} {predicted_class}{fallback_note} → {area_pct:.1f}% de la imagen"
        )
        print(f"  → U-NET final: {predicted_class}{fallback_note} | {area_pct:.1f}% area")

        # ── Persistir para refrescar con filtro HU y para comparar ──────────
        self._unet_seg_mask = seg_mask.copy()
        self._unet_seg_rgb  = rgb.copy()
        self._view.show_compare_button()

    def _on_unet_error(self, message: str) -> None:
        """Maneja errores del worker U-NET."""
        if self._view is None:
            return
        self._view.btn_evaluate.setEnabled(True)
        self._view.set_status("❌ Error en U-NET")
        print(f"  [ERROR U-NET] {message}")
        QMessageBox.critical(self._view, "Error en segmentación U-NET", message)

    # ------------------------------------------------------------------
    # Comparación con máscara de referencia (ground truth)
    # ------------------------------------------------------------------

    def on_compare_mask(self) -> None:
        """
        Abre un explorador para elegir la máscara de referencia (PNG, NPY, etc.)
        y la superpone semitransparente en verde (#a6e3a1, 50% alpha) sobre
        la imagen resultado U-NET en el panel de clasificación.

        El directorio del diálogo se guarda/recupera independientemente del
        diálogo principal de imagen.
        """
        if self._view is None or self._unet_seg_rgb is None:
            return

        # ── Diálogo con path propio ────────────────────────────────────
        last_dir = AppSettings.get_last_mask_dir()
        mask_path, _ = QFileDialog.getOpenFileName(
            self._view,
            "Seleccionar máscara de referencia",
            last_dir,
            "Imágenes (*.png *.jpg *.bmp *.tif *.tiff *.npy);;Todos (*)",
        )
        if not mask_path:
            return
        AppSettings.save_last_mask_dir(mask_path)

        # ── Cargar imagen de referencia en colores nativos ──────────────────
        try:
            from pathlib import Path as _Path
            from PIL import Image as PILImage
            ext = _Path(mask_path).suffix.lower()

            if ext == ".npy":
                # Array binario → mostrar en escala de grises
                raw = np.load(mask_path)
                raw_u8 = (raw.squeeze() * 255).clip(0, 255).astype(np.uint8)
                ref_rgb_arr = np.stack([raw_u8, raw_u8, raw_u8], axis=-1)
            else:
                # PNG/JPG/TIFF: preservar colores nativos  ← sin convert("L")
                pil = PILImage.open(mask_path).convert("RGB")
                ref_rgb_arr = np.asarray(pil).astype(np.uint8)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self._view, "Error al cargar máscara", str(exc)
            )
            return

        # ── Mostrar en el panel IZQUIERDO con sus colores originales ───────
        ref_rgb_arr = np.ascontiguousarray(ref_rgb_arr)
        hr, wr = ref_rgb_arr.shape[:2]
        q_ref = QImage(ref_rgb_arr.data, wr, hr, wr * 3, QImage.Format.Format_RGB888)
        px_ref = QPixmap.fromImage(q_ref)

        if not px_ref.isNull():
            self._view.set_original_image(px_ref)

        # Cobertura aproximada (píxeles no-negro)
        gray = ref_rgb_arr.mean(axis=-1)
        area_ref_pct = 100.0 * float(np.sum(gray > 30)) / max(gray.size, 1)
        self._view.set_status(
            f"✅ Máscara de referencia en panel izq — {area_ref_pct:.1f}% cobertura"
        )
        print(f"  → Máscara referencia cargada: {mask_path} | {area_ref_pct:.1f}% cobertura")

    def _on_prediction_ready(self, pred_mask: np.ndarray) -> None:
        """
        Recibe la máscara de predicción desde el worker y actualiza la vista.

        pred_mask: uint8 (H, W)
            0 = Normal  (no se pinta)
            1 = Bleeding (rojo)
            2 = Ischemia (azul)
        """
        if self._view is None or self._ct is None:
            return

        self._view.btn_evaluate.setEnabled(True)

        # Guardar máscara como estado persistente (para refrescarse con el filtro)
        self._pred_mask = pred_mask

        # Conteo de píxeles por clase
        n_bleeding = int(np.sum(pred_mask == 1))
        n_ischemia = int(np.sum(pred_mask == 2))
        total_brain = int(np.sum(self._main_ctrl.get_brain_mask() > 0))

        if n_bleeding == 0 and n_ischemia == 0:
            self._view.set_status("✅ Predicción completada: Normal")
            self._view.set_classification_result(
                "✅ NORMAL\n\nNo se detectaron regiones de\nSangrado ni Isquemia."
            )
            return

        # ── Renderizar overlay con el filtro HU activo ───────────────────
        self._refresh_prediction_panel()

        lines = []
        if n_bleeding > 0:
            pct_b = 100.0 * n_bleeding / max(total_brain, 1)
            lines.append(f"🔴 Sangrado: {pct_b:.1f}% del cerebro")
        if n_ischemia > 0:
            pct_i = 100.0 * n_ischemia / max(total_brain, 1)
            lines.append(f"🔵 Isquemia: {pct_i:.1f}% del cerebro")

        status = " | ".join(lines)
        self._view.set_status(f"✅ {status}")
        print(f"  → Predicción: {status}")

    def _on_prediction_error(self, message: str) -> None:
        """Maneja errores del worker de predicción."""
        if self._view is None:
            return
        self._view.btn_evaluate.setEnabled(True)
        self._view.set_status(f"❌ Error en predicción")
        self._view.set_classification_result(f"❌ Error:\n{message}")
        print(f"  [ERROR RF] {message}")
        QMessageBox.critical(self._view, "Error en predicción", message)

    # ------------------------------------------------------------------
    # Renderer de overlay de predicción
    # ------------------------------------------------------------------

    @staticmethod
    def _render_prediction_pixmap(
        image: np.ndarray,
        pred_mask: np.ndarray,
        vmin: float = 0.0,
        vmax: float = 80.0,
    ) -> QPixmap | None:
        """
        Combina la imagen CT (ventana HU configurable) con el overlay
        de predicción:
            Rojo  (#e06c75, alpha 0.55) → Bleeding (label 1)
            Azul  (#89b4fa, alpha 0.55) → Ischemia (label 2)

        Args:
            image:     Array 2D float (HU).
            pred_mask: Array uint8 (H, W) con labels 0/1/2.
            vmin:      Límite inferior de la ventana HU.
            vmax:      Límite superior de la ventana HU.
        """
        try:
            span = vmax - vmin
            arr = np.clip(image, vmin, vmax).astype(np.float32)
            arr = (arr - vmin) / span if span > 0 else np.zeros_like(arr)
            rgb = np.stack([arr, arr, arr], axis=-1)

            # Overlay Bleeding — rojo #e06c75
            bleed_color = np.array([0xe0 / 255.0, 0x6c / 255.0, 0x75 / 255.0],
                                   dtype=np.float32)
            bleed_region = pred_mask == 1
            if bleed_region.any():
                rgb[bleed_region] = (
                    0.45 * rgb[bleed_region] + 0.55 * bleed_color
                ).clip(0, 1)

            # Overlay Ischemia — azul #89b4fa
            isch_color = np.array([0x89 / 255.0, 0xb4 / 255.0, 0xfa / 255.0],
                                  dtype=np.float32)
            isch_region = pred_mask == 2
            if isch_region.any():
                rgb[isch_region] = (
                    0.45 * rgb[isch_region] + 0.55 * isch_color
                ).clip(0, 1)

            # Convertir a QImage RGB888
            rgb_u8 = (rgb * 255).clip(0, 255).astype(np.uint8)
            rgb_u8 = np.ascontiguousarray(rgb_u8)
            h, w, _ = rgb_u8.shape
            q_img = QImage(rgb_u8.data, w, h, w * 3, QImage.Format.Format_RGB888)
            px = QPixmap.fromImage(q_img)
            return px if not px.isNull() else None
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] _render_prediction_pixmap: {exc}")
            return None

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _active_image(self) -> np.ndarray | None:
        """Devuelve el array 2D activo según el modo de vista seleccionado."""
        if self._view_mode == _MODE_BRAIN and self._brain_image is not None:
            return self._brain_image
        return self._ct.clean_image if self._ct is not None else None

    def _refresh_display(self) -> None:
        """Regenera ambos paneles con el filtro HU actual."""
        if self._view is None:
            return
        image = self._active_image()
        if image is None:
            return

        # Panel izquierdo: imagen original / cerebro
        if self._hu_filter is not None:
            vmin, vmax = self._hu_filter
            pixmap = self._array_to_pixmap_windowed(image, vmin, vmax)
        else:
            pixmap = self._array_to_pixmap_full_range(image)

        if pixmap:
            self._view.set_original_image(pixmap)

        # Panel derecho: prioridad U-NET > CNN > RF
        self._refresh_unet_panel()
        self._refresh_prediction_panel()
        self._refresh_cnn_panel()

    def _refresh_unet_panel(self) -> None:
        """
        Re-renderiza el panel de resultado U-NET con el filtro HU activo.
        Solo actúa si hay una máscara U-NET almacenada.
        Tiene prioridad: si existe máscara U-NET, no se llama a _refresh_cnn_panel.
        """
        if (
            self._view is None
            or self._ct is None
            or self._unet_seg_mask is None
            or self._cnn_predicted_class is None
        ):
            return

        predicted_class = self._cnn_predicted_class
        overlay_colors = {
            "Bleeding": np.array([0xe0/255.0, 0x6c/255.0, 0x75/255.0], dtype=np.float32),
            "Ischemia": np.array([0x89/255.0, 0xb4/255.0, 0xfa/255.0], dtype=np.float32),
        }
        color = overlay_colors.get(predicted_class, np.array([1.0, 1.0, 0.0], dtype=np.float32))

        image = self._active_image()
        if image is None:
            return

        if self._hu_filter is not None:
            vmin, vmax = self._hu_filter
        else:
            vmin, vmax = 0.0, 80.0

        span = vmax - vmin
        arr = np.clip(image, vmin, vmax).astype(np.float32)
        arr = (arr - vmin) / span if span > 0 else np.zeros_like(arr)
        rgb = np.stack([arr, arr, arr], axis=-1)

        mask = self._unet_seg_mask
        # Reescalar si la imagen cambio de tamaño (ej. cambio de vista)
        h, w = image.shape
        if mask.shape != (h, w):
            from PIL import Image as PILImage
            m_pil = PILImage.fromarray((mask * 255).clip(0,255).astype(np.uint8))
            m_pil = m_pil.resize((w, h), PILImage.NEAREST)
            mask = np.asarray(m_pil).astype(np.float32) / 255.0

        region = mask > 0.5
        if region.any():
            rgb[region] = (0.45 * rgb[region] + 0.55 * color).clip(0, 1)

        # Actualizar también el rgb base para comparar
        self._unet_seg_rgb = rgb.copy()

        rgb_u8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        rgb_u8 = np.ascontiguousarray(rgb_u8)
        h2, w2, _ = rgb_u8.shape
        q_img = QImage(rgb_u8.data, w2, h2, w2 * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        if not pixmap.isNull():
            from logic.cnn_logic.cnn_worker import CNN_CLASS_NAMES
            confidence = (
                (self._cnn_probs or [0.0])[CNN_CLASS_NAMES.index(predicted_class)] * 100.0
                if self._cnn_probs and predicted_class in CNN_CLASS_NAMES else 0.0
            )
            scaled = pixmap.scaled(
                self._view.label_classification_result.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            final = self._draw_prediction_badge(scaled, predicted_class, confidence)
            self._view.label_classification_result.setStyleSheet(
                "QLabel { background-color: #16161e; border: 2px solid #45475a;"
                " border-radius: 6px; }"
            )
            self._view.label_classification_result.setPixmap(final)

    def _refresh_prediction_panel(self) -> None:
        """
        Re-renderiza el panel de resultado con el filtro HU activo.
        Solo actua si hay una predicción almacenada (_pred_mask != None).
        """
        if self._view is None or self._ct is None or self._pred_mask is None:
            return

        # Determinar rango de representación
        if self._hu_filter is not None:
            vmin, vmax = self._hu_filter
        else:
            vmin, vmax = 0.0, 80.0   # ventana cerebral por defecto

        pixmap = self._render_prediction_pixmap(
            self._ct.clean_image, self._pred_mask, vmin=vmin, vmax=vmax
        )
        if pixmap:
            self._view.label_classification_result.setStyleSheet(
                "QLabel { background-color: #16161e; border: 2px solid #45475a;"
                " border-radius: 6px; }"
            )
            self._view.label_classification_result.setPixmap(
                pixmap.scaled(
                    self._view.label_classification_result.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _refresh_cnn_panel(self) -> None:
        """
        Re-renderiza el panel de resultado CNN con el filtro HU activo.
        Solo actúa si hay un resultado CNN almacenado.
        """
        if (
            self._view is None
            or self._ct is None
            or self._cnn_predicted_class is None
            or self._cnn_probs is None
        ):
            return

        from logic.cnn_logic.cnn_worker import CNN_CLASS_NAMES
        predicted_class = self._cnn_predicted_class
        confidence = self._cnn_probs[CNN_CLASS_NAMES.index(predicted_class)] * 100.0

        image = self._active_image()
        if image is None:
            return

        if self._hu_filter is not None:
            vmin, vmax = self._hu_filter
            pixmap = self._array_to_pixmap_windowed(image, vmin, vmax)
        else:
            pixmap = self._array_to_pixmap_full_range(image)

        if pixmap:
            scaled = pixmap.scaled(
                self._view.label_classification_result.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            final = self._draw_prediction_badge(scaled, predicted_class, confidence)
            self._view.label_classification_result.setStyleSheet(
                "QLabel { background-color: #16161e; border: 2px solid #45475a;"
                " border-radius: 6px; }"
            )
            self._view.label_classification_result.setPixmap(final)


    # ── Conversores numpy → QPixmap ────────────────────────────────────

    @staticmethod
    def _array_to_pixmap_full_range(image: np.ndarray) -> QPixmap | None:
        """Normaliza al rango completo del array → QPixmap."""
        try:
            arr = image.astype(np.float32)
            a_min, a_max = arr.min(), arr.max()
            if a_max > a_min:
                arr = ((arr - a_min) / (a_max - a_min) * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
            arr = np.ascontiguousarray(arr)
            h, w = arr.shape
            q_img = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
            px = QPixmap.fromImage(q_img)
            return px if not px.isNull() else None
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] _array_to_pixmap_full_range: {exc}")
            return None

    @staticmethod
    def _array_to_pixmap_windowed(
        image: np.ndarray, vmin: float, vmax: float
    ) -> QPixmap | None:
        """Aplica ventana HU [vmin, vmax] → QPixmap."""
        try:
            arr = np.clip(image, vmin, vmax).astype(np.float32)
            span = vmax - vmin
            arr = ((arr - vmin) / span * 255).astype(np.uint8) if span > 0 \
                else np.zeros_like(arr, dtype=np.uint8)
            arr = np.ascontiguousarray(arr)
            h, w = arr.shape
            q_img = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
            px = QPixmap.fromImage(q_img)
            return px if not px.isNull() else None
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] _array_to_pixmap_windowed: {exc}")
            return None
