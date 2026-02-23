import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CTProcessor:
    def __init__(self, image_array, corner_size=30):
        # ... (Tu init y funciones internas siguen igual) ...
        self.raw_image = image_array
        self.corner_size = corner_size
        self.was_processed = False
        
        self.raw_hist_data = self._compute_histogram(self.raw_image)
        self.corner_img, self.corner_hist_data = self._analyze_corner()
        self.clean_image, self.clean_hist_data = self._apply_advanced_correction()

    def _compute_histogram(self, img):
        return img.flatten()

    def _analyze_corner(self):
        corner = self.raw_image[0:self.corner_size, 0:self.corner_size]
        return corner, corner.flatten()

    def _apply_advanced_correction(self):
        img_copy = self.raw_image.copy().astype(np.float32)
        corner = img_copy[0:self.corner_size, 0:self.corner_size]

        min_corner = np.min(corner)
        std_corner = np.std(corner)

        # 1. ¿Hay ruido en el corner? (Esto nos dice qué es el "Primer Pico")
        if std_corner < 1.0:
            # --- CASO A: ES UNA MÁSCARA ARTIFICIAL ---
            # El primer pico es la máscara. El "aire real" es el segundo pico.
            
            # Saltamos el primer pico (la máscara y cualquier ruido pegado a ella)
            mask_value = min_corner
            non_mask_pixels = img_copy[img_copy > (mask_value + 5)]

            if len(non_mask_pixels) > 0:
                # Encontramos dónde empieza el aire real (usamos percentil 0.1 para ignorar píxeles rotos)
                real_air_value = np.percentile(non_mask_pixels, 0.1)

                # Fijamos este aire a -1000, desplazando todo el histograma
                offset = real_air_value + 1000
                img_copy -= offset

                self.status_msg = f"Corregido (Máscara saltada. Offset: -{int(offset)})"
            else:
                self.status_msg = "Error: Imagen sin tejido"

        else:
            # --- CASO B: ES RUIDO NATURAL ---
            # El primer pico YA ES el aire real (ej: el 7191).
            
            # Agarramos el valor del aire (la mediana del corner es súper estable)
            real_air_value = np.median(corner)

            # Fijamos el aire a -1000, desplazando todo el histograma
            offset = real_air_value + 1000
            img_copy -= offset

            # Ahora que el aire está anclado perfecto en -1000, cumplimos tu regla:
            # "a ese aire le resto 1000". Tomamos todo lo que ronda el -1000 y lo hundimos.
            mask_air = (img_copy >= -1050) & (img_copy <= -950)
            img_copy[mask_air] -= 1000

            self.status_msg = f"Corregido (Aire alineado y Abismo creado. Offset: -{int(offset)})"

        return img_copy, img_copy.flatten()

    def plot(self):
        """
        Plotea 2 filas:
        Fila 1: Análisis técnico (Cruda, Histogramas, Corregida Full).
        Fila 2: Visualización clínica (Filtros progresivos 0-20, 0-40, 0-60, 0-80, 0-100).
        """
        # Grilla unificada de 2 filas y 5 columnas
        fig = plt.figure(figsize=(20, 10))
        
        # --- FILA 1: ANÁLISIS TÉCNICO (5 columnas) ---
        ax1 = plt.subplot(2, 5, 1)
        ax2 = plt.subplot(2, 5, 2)
        ax3 = plt.subplot(2, 5, 3)
        ax4 = plt.subplot(2, 5, 4)
        ax5 = plt.subplot(2, 5, 5)

        status = "Corregido" if self.was_processed else "Sin Cambios"
        fig.suptitle(f"Análisis de Procesamiento CT | Estado: {status}", fontsize=16)

        # 1. IMAGEN CRUDA
        ax1.imshow(self.raw_image, cmap='gray')
        ax1.set_title("1. Imagen Cruda")
        ax1.axis('off')
        rect = patches.Rectangle((0, 0), self.corner_size, self.corner_size, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)

        # 2. HISTOGRAMA CRUDO
        ax2.hist(self.raw_hist_data, bins=100, color='gray', log=True)
        ax2.set_title("2. Histograma Crudo")
        ax2.grid(True, alpha=0.3)

        # 3. RUIDO ESQUINA
        ax3.hist(self.corner_hist_data, bins='auto', color='red', alpha=0.7)
        ax3.set_title(f"3. Ruido Esquina\nMedia: {np.mean(self.corner_img):.1f}")
        ax3.grid(True, alpha=0.3)

        # 4. HISTOGRAMA CORREGIDO
        ax4.hist(self.clean_hist_data, bins=100, color='teal', log=True)
        ax4.set_title("4. Histograma Corregido")
        ax4.grid(True, alpha=0.3)
        ax4.axvline(-1000, color='red', linestyle='--', alpha=0.5, label='Aire Std')
        ax4.axvline(0, color='orange', linestyle=':', alpha=0.5, label='Tejido')
        ax4.legend(fontsize='small')

        # 5. IMAGEN CORREGIDA (FULL RANGE - SIN FILTRO)
        ax5.imshow(self.clean_image, cmap='gray') 
        ax5.set_title("5. Corregida (Rango Completo)\nSin Ventana")
        ax5.axis('off')

        # --- FILA 2: FILTROS CLÍNICOS (5 columnas alineadas) ---
        filtros = [(0, 20), (0, 40), (0, 60), (0, 80), (0, 100)]
        
        for i, (vmin, vmax) in enumerate(filtros):
            # Posición: Fila 2, Columna i+1. 
            # Como la primera fila ocupa del 1 al 5, la segunda fila empieza en 6.
            ax_filt = plt.subplot(2, 5, 6 + i) 
            
            ax_filt.imshow(self.clean_image, cmap='gray', vmin=vmin, vmax=vmax)
            ax_filt.set_title(f"Ventana: {vmin} - {vmax} HU")
            ax_filt.axis('off')

        plt.tight_layout()
        plt.show()

class CTManager:
    @staticmethod
    def process_batch(data):
        """
        Recorre el diccionario anidado, procesa cada imagen DICOM 
        usando CTProcessor y reconstruye la estructura manteniendo los overlays.
        """
        processed_data = {}
        
        for cls, content in data.items():
            # Si la clase no tiene datos (es None), la saltamos y mantenemos el None
            if content is None:
                processed_data[cls] = None
                continue
                
            processed_data[cls] = {}
            
            # Procesamos SOLAMENTE el array que está bajo la llave 'dicom'
            processed_data[cls]['dicom'] = np.array([CTProcessor(img) for img in content['dicom']])
            
            # Copiamos el array de overlays tal cual estaba (no requiere procesamiento HU)
            processed_data[cls]['overlay'] = content['overlay']
            
        return processed_data

    @staticmethod
    def process_batch_clean_image(data):
        """
        Recorre el diccionario anidado, procesa cada imagen DICOM 
        y extrae únicamente el 'clean_image', manteniendo la estructura y los overlays.
        """
        processed_data = {}
        
        for cls, content in data.items():
            # Si la clase no tiene datos (es None), la saltamos y mantenemos el None
            if content is None:
                processed_data[cls] = None
                continue
                
            processed_data[cls] = {}
            
            # Procesamos y extraemos SOLO el clean_image de los DICOMs
            processed_data[cls]['dicom'] = np.array([CTProcessor(img).clean_image for img in content['dicom']])
            
            # Copiamos el array de overlays tal cual estaba
            processed_data[cls]['overlay'] = content['overlay']
            
        return processed_data