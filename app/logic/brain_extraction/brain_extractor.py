import cv2
import numpy as np
import matplotlib.pyplot as plt

class BrainExtractor:
    def __init__(self, image):
        self.image = image

    def initial_thresholding(self, img):
        # Definimos un rango amplio que incluya el cerebro pero excluya el hueso más denso y el aire.
        # Rango típico: mayor que aire (-100) y menor que hueso (250).
        mask_bool = (img > -100) & (img < 250)
        # convertimos a unit8 (0 a 255) porque OpenCV lo necesita así
        initial_mask = mask_bool.astype(np.uint8) * 255
        
        return initial_mask
    
    def morphological_erosion(self, initial_mask):
        # Usamos un kernel (un cuadradito de 5x5 pixeles) para "limar" los bordes.
        # Esto rompe las finas conexiones entre el cerebro y el cráneo/ojos.
        kernel = np.ones((5, 5), np.uint8)
        # Iterations=1 suele ser suficiente, a veces 2 si el hueso está muy pegado.
        eroded_mask = cv2.erode(initial_mask, kernel, iterations=1)
        return eroded_mask, kernel
    
    def largest_connected_component(self, eroded_mask, initial_mask):
        # Esta función encuentra todas las "islas" de pixeles blancos.
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(eroded_mask)

        # Buscamos cuál es la isla más grande (ignorando el fondo, que es label 0)
        max_area = 0
        max_label = 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > max_area:
                max_area = stats[i, cv2.CC_STAT_AREA]
                max_label = i
                
        # Creamos una nueva máscara solo con esa isla ganadora
        lcc_mask = np.zeros_like(initial_mask)
        # Si encontramos algo (por seguridad), lo pintamos de blanco
        if max_label > 0:
            lcc_mask[labels_im == max_label] = 255
            
        return lcc_mask
    
    def dilate_and_fill(self, lcc_mask, kernel):
        # Primero, dilatamos para recuperar el volumen que perdimos en la erosión.
        dilated_mask = cv2.dilate(lcc_mask, kernel, iterations=2) # Un poco más de dilatación para asegurar bordes

        # Segundo, rellenamos agujeros internos (ej. ventrículos oscuros o isquemias).
        # La forma robusta en OpenCV es encontrar el contorno externo y rellenarlo.
        final_mask = np.zeros_like(dilated_mask)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Tomamos el contorno más grande (el borde exterior del cerebro)
            c = max(contours, key=cv2.contourArea)
            # Dibujamos ese contorno relleno (thickness=cv2.FILLED)
            cv2.drawContours(final_mask, [c], 0, 255, thickness=cv2.FILLED)
        else:
            final_mask = dilated_mask # Fallback por si acaso
            
        return dilated_mask, final_mask
    
    def final_application(self, img, final_mask):
        # Normalizamos la máscara a 0.0 y 1.0 para multiplicar
        final_mask_norm = final_mask.astype(np.float32) / 255.0
        # Multiplicamos la imagen original por la máscara. Lo de afuera se vuelve 0.
        skull_stripped_image = img * final_mask_norm
        return skull_stripped_image
    
    def extract_brain(self, plot = True):
        initial_mask = self.initial_thresholding(self.image)
        eroded_mask, kernel = self.morphological_erosion(initial_mask)
        lcc_mask = self.largest_connected_component(eroded_mask, initial_mask)
        dilated_mask, final_mask = self.dilate_and_fill(lcc_mask, kernel)
        skull_stripped_image = self.final_application(self.image, final_mask)
        
        if plot:
            self.plot_steps(initial_mask, eroded_mask, lcc_mask, dilated_mask, final_mask, skull_stripped_image)
        
        return skull_stripped_image, final_mask

    def plot_steps(self, initial_mask, eroded_mask, lcc_mask, dilated_mask, final_mask, skull_stripped_image):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Validación de Pasos de Skull Stripping", fontsize=16)

        # Fila 1
        axes[0,0].imshow(initial_mask, cmap='gray')
        axes[0,0].set_title("1. Threshold Inicial (-100 a 250 HU)\n(Nótese el cráneo pegado)")
        
        axes[0,1].imshow(eroded_mask, cmap='gray')
        axes[0,1].set_title("2. Erosión\n(Se rompen los puentes)")
        
        axes[0,2].imshow(lcc_mask, cmap='gray')
        axes[0,2].set_title("3. Componente Más Grande\n(Solo queda el cerebro erosionado)")

        # Fila 2
        axes[1,0].imshow(dilated_mask, cmap='gray')
        axes[1,0].set_title("4. Dilatación\n(Se recupera el tamaño)")
        
        axes[1,1].imshow(final_mask, cmap='gray')
        axes[1,1].set_title("5. Máscara Final Rellena\n(Sin agujeros internos)")
        
        # Resultado final: Usamos una ventana de cerebro (0 a 80) para ver bien el tejido
        axes[1,2].imshow(skull_stripped_image, cmap='gray', vmin=0, vmax=80)
        axes[1,2].set_title("RESULTADO FINAL (Aplicado)\n(Ventana Clínica 0-80 HU)")

        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')
        plt.tight_layout()
        plt.show()

class BrainExtractorManager:
    def __init__(self, dataset):
        """
        Inicializa el manager con el dataset procesado (idealmente las imágenes limpias
        salidas de CTManager.process_batch_clean_image).
        """
        self.dataset = dataset

    def extract_all_brains(self, plot_examples=False):
        """
        Itera sobre el dataset, aplica BrainExtractor a cada imagen DICOM y 
        añade el resultado bajo la llave 'brain'.
        
        Args:
            plot_examples (bool): Si es True, plotea el proceso para la PRIMERA 
                                  imagen de cada clase a modo de validación.
                                  
        Returns:
            dict: El dataset actualizado con la estructura {'dicom': ..., 'overlay': ..., 'brain': ...}
        """
        processed_data = {}
        
        print("Iniciando extracción de cerebros...")
        
        for cls, content in self.dataset.items():
            if content is None:
                processed_data[cls] = None
                continue
                
            print(f"  -> Procesando clase: {cls} ({len(content['dicom'])} imágenes)")
            processed_data[cls] = {}
            
            # Copiamos dicom y overlay para no perderlos
            processed_data[cls]['dicom'] = content['dicom']
            processed_data[cls]['overlay'] = content['overlay']
            
            brains = []
            
            for idx, img in enumerate(content['dicom']):
                # Instanciamos el extractor con la imagen actual
                extractor = BrainExtractor(img)
                
                # Solo ploteamos si nos lo piden Y si es la primera imagen de la clase
                should_plot = plot_examples and (idx == 0)
                
                # Obtenemos la imagen extraída (ignoramos la máscara final que devuelve)
                _, stripped_mask = extractor.extract_brain(plot=should_plot)
                
                brains.append(stripped_mask)
                
            # Convertimos la lista de cerebros a numpy array y la guardamos
            processed_data[cls]['brain'] = np.array(brains)
            
        print("Extracción completada.")
        return processed_data