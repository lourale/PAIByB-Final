import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

class NoiseAnalyzer:
    def __init__(self, dataset):
        self.dataset = dataset

    def LSDSM_dataset(self, n_plots=0, window_size=5):
        """
        Aplica el método LSDSM a cada imagen del dataset y devuelve un nuevo dataset con los mapas de desviación estándar local.
        Si plot=True, muestra ejemplos de los mapas de ruido para cada clase.
        """
        lsdsms = {}
        
        for class_name, data in self.dataset.items():
            if data is not None:
                dicoms = data['dicom']
                lsdsms[class_name] = []
                for img in dicoms:
                    mapa_std = self.LSDSM_image(img, plot=False, window_size=window_size)
                    lsdsms[class_name].append(mapa_std)
        
        # ploteo solo n_plots ejemplos por clase para no saturar
        if n_plots != 0:
            for class_name, lsdsms_list in lsdsms.items():
                for i in range(min(n_plots, len(lsdsms_list))):
                    print(f"Clase: {class_name}, Imagen {i+1}/{min(n_plots, len(lsdsms_list))}")
                    self.LSDSM_image(self.dataset[class_name]['dicom'][i], plot=True, window_size=window_size)
                    
        return lsdsms

    def LSDSM_image(self, image, plot = False, window_size=5):
        """
        Calcula el mapa de desviación estándar local (LSDSM, por sus siglas en inglés) para una imagen dada.
        Devuelve un mapa de desviación estándar local.
        """        
        # Calcular media y media de los cuadrados localmente
        c1 = uniform_filter(image, window_size, mode='reflect')
        c2 = uniform_filter(image**2, window_size, mode='reflect')
    
        # La varianza local es E[X^2] - (E[X])^2
        # Usamos max(0) para evitar raíces negativas por errores de redondeo de punto flotante
        mapa_std = np.sqrt(np.maximum(c2 - c1**2, 0))
        
        if plot:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            axs[0].imshow(image, cmap='gray', vmin=0, vmax=100)
            axs[0].set_title('Imagen Original')
            axs[0].axis('off')
            
            # Mostrar el mapa de ruido (desviación estándar)
            im1 = axs[1].imshow(mapa_std, cmap='magma', vmin=0, vmax=100)
            axs[1].set_title(f'Mapa de Ruido (Ventana {window_size}x{window_size})')
            axs[1].axis('off')
            plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
            
            # Histograma
            axs[2].hist(image.ravel(), bins=100, color='gray', alpha=0.7)
            axs[2].set_title('Histograma de Intensidades')
            axs[2].set_yscale('log') # Escala logarítmica para ver bien las colas de ruido
            
            plt.tight_layout()
            plt.show()
        
        return mapa_std