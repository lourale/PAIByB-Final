import numpy as np
from scipy.stats import skew, kurtosis
from scipy.ndimage import convolve
from skimage.feature import graycomatrix, graycoprops
import pandas as pd

class FeatureExtract:
    def __init__(self, patch):
        """
        Recibe un kernel de 32x32.
        Asumimos que los valores están en Unidades Hounsfield (HU).
        """
        self.patch = patch.astype(np.float32)
        
        # Para la textura (GLCM) necesitamos enteros positivos. 
        # Recortamos a la ventana cerebral (0-100 HU) y discretizamos a 32 niveles.
        patch_clipped = np.clip(self.patch, 0, 100)
        self.patch_binned = np.digitize(patch_clipped, bins=np.linspace(0, 100, 32)) - 1
        self.patch_binned = self.patch_binned.astype(np.uint8)

    def _calc_first_order(self):
        """
        Extrae estadísticas globales. 
        Reflejan la atenuación basal y la densidad del tejido.
        """
        # Aplanamos el parche a un vector 1D para estadísticas globales
        flat_patch = self.patch.flatten()
        
        # 1. Atenuación Media [cite: 8, 24]
        mean_hu = np.mean(flat_patch)
        
        # 2. Energía de Primer Orden: Suma de los píxeles al cuadrado[cite: 144].
        # Altamente predictivo para identificar hematomas compactos y densos[cite: 147].
        energy_1st = np.sum(flat_patch ** 2)
        
        # 3. Asimetría (Skewness) y 4. Curtosis (Kurtosis)[cite: 151].
        # Desviaciones de la campana de Gauss delatan procesos asimétricos.
        skew_val = skew(flat_patch)
        kurt_val = kurtosis(flat_patch)
        
        return {
            "mean_hu": mean_hu,
            "energy_1st": energy_1st,
            "skewness": skew_val if not np.isnan(skew_val) else 0.0,
            "kurtosis": kurt_val if not np.isnan(kurt_val) else 0.0,
            "variance": np.var(flat_patch)
        }

    def _calc_glcm_texture(self):
        """
        Extrae características de Segundo Orden (Textura Espacial)[cite: 157].
        Mide la heterogeneidad y fragmentación celular[cite: 159].
        """
        # Calculamos la matriz iterando en 1 pixel de distancia, en 4 ángulos básicos
        # (0, 45, 90, 135 grados) [cite: 161]
        glcm = graycomatrix(self.patch_binned, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=32, symmetric=True, normed=True)
        
        # Extraemos las propiedades promediando los 4 ángulos
        # Homogeneidad baja indica lisis celular [cite: 163]
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        # Contraste alto delata aspereza microscópica pre-hemorrágica [cite: 168]
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))
        energy_glcm = np.mean(graycoprops(glcm, 'energy'))
        
        # Entropía de GLCM (calculada manualmente a partir de la matriz normalizada)
        # Alta entropía indica el mosaico caótico de necrosis por licuefacción [cite: 164]
        glcm_nonzero = glcm[glcm > 0]
        entropy = -np.sum(glcm_nonzero * np.log2(glcm_nonzero))
        
        return {
            "glcm_homogeneity": homogeneity,
            "glcm_contrast": contrast,
            "glcm_correlation": correlation,
            "glcm_energy": energy_glcm,
            "glcm_entropy": entropy
        }

    def _calc_ldop_gradients(self):
        """
        Descriptores Locales Especializados (Aproximación a LDOP)[cite: 220].
        Calcula derivadas direccionales cruzadas (gradientes) para aislar 
        anomalías sutiles del ruido de fondo[cite: 223, 225].
        """
        # Kernels de convolución para detectar cambios en distintas direcciones
        k_0 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # Horizontal
        k_90 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) # Vertical
        k_45 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]) # Diagonal 1
        k_135 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]) # Diagonal 2
        
        # Aplicamos convolución sobre el parche crudo
        grad_0 = convolve(self.patch, k_0)
        grad_90 = convolve(self.patch, k_90)
        grad_45 = convolve(self.patch, k_45)
        grad_135 = convolve(self.patch, k_135)
        
        # Promediamos la magnitud absoluta de estos micro-gradientes [cite: 229]
        return {
            "ldop_grad_0_mean": np.mean(np.abs(grad_0)),
            "ldop_grad_90_mean": np.mean(np.abs(grad_90)),
            "ldop_grad_45_mean": np.mean(np.abs(grad_45)),
            "ldop_grad_135_mean": np.mean(np.abs(grad_135))
        }

    def analyze(self):
        """
        Ejecuta todas las funciones internas y consolida los vectores en 
        un único diccionario unidimensional.
        """
        features = {}
        features.update(self._calc_first_order())
        features.update(self._calc_glcm_texture())
        features.update(self._calc_ldop_gradients())
        return features

class FeatureExtractManager:
    def __init__(self, kernels_dict):
        """
        Recibe el diccionario balanceado generado por KernelExtractorManager.
        Ej: {'Normal': [patch1, ...], 'Bleeding': [...], 'Ischemia': [...]}
        """
        self.kernels_dict = kernels_dict

    def extract_all_to_dataframe(self):
        """
        Itera por cada parche de cada clase, extrae sus métricas y ensambla
        una estructura tabular ideal para Machine Learning (Pandas DataFrame).
        """
        all_features = []
        
        print("Iniciando extracción radiómica (First-Order, GLCM, LDOP-Gradients)...")
        
        for cls, patches in self.kernels_dict.items():
            if patches is None or len(patches) == 0:
                continue
                
            print(f"  -> Minando características para clase: {cls} ({len(patches)} parches)")
            
            for patch in patches:
                # Instanciamos el extractor individual
                extractor = FeatureExtract(patch)
                # Obtenemos el diccionario con los números calculados
                patch_features = extractor.analyze()
                
                # Le inyectamos la etiqueta vital para que la IA sepa qué es
                patch_features['label'] = cls
                
                # Lo guardamos en nuestra lista maestra
                all_features.append(patch_features)
                
        # Convertimos la lista de diccionarios en un DataFrame tabular
        df = pd.DataFrame(all_features)
        
        print("-" * 50)
        print("EXTRACCIÓN COMPLETADA.")
        print(f"Dataset estructurado final: {df.shape[0]} filas (ejemplos) x {df.shape[1]} columnas (features + label)")
        print("-" * 50)
        
        return df

# --- EJEMPLO DE USO ---
# Asumiendo que 'kernels_dataset' es el diccionario que obtuviste en el paso anterior:
# feature_manager = FeatureExtractManager(kernels_dataset)
# df_features = feature_manager.extract_all_to_dataframe()

# print(df_features.head()) # Para ver las primeras 5 filas de tus datos