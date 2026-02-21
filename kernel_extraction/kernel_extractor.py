import numpy as np
import random

class KernelExtractor:
    def __init__(self, dicom_img, brain_mask, overlay=None, patch_size=32):
        self.dicom = dicom_img
        self.brain_mask = brain_mask
        self.overlay = overlay
        self.patch_size = patch_size
        self.half = patch_size // 2

    def is_valid_patch(self, y, x):
        """
        Valida usando la MÁSCARA del cerebro para asegurar que 
        estamos dentro del área de interés.
        """
        if y - self.half < 0 or y + self.half >= self.dicom.shape[0]: return False
        if x - self.half < 0 or x + self.half >= self.dicom.shape[1]: return False
        
        patch_mask = self.brain_mask[y-self.half : y+self.half, x-self.half : x+self.half]
        
        # Exigimos que al menos el 50% del kernel sea tejido (valores > 0 en la máscara)
        if np.count_nonzero(patch_mask) < (self.patch_size * self.patch_size * 0.5):
            return False
        return True

    def _get_clean_patch(self, y, x):
        """
        Extrae el parche de la imagen DICOM y lo multiplica por la máscara
        para eliminar cualquier resto de cráneo que se cuele en los bordes.
        """
        dicom_patch = self.dicom[y-self.half : y+self.half, x-self.half : x+self.half]
        mask_patch = self.brain_mask[y-self.half : y+self.half, x-self.half : x+self.half]
        
        # Convertimos la máscara a 1s y 0s lógicos y multiplicamos
        binary_mask = (mask_patch > 0).astype(dicom_patch.dtype)
        return dicom_patch * binary_mask

    def extract_pathological(self, max_patches=3):
        patches = []
        if self.overlay is None: return patches
        
        # Coordenadas de la lesión
        y_idx, x_idx = np.where(self.overlay > 0)
        if len(y_idx) == 0: return patches

        coords = list(zip(y_idx, x_idx))
        random.shuffle(coords)
        
        for y, x in coords:
            if self.is_valid_patch(y, x):
                patches.append(self._get_clean_patch(y, x))
                if len(patches) >= max_patches:
                    break
        return patches

    def extract_normal(self, num_patches=1):
        patches = []
        
        # Coordenadas del cerebro sano
        y_idx, x_idx = np.where(self.brain_mask > 0)
        if len(y_idx) == 0: return patches
        
        coords = list(zip(y_idx, x_idx))
        random.shuffle(coords)

        for y, x in coords:
            if self.is_valid_patch(y, x):
                patches.append(self._get_clean_patch(y, x))
                if len(patches) >= num_patches:
                    break
        return patches

class KernelExtractorManager:
    def __init__(self, dataset, patch_size=32):
        self.dataset = dataset
        self.patch_size = patch_size
        
    def extract_all(self, patches_per_lesion=3):
        results = {'Bleeding': [], 'Ischemia': [], 'Normal': []}
        print(f"Iniciando extracción de kernels ({self.patch_size}x{self.patch_size})...")
        
        # 1. Extraer todo el tejido enfermo disponible
        for cls in ['Bleeding', 'Ischemia']:
            if cls not in self.dataset or self.dataset[cls] is None: 
                continue
                
            print(f"  -> Procesando {cls}...")
            for dicom, brain_mask, overlay in zip(self.dataset[cls]['dicom'], 
                                                  self.dataset[cls]['brain'], 
                                                  self.dataset[cls]['overlay']):
                
                extractor = KernelExtractor(dicom, brain_mask, overlay, self.patch_size)
                patches = extractor.extract_pathological(max_patches=patches_per_lesion)
                results[cls].extend(patches)
                
        # 2. Balancear patologías (1:1 entre Bleeding e Ischemia)
        count_bleeding = len(results['Bleeding'])
        count_ischemia = len(results['Ischemia'])
        
        # Encontramos la clase minoritaria
        min_pathology_count = min(count_bleeding, count_ischemia)
        
        if min_pathology_count > 0:
            print(f"\n  -> Balanceando patologías al mínimo común: {min_pathology_count} parches.")
            
            # Mezclamos y recortamos Bleeding
            random.shuffle(results['Bleeding'])
            results['Bleeding'] = results['Bleeding'][:min_pathology_count]
            
            # Mezclamos y recortamos Ischemia
            random.shuffle(results['Ischemia'])
            results['Ischemia'] = results['Ischemia'][:min_pathology_count]
        else:
            print("\n  -> Advertencia: No se encontraron suficientes parches patológicos.")

        # 3. Extraer tejido sano (1:1 con la patología minoritaria)
        if 'Normal' in self.dataset and self.dataset['Normal'] is not None and min_pathology_count > 0:
            normal_images = len(self.dataset['Normal']['dicom'])
            
            if normal_images > 0:
                # Calculamos cuántos parches sanos necesitamos por cada imagen normal
                patches_per_normal = max(1, min_pathology_count // normal_images + 1)
                print(f"  -> Procesando Normal (Buscando {min_pathology_count} parches para balance 1:1:1)...")
                
                for dicom, brain_mask in zip(self.dataset['Normal']['dicom'], self.dataset['Normal']['brain']):
                    extractor = KernelExtractor(dicom, brain_mask, overlay=None, patch_size=self.patch_size)
                    patches = extractor.extract_normal(num_patches=patches_per_normal)
                    results['Normal'].extend(patches)
                    
                # Recorte final exacto para la clase Normal
                random.shuffle(results['Normal'])
                results['Normal'] = results['Normal'][:min_pathology_count]

        # 4. Convertir a diccionarios Numpy
        final_dict = {}
        print("-" * 50)
        print("RESUMEN DE KERNELS EXTRAÍDOS (BALANCE ESTRICTO 1:1:1):")
        for cls in results:
            final_dict[cls] = np.array(results[cls])
            print(f"[{cls}]: {final_dict[cls].shape[0]} parches de tamaño {self.patch_size}x{self.patch_size}")
        print("-" * 50)
            
        return final_dict