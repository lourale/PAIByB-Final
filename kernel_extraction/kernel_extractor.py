import numpy as np
import random
from scipy import ndimage

PAD = 6
MIN = 4
LARGE_THRESH = 20    # Si el alto o ancho supera esto, se fragmenta
SUB_PATCH_SIZE = 15  # Tamaño de los sub-kernels a extraer (ej. 15x15)

class KernelExtractor:
    def __init__(self, dicom_img, brain_mask, overlay=None):
        self.dicom = dicom_img
        self.brain_mask = brain_mask
        self.overlay = overlay

    def is_valid_patch(self, y_min, y_max, x_min, x_max):
        if (y_max - y_min) < MIN or (x_max - x_min) < MIN:
            return False

        if y_min < 0 or y_max > self.dicom.shape[0]: return False
        if x_min < 0 or x_max > self.dicom.shape[1]: return False
        
        patch_mask = self.brain_mask[y_min:y_max, x_min:x_max]
        area = (y_max - y_min) * (x_max - x_min)
        
        if np.count_nonzero(patch_mask) < (area * 0.90):
            return False
            
        dicom_patch = self.dicom[y_min:y_max, x_min:x_max]
        
        if np.max(dicom_patch) > 120 or np.min(dicom_patch) < -50:
            return False
            
        return True

    def _get_clean_patch(self, y_min, y_max, x_min, x_max):
        dicom_patch = self.dicom[y_min:y_max, x_min:x_max]
        mask_patch = self.brain_mask[y_min:y_max, x_min:x_max]
        binary_mask = (mask_patch > 0).astype(dicom_patch.dtype)
        return dicom_patch * binary_mask

    def extract_pathological(self, pad=PAD, boost_multiplier=1):
        """
        boost_multiplier: Permite extraer más parches solapados para clases minoritarias.
        """
        patches = []
        if self.overlay is None: return patches
        
        dilated_overlay = ndimage.binary_dilation(self.overlay > 0, iterations=pad)
        labeled_overlay, num_features = ndimage.label(dilated_overlay)
        if num_features == 0: return patches
        
        slices = ndimage.find_objects(labeled_overlay)
        
        for i, lesion_slice in enumerate(slices):
            label_idx = i + 1 
            
            y_min, y_max = lesion_slice[0].start, lesion_slice[0].stop
            x_min, x_max = lesion_slice[1].start, lesion_slice[1].stop
            
            h = y_max - y_min
            w = x_max - x_min
            
            if h >= LARGE_THRESH or w >= LARGE_THRESH:
                lesion_mask = (labeled_overlay == label_idx)
                
                # Erosión adaptativa: si la lesión es enorme, erosionamos más para definir el centro
                erosion_iters = max(3, min(h, w) // 6)
                internal_mask = ndimage.binary_erosion(lesion_mask, iterations=erosion_iters)
                border_mask = lesion_mask ^ internal_mask 
                
                int_y, int_x = np.where(internal_mask)
                border_y, border_x = np.where(border_mask)
                
                # ---------------------------------------------------------
                # MUESTREO PROPORCIONAL AL ÁREA TISULAR
                # ---------------------------------------------------------
                area_int = len(int_y)
                area_border = len(border_y)
                patch_area = SUB_PATCH_SIZE ** 2
                
                # Calculamos cuántos parches caben físicamente y le sumamos un porcentaje de solapamiento
                # Aseguramos al menos 2 del centro y 3 de los bordes.
                num_int = max(2, int((area_int / patch_area) * 1.5)) * boost_multiplier
                num_border = max(3, int((area_border / patch_area) * 2.0)) * boost_multiplier
                
                half_sub = SUB_PATCH_SIZE // 2
                
                def sample_and_extract(y_coords, x_coords, num_samples):
                    if len(y_coords) == 0: return
                    coords = list(zip(y_coords, x_coords))
                    # Mezclamos para aleatoriedad y evitar tomar parches secuenciales redundantes
                    random.shuffle(coords)
                    
                    extracted = 0
                    for cy, cx in coords:
                        sy_min, sy_max = cy - half_sub, cy + (SUB_PATCH_SIZE - half_sub)
                        sx_min, sx_max = cx - half_sub, cx + (SUB_PATCH_SIZE - half_sub)
                        
                        if self.is_valid_patch(sy_min, sy_max, sx_min, sx_max):
                            patches.append(self._get_clean_patch(sy_min, sy_max, sx_min, sx_max))
                            extracted += 1
                            if extracted >= num_samples:
                                break
                                
                sample_and_extract(int_y, int_x, num_samples=num_int)
                sample_and_extract(border_y, border_x, num_samples=num_border)
                
            else:
                if self.is_valid_patch(y_min, y_max, x_min, x_max):
                    patches.append(self._get_clean_patch(y_min, y_max, x_min, x_max))
                
        return patches

    def extract_normal(self, target_shapes):
        patches = []
        y_idx, x_idx = np.where(self.brain_mask > 0)
        
        if len(y_idx) == 0 or not target_shapes: 
            return patches
            
        coords = list(zip(y_idx, x_idx))
        
        for shape in target_shapes:
            h, w = shape
            half_h, half_w = h // 2, w // 2
            
            random.shuffle(coords)
            patch_found = False
            
            for y, x in coords:
                y_min, y_max = y - half_h, y + (h - half_h)
                x_min, x_max = x - half_w, x + (w - half_w)
                
                if self.is_valid_patch(y_min, y_max, x_min, x_max):
                    patches.append(self._get_clean_patch(y_min, y_max, x_min, x_max))
                    patch_found = True
                    break 
                    
            if not patch_found: pass 
                
        return patches

class KernelExtractorManager:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def extract_all(self):
        results = {'Bleeding': [], 'Ischemia': [], 'Normal': []}
        print("Iniciando extracción por componentes conectados (Lesion-Centric) con Muestreo Proporcional...")
        
        for cls in ['Bleeding', 'Ischemia']:
            if cls not in self.dataset or self.dataset[cls] is None: 
                continue
                
            print(f"  -> Minando lesiones completas de {cls}...")
            
            # ---------------------------------------------------------
            # DATA AUGMENTATION INTELIGENTE PARA LA MINORÍA
            # Si es Isquemia (la clase que tenías muy baja), le pedimos 
            # al extractor que genere 3 veces más sub-parches solapados.
            # ---------------------------------------------------------
            boost = 3 if cls == 'Ischemia' else 1
            
            for dicom, brain_mask, overlay in zip(self.dataset[cls]['dicom'], 
                                                  self.dataset[cls]['brain'], 
                                                  self.dataset[cls]['overlay']):
                
                extractor = KernelExtractor(dicom, brain_mask, overlay)
                patches = extractor.extract_pathological(pad=PAD, boost_multiplier=boost) 
                results[cls].extend(patches)
                
        count_bleeding = len(results['Bleeding'])
        count_ischemia = len(results['Ischemia'])
        
        min_pathology_count = min(count_bleeding, count_ischemia)
        print(f"\nTotal bruto extraído: {count_bleeding} Bleeding, {count_ischemia} Ischemia.")
        
        if min_pathology_count > 0:
            print(f"  -> Aplicando Undersampling: Recortando a {min_pathology_count} parches por clase...")
            
            random.seed(42)
            random.shuffle(results['Bleeding'])
            results['Bleeding'] = results['Bleeding'][:min_pathology_count]
            
            random.shuffle(results['Ischemia'])
            results['Ischemia'] = results['Ischemia'][:min_pathology_count]
            
            pathological_shapes = [p.shape for p in results['Bleeding']] + [p.shape for p in results['Ischemia']]
        else:
            print("  -> ADVERTENCIA: Una de las patologías no tiene parches.")
            pathological_shapes = []

        if 'Normal' in self.dataset and self.dataset['Normal'] is not None and min_pathology_count > 0:
            normal_images = len(self.dataset['Normal']['dicom'])
            
            random.shuffle(pathological_shapes)
            target_normal_shapes = pathological_shapes[:min_pathology_count]
            
            print(f"  -> Procesando Normal: Buscando exactamente {min_pathology_count} parches con formas específicas...")
            
            if normal_images > 0:
                shapes_per_patient = np.array_split(target_normal_shapes, normal_images)
                
                for i, (dicom, brain_mask) in enumerate(zip(self.dataset['Normal']['dicom'], self.dataset['Normal']['brain'])):
                    
                    target_shapes_for_this_img = shapes_per_patient[i].tolist()
                    if not target_shapes_for_this_img:
                        continue
                        
                    extractor = KernelExtractor(dicom, brain_mask)
                    patches = extractor.extract_normal(target_shapes_for_this_img)
                    results['Normal'].extend(patches)
                    
            random.shuffle(results['Normal'])
            results['Normal'] = results['Normal'][:min_pathology_count]

        final_dict = {}
        print("-" * 50)
        print("RESUMEN DE KERNELS EXTRAÍDOS (BALANCE ESTRICTO 1:1:1):")
        for cls in results:
            clean_list = [p for p in results[cls] if p is not None]
            final_dict[cls] = np.array(clean_list, dtype=object)
            print(f"[{cls}]: {len(final_dict[cls])} parches anatómicos.")
        print("-" * 50)
            
        return final_dict