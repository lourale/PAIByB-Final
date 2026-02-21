import os
import pydicom
import numpy as np
import cv2
from pathlib import Path

class DicomToNumpy:
    def __init__(self, root_dir):
        """
        Inicializa el procesador.
        Args:
            root_dir (str): Nombre de la carpeta raíz (ej: 'Brain_Stroke_CT_Dataset')
        """
        self.root_path = Path(root_dir)
        self.output_path = Path(f"{root_dir}_numpy")
        self.classes = ["Bleeding", "Ischemia", "Normal"]
        self.processing_stats = {}

    def generate_files(self):
        """
        Lee los DICOM y los OVERLAYS, filtra por tamaño 512x512,
        extrae las máscaras por color y guarda archivos .npy organizados.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Iniciando procesamiento desde: {self.root_path}")
        print(f"Destino: {self.output_path}\n")

        for class_name in self.classes:
            dicom_dir = self.root_path / class_name / "DICOM"
            overlay_dir = self.root_path / class_name / "OVERLAY"
            
            if not dicom_dir.exists():
                print(f"Advertencia: No se encontró el directorio {dicom_dir}")
                continue

            # Mapeamos los archivos de overlay existentes por su nombre sin extensión
            overlay_dict = {}
            if overlay_dir.exists():
                for ext in ['.png', '.jpg', '.jpeg']:
                    for f in overlay_dir.glob(f"*{ext}"):
                        overlay_dict[f.stem] = f

            valid_dicoms = []
            valid_overlays = []
            skipped_count = 0
            
            files = list(dicom_dir.glob("*.dcm"))
            
            for file_path in files:
                try:
                    ds = pydicom.dcmread(file_path)
                    
                    if ds.pixel_array.shape == (512, 512):
                        valid_dicoms.append(ds.pixel_array)
                        
                        # --- PROCESAMIENTO DE LA MÁSCARA (OVERLAY) ---
                        # Creamos una máscara de fondo (todo negro/0) por defecto
                        mask = np.zeros((512, 512), dtype=np.uint8)
                        
                        # Si existe un overlay con el mismo nombre que el dicom
                        if file_path.stem in overlay_dict:
                            img_overlay = cv2.imread(str(overlay_dict[file_path.stem]))
                            
                            if img_overlay is not None:
                                img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
                                
                                if img_overlay.shape[:2] == (512, 512):
                                    # Detectar rojo (Alto R, bajo G y B)
                                    red_pixels = (img_overlay[:,:,0] > 150) & (img_overlay[:,:,1] < 100) & (img_overlay[:,:,2] < 100)
                                    # Detectar verde (Alto G, bajo R y B)
                                    green_pixels = (img_overlay[:,:,1] > 150) & (img_overlay[:,:,0] < 100) & (img_overlay[:,:,2] < 100)
                                    
                                    # Asignar valores a la máscara
                                    mask[green_pixels] = 128  # Gris para lo verde
                                    mask[red_pixels] = 255    # Blanco para lo rojo (Cambia 255 por 0 si obligatoriamente lo quieres negro)
                        
                        # Agregamos la máscara (estará vacía para la clase "Normal" o si no hay overlay)
                        valid_overlays.append(mask)
                    else:
                        skipped_count += 1
                        
                except Exception as e:
                    print(f"Error leyendo {file_path.name}: {e}")

            if valid_dicoms:
                dicom_array = np.array(valid_dicoms)
                overlay_array = np.array(valid_overlays)
                
                # Guardamos dos archivos por clase
                dicom_save_file = self.output_path / f"{class_name}_dicom.npy"
                overlay_save_file = self.output_path / f"{class_name}_overlay.npy"
                
                np.save(dicom_save_file, dicom_array)
                np.save(overlay_save_file, overlay_array)
                
                self.processing_stats[class_name] = {
                    "saved": len(valid_dicoms),
                    "skipped": skipped_count,
                    "shape": dicom_array.shape
                }
            else:
                print(f"No se encontraron imágenes válidas para {class_name}")

    def summary(self):
        """Imprime un resumen de cómo se almacenaron los datos."""
        print("-" * 50)
        print("RESUMEN DEL PROCESAMIENTO")
        print("-" * 50)
        
        if not self.processing_stats:
            print("No hay datos procesados. Ejecuta generate_files() primero.")
            return

        total_imgs = 0
        for class_name, stats in self.processing_stats.items():
            print(f"Clase: {class_name}")
            print(f"  - Imágenes/Máscaras (512x512): {stats['saved']}")
            print(f"  - Descartadas (otro tamaño): {stats['skipped']}")
            print(f"  - Shape final de los arrays: {stats['shape']}")
            total_imgs += stats['saved']
            print("." * 30)
            
        print(f"TOTAL PARES (DICOM+MASCARA) PROCESADOS: {total_imgs}")
        print("-" * 50)

    def get_data(self):
        """
        Devuelve un diccionario anidado con los DICOM y sus OVERLAYS.
        Returns:
            dict: { 'Bleeding': {'dicom': array, 'overlay': array}, ... }
        """
        data = {}
        print("Cargando datos en memoria...")
        
        for class_name in self.classes:
            dicom_path = self.output_path / f"{class_name}_dicom.npy"
            overlay_path = self.output_path / f"{class_name}_overlay.npy"
            
            data[class_name] = {}
            
            if dicom_path.exists() and overlay_path.exists():
                data[class_name]['dicom'] = np.load(dicom_path)
                data[class_name]['overlay'] = np.load(overlay_path)
                print(f"  -> Cargado {class_name}: DICOM {data[class_name]['dicom'].shape}, MASK {data[class_name]['overlay'].shape}")
            else:
                print(f"  -> Archivos no encontrados para {class_name}")
                data[class_name] = None
                
        return data
    
    def get_subset(self, limit=1000):
        """
        Igual que get_data, pero limitado a 'limit' cantidad usando memoria eficiente.
        """
        subset_data = {}
        print(f"Cargando subconjunto de datos (Primeras {limit} imágenes)...")
        
        for class_name in self.classes:
            dicom_path = self.output_path / f"{class_name}_dicom.npy"
            overlay_path = self.output_path / f"{class_name}_overlay.npy"
            
            subset_data[class_name] = {}
            
            if dicom_path.exists() and overlay_path.exists():
                dicom_mmap = np.load(dicom_path, mmap_mode='r')
                overlay_mmap = np.load(overlay_path, mmap_mode='r')
                
                subset_data[class_name]['dicom'] = np.array(dicom_mmap[:limit])
                subset_data[class_name]['overlay'] = np.array(overlay_mmap[:limit])
                
                print(f"  -> Cargado subconjunto {class_name}: {subset_data[class_name]['dicom'].shape}")
            else:
                subset_data[class_name] = None
                
        return subset_data

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    processor = DicomToNumpy("Brain_Stroke_CT_Dataset")

    # 1. Generar los archivos (descomenta la siguiente línea la primera vez que lo corras)
    # processor.generate_files()

    # 2. Ver el resumen
    processor.summary()

    # 3. Obtener los datos
    dataset = processor.get_data()
    
    # 4. Verificación
    if dataset['Bleeding'] is not None:
        print(f"\nForma del DICOM: {dataset['Bleeding']['dicom'][0].shape}")
        print(f"Forma del OVERLAY: {dataset['Bleeding']['overlay'][0].shape}")
        print(f"Valores únicos en la máscara: {np.unique(dataset['Bleeding']['overlay'][0])}")