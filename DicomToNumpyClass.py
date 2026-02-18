import os
import pydicom
import numpy as np
from pathlib import Path

class DicomToNumpy:
    def __init__(self, root_dir):
        """
        Inicializa el procesador.
        Args:
            root_dir (str): Nombre de la carpeta raíz (ej: 'Brain_Stroke_CT_Dataset')
        """
        self.root_path = Path(root_dir)
        # Define el nombre de la carpeta de salida automáticamente
        self.output_path = Path(f"{root_dir}_numpy")
        
        # Las clases basadas en tu estructura de carpetas
        self.classes = ["Bleeding", "Ischemia", "Normal"]
        
        # Diccionario para guardar estadísticas internas
        self.processing_stats = {}

    def generate_files(self):
        """
        Lee los DICOM, filtra por tamaño 512x512 y guarda archivos .npy
        organizados por clase.
        """
        # Crear directorio de salida si no existe
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Iniciando procesamiento desde: {self.root_path}")
        print(f"Destino: {self.output_path}\n")

        for class_name in self.classes:
            # Construir ruta: Raiz/Clase/DICOM
            dicom_dir = self.root_path / class_name / "DICOM"
            
            if not dicom_dir.exists():
                print(f"Advertencia: No se encontró el directorio {dicom_dir}")
                continue

            valid_images = []
            skipped_count = 0
            
            # Iterar sobre archivos .dcm
            files = list(dicom_dir.glob("*.dcm"))
            
            for file_path in files:
                try:
                    ds = pydicom.dcmread(file_path)
                    
                    # Verificar dimensión 512x512
                    if ds.pixel_array.shape == (512, 512):
                        # Agregamos la imagen a la lista (normalizada o raw según prefieras)
                        # Aquí guardamos los datos crudos (raw pixel data)
                        valid_images.append(ds.pixel_array)
                    else:
                        skipped_count += 1
                        
                except Exception as e:
                    print(f"Error leyendo {file_path.name}: {e}")

            # Convertir a array de numpy y guardar
            if valid_images:
                data_array = np.array(valid_images)
                save_file = self.output_path / f"{class_name}.npy"
                np.save(save_file, data_array)
                
                # Guardar estadísticas
                self.processing_stats[class_name] = {
                    "saved": len(valid_images),
                    "skipped": skipped_count,
                    "shape": data_array.shape,
                    "file_path": str(save_file)
                }
            else:
                print(f"No se encontraron imágenes válidas para {class_name}")

    def summary(self):
        """
        Imprime un resumen de cómo se almacenaron los datos.
        """
        print("-" * 50)
        print("RESUMEN DEL PROCESAMIENTO")
        print("-" * 50)
        
        if not self.processing_stats:
            print("No hay datos procesados. Ejecuta generate_files() primero.")
            return

        total_imgs = 0
        for class_name, stats in self.processing_stats.items():
            print(f"Clase: {class_name}")
            print(f"  - Imágenes guardadas (512x512): {stats['saved']}")
            print(f"  - Imágenes descartadas (otro tamaño): {stats['skipped']}")
            print(f"  - Shape final del array: {stats['shape']}")
            print(f"  - Guardado en: {stats['file_path']}")
            total_imgs += stats['saved']
            print("." * 30)
            
        print(f"TOTAL IMÁGENES PROCESADAS: {total_imgs}")
        print("-" * 50)

    def get_data(self):
        """
        Carga y devuelve toda la información de los archivos ya generados.
        Returns:
            dict: Un diccionario donde las claves son las clases ('Bleeding', etc.)
                  y los valores son los arrays numpy con las imágenes.
        """
        data = {}
        print("Cargando datos en memoria...")
        
        for class_name in self.classes:
            file_path = self.output_path / f"{class_name}.npy"
            
            if file_path.exists():
                # Cargamos el archivo .npy
                images = np.load(file_path)
                data[class_name] = images
                print(f"  -> Cargado {class_name}: {images.shape}")
            else:
                print(f"  -> Archivo no encontrado para {class_name}")
                data[class_name] = None
                
        return data

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Instanciamos la clase apuntando a tu carpeta raiz
    processor = DicomToNumpy("Brain_Stroke_CT_Dataset")

    # 2. Ver el resumen
    processor.summary()

    # 3. Obtener los datos para trabajar (ej. para meter en un modelo)
    dataset = processor.get_data()
    
    # Ejemplo de verificación
    if dataset['Bleeding'] is not None:
        print(f"\nVerificación: La primera imagen de Bleeding tiene forma {dataset['Bleeding'][0].shape}")