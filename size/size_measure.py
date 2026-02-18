import os
import pydicom
import cv2
from collections import Counter
from pathlib import Path

def measure_size(folder_path):
    # Contador para agrupar tamaños: {(alto, ancho): cantidad}
    conteo_tamanos = Counter()
    archivos_corruptos = []
    total_leidos = 0
    
    print(f"---  {folder_path} ---")
    
    # Listamos archivos
    archivos = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    for nombre in archivos:
        ruta_completa = os.path.join(folder_path, nombre)
        extension = nombre.split('.')[-1].lower()
        
        dims = None # (alto, ancho)
        
        try:
            if extension == 'dcm':
                # TRUCO DE VELOCIDAD: stop_before_pixels=True 
                # Lee solo los metadatos sin cargar la imagen pesada a la RAM.
                ds = pydicom.dcmread(ruta_completa, stop_before_pixels=True, force=True)
                
                if 'Rows' in ds and 'Columns' in ds:
                    dims = (int(ds.Rows), int(ds.Columns))
                else:
                    archivos_corruptos.append(f"{nombre} (Sin etiquetas Rows/Columns)")
                    
            elif extension in ['png', 'jpg', 'jpeg', 'bmp', 'tif']:
                # Para imágenes normales usamos OpenCV rápido
                # Leemos solo el header si es posible, o cargamos rápido
                img = cv2.imread(ruta_completa, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    dims = img.shape[:2] # (Alto, Ancho)
                else:
                    archivos_corruptos.append(f"{nombre} (cv2 no pudo leerlo)")
            
            # Si obtuvimos dimensiones, las contamos
            if dims:
                conteo_tamanos[dims] += 1
                total_leidos += 1
                
        except Exception as e:
            archivos_corruptos.append(f"{nombre} (Error: {str(e)})")

    # --- REPORTE FINAL ---
    print(f"Total archivos analizados con éxito: {total_leidos}")
    
    if archivos_corruptos:
        print(f"Archivos fallidos/ignorados: {len(archivos_corruptos)}")
        # Descomenta la siguiente línea si quieres ver cuáles fallaron
        # print(archivos_corruptos)

    # Ordenamos por cantidad (los más comunes primero)
    for tamano, cantidad in conteo_tamanos.most_common():
        estado = "✅ Dominante" if cantidad == max(conteo_tamanos.values()) else "⚠️ Diferente"
        es_cuadrada = "Cuadrada" if tamano[0] == tamano[1] else "Rectangular"
        
        print(f"  • {tamano[0]} x {tamano[1]} píxeles ({es_cuadrada}): {cantidad} archivos  {estado}")

