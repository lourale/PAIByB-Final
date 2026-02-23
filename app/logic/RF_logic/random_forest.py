import numpy as np
import scipy.ndimage as ndimage
import pandas as pd
from logic.feature_extract_logic.feature_extract import FeatureExtract
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BEST_FEATURES = ['variance', 'glcm_correlation', 'energy_1st', 'glcm_entropy', 'skewness', 'kurtosis', 'mean_hu']
PATCH_SIZE = 20
STRIDE = 20

def predict_image_rf_sliding_window(dicom, brain_mask, model, best_features, patch_size, stride):
    h, w = dicom.shape
    half = patch_size // 2
    
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    
    features_batch = []
    coords = []
    
    # ---------------------------------------------------------
    # TRUCO MAESTRO 1: LA ZONA SEGURA (Erosión)
    # Encogemos la máscara la mitad del tamaño del parche. 
    # Así, el centro de la ventana jamás se acercará al borde del cráneo.
    # ---------------------------------------------------------
    safe_mask = ndimage.binary_erosion(brain_mask, iterations=half)
    
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            
            # 1. Solo evaluamos si el centro está dentro de la ZONA SEGURA
            if safe_mask[y, x]:
                patch = dicom[y-half : y+half, x-half : x+half]
                
                # ---------------------------------------------------------
                # NUEVO FILTRO FÍSICO ESTRICTO
                # Descartamos líquido cefalorraquídeo (ventrículos) y calcificaciones.
                # ---------------------------------------------------------
                if np.min(patch) < 15 or np.max(patch) > 85:
                    continue 
                
                extractor = FeatureExtract(patch)
                all_features = extractor.analyze()
                
                patch_vector = [all_features[f] for f in best_features]
                features_batch.append(patch_vector)
                coords.append((y, x))
                
    if not features_batch:
        return pred_mask
        
    X_batch = pd.DataFrame(features_batch, columns=best_features)
    
    # ---------------------------------------------------------
    # NUEVA PREDICCIÓN: UMBRAL DE CONFIANZA
    # Usamos predict_proba para saber qué tan seguro está el modelo
    # ---------------------------------------------------------
    probabilities = model.predict_proba(X_batch)
    classes = model.classes_
    
    s_half = stride // 2
    for i, (y, x) in enumerate(coords):
        probs = probabilities[i]
        max_prob = np.max(probs)
        pred_label = classes[np.argmax(probs)]
        
        # Solo pinta la predicción si está más de un 75% seguro
        if max_prob > 0.75:
            if pred_label == 'Bleeding':
                pred_mask[y-s_half : y+s_half, x-s_half : x+s_half] = 1
            elif pred_label == 'Ischemia':
                pred_mask[y-s_half : y+s_half, x-s_half : x+s_half] = 2

    # Suavizado morfológico
    mask_bleeding = pred_mask == 1
    mask_ischemia = pred_mask == 2
    
    mask_bleeding = ndimage.binary_opening(mask_bleeding, structure=np.ones((3,3)))
    mask_bleeding = ndimage.binary_closing(mask_bleeding, structure=np.ones((5,5)))
    
    mask_ischemia = ndimage.binary_opening(mask_ischemia, structure=np.ones((3,3)))
    mask_ischemia = ndimage.binary_closing(mask_ischemia, structure=np.ones((5,5)))
    
    final_mask = np.zeros_like(pred_mask)
    final_mask[mask_bleeding] = 1
    final_mask[mask_ischemia] = 2
            
    return final_mask

def plot_rf_test_results(test_dataset, cls_target, model, num_samples=2, best_features=None, patch_size=None, stride=None):
    """
    Evalúa pacientes reales del test set y plotea el resultado comparativo.
    """
    print(f"\n--- FASE 3: EVALUACIÓN VISUAL ({cls_target}) ---")
    print(f"Deslizando ventana (stride={stride}) sobre {num_samples} pacientes...")
    
    total_imgs = len(test_dataset[cls_target]['dicom'])
    indices = np.random.choice(total_imgs, num_samples, replace=False)
    
    cmap_custom = ListedColormap(['none', 'red', 'blue'])
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6 * num_samples))
    if num_samples == 1: axes = [axes]
        
    for i, idx in enumerate(indices):
        dicom = test_dataset[cls_target]['dicom'][idx]
        brain = test_dataset[cls_target]['brain'][idx]
        overlay_true = test_dataset[cls_target]['overlay'][idx] if test_dataset[cls_target]['overlay'] is not None else np.zeros_like(dicom)
        
        # 1. Inferencia
        pred_mask = predict_image_rf_sliding_window(dicom, brain, model, best_features, patch_size, stride)
        
        # 2. Plot: Realidad
        axes[i][0].imshow(dicom, cmap='gray', vmin=0, vmax=80)
        mask_to_plot = np.zeros_like(overlay_true)
        if cls_target == 'Bleeding': mask_to_plot[overlay_true > 0] = 1
        elif cls_target == 'Ischemia': mask_to_plot[overlay_true > 0] = 2
        
        axes[i][0].imshow(mask_to_plot, cmap=cmap_custom, alpha=0.5, interpolation='none')
        axes[i][0].set_title(f"Ground Truth - {cls_target}", fontsize=12)
        axes[i][0].axis('off')
        
        # 3. Plot: Predicción IA
        axes[i][1].imshow(dicom, cmap='gray', vmin=0, vmax=80)
        axes[i][1].imshow(pred_mask, cmap=cmap_custom, alpha=0.5, interpolation='none')
        axes[i][1].set_title(f"Predicción RF (Sliding Window)", fontsize=12)
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.show()