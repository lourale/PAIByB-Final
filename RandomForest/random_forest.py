import numpy as np
import scipy.ndimage as ndimage
import pandas as pd
from feature_analysis.feature_extract import FeatureExtract
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def predict_image_rf_sliding_window_multiscale(dicom, brain_mask, model, best_features, patch_sizes):
    """
    Desliza ventanas de múltiples tamaños. El salto (stride) se calcula 
    automáticamente como la mitad del tamaño de la ventana actual.
    """
    h, w = dicom.shape
    num_classes = len(model.classes_)
    
    # ---------------------------------------------------------
    # MAPAS DE VOTACIÓN PARA SOFT VOTING MULTIESCALA
    # ---------------------------------------------------------
    prob_map = np.zeros((h, w, num_classes), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    features_batch = []
    coords = []
    
    # ---------------------------------------------------------
    # BUCLE MULTIESCALA: Iteramos sobre la lista de tamaños
    # ---------------------------------------------------------
    for p_size in patch_sizes:
        half = p_size // 2
        
        # Movimiento dinámico: salto igual a la mitad de la ventana
        stride = max(1, half) 
        
        # La zona segura se recalcula porque depende del tamaño del parche actual
        safe_mask = ndimage.binary_erosion(brain_mask, iterations=half)
        
        for y in range(half, h - half, stride):
            for x in range(half, w - half, stride):
                
                if safe_mask[y, x]:
                    patch = dicom[y-half : y+half, x-half : x+half]
                    
                    if np.min(patch) < 15 or np.max(patch) > 85:
                        continue 
                    
                    extractor = FeatureExtract(patch)
                    all_features = extractor.analyze()
                    
                    patch_vector = [all_features[f] for f in best_features]
                    features_batch.append(patch_vector)
                    
                    # Guardamos la coordenada Y, X y también 'half' de esta escala
                    coords.append((y, x, half))
                    
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    if not features_batch:
        return pred_mask
        
    X_batch = pd.DataFrame(features_batch, columns=best_features)
    
    probabilities = model.predict_proba(X_batch)
    classes = list(model.classes_)
    
    # ---------------------------------------------------------
    # MAGIA DEL SOLAPAMIENTO: ACUMULACIÓN MASIVA DE VOTOS
    # ---------------------------------------------------------
    for i, (y, x, hf) in enumerate(coords):
        probs = probabilities[i]
        
        # El voto se derrama según el tamaño 'hf' que tenía el parche evaluado
        prob_map[y-hf : y+hf, x-hf : x+hf] += probs
        count_map[y-hf : y+hf, x-hf : x+hf] += 1

    # ---------------------------------------------------------
    # CÁLCULO DEL PROMEDIO GRANULAR
    # ---------------------------------------------------------
    valid_pixels = count_map > 0
    prob_map[valid_pixels] /= count_map[valid_pixels][:, None]
    
    if valid_pixels.any():
        winner_indices = np.argmax(prob_map, axis=2)
        max_probs = np.max(prob_map, axis=2)
        
        # Umbral de confianza estricto (> 75% seguro)
        mask_confident = (max_probs > 0.75) & valid_pixels
        
        idx_bleeding = classes.index('Bleeding') if 'Bleeding' in classes else -1
        idx_ischemia = classes.index('Ischemia') if 'Ischemia' in classes else -1
        
        if idx_bleeding != -1:
            pred_mask[mask_confident & (winner_indices == idx_bleeding)] = 1
        if idx_ischemia != -1:
            pred_mask[mask_confident & (winner_indices == idx_ischemia)] = 2

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

def plot_rf_test_results(test_dataset, cls_target, model, num_samples=2, best_features=None, patch_sizes=None):
    """
    Evalúa pacientes reales deslizando múltiples ventanas (Multiescala).
    No requiere el parámetro 'stride' ya que se calcula internamente.
    """
    if patch_sizes is None:
        patch_sizes = [10, 15, 20]
        
    print(f"\n--- FASE 3: EVALUACIÓN VISUAL ({cls_target}) ---")
    print(f"Evaluando escalas: {patch_sizes} (Movimiento al 50% de cada parche)")
    print(f"Evaluando {num_samples} pacientes...")
    
    total_imgs = len(test_dataset[cls_target]['dicom'])
    indices = np.random.choice(total_imgs, num_samples, replace=False)
    
    cmap_custom = ListedColormap(['none', 'red', 'blue'])
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6 * num_samples))
    if num_samples == 1: axes = [axes]
        
    for i, idx in enumerate(indices):
        dicom = test_dataset[cls_target]['dicom'][idx]
        brain = test_dataset[cls_target]['brain'][idx]
        overlay_true = test_dataset[cls_target]['overlay'][idx] if test_dataset[cls_target]['overlay'] is not None else np.zeros_like(dicom)
        
        # 1. Inferencia Multiescala
        pred_mask = predict_image_rf_sliding_window_multiscale(dicom, brain, model, best_features, patch_sizes)
        
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
        axes[i][1].set_title(f"Predicción RF Multiescala (Soft Voting)", fontsize=12)
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.show()