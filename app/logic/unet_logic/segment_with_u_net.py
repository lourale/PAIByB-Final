import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#BLEEDING_MODEL_PATH = '/content/drive/MyDrive/_tp_models/best_unet_bleeding.keras'
#ISCHEMIA_MODEL_PATH = '/content/drive/MyDrive/_tp_models/best_unet_ischemic.keras'


def dice_coef(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    num = 2.0 * tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    den = tf.reduce_sum(y_true + y_pred, axis=[1,2,3]) + eps
    return tf.reduce_mean(num / den)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

bce = tf.keras.losses.BinaryCrossentropy()

def bce_dice_weighted_loss(y_true, y_pred):
    return W_BCE * bce(y_true, y_pred) + W_DICE * dice_loss(y_true, y_pred)

# (Opcional) IoU simple para monitorear
def iou_coef(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3]) - inter
    return tf.reduce_mean((inter + eps) / (union + eps))

def segment_bleeding_with_unet(image_path, model_path, IMG_SIZE = 384, W_BCE  = 0.55, W_DICE = 0.45, PRED_THR = 0.65):
    """
    Segmenta una imagen PNG usando un modelo U-Net pre-entrenado.

    Args:
        image_path (str): Ruta al archivo PNG de la imagen de entrada.
        model_path (str): Ruta al archivo .keras del modelo U-Net.

    Returns:
        np.ndarray: Máscara binaria segmentada (IMG_SIZE, IMG_SIZE), tipo float32.
    """
    # Cargar el modelo con los objetos personalizados
    # Asegúrate de que las funciones bce_dice_weighted_loss, dice_coef, iou_coef y dice_loss
    # estén definidas en el ámbito global del notebook o pasarlas explícitamente.
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'bce_dice_weighted_loss': bce_dice_weighted_loss,
                        'dice_coef': dice_coef,
                        'iou_coef': iou_coef,
                        'dice_loss': dice_loss}
    )

    # Preprocesar la imagen de entrada
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1) # Decodificar a escala de grises (1 canal)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method='bilinear')
    img = tf.expand_dims(img, axis=0) # Añadir dimensión de batch (1, IMG_SIZE, IMG_SIZE, 1)

    # Realizar la predicción
    prediction = model.predict(img, verbose=0)

    # Binarizar la predicción usando el umbral global PRED_THR
    binary_mask = (prediction > PRED_THR).astype(np.float32)

    # Eliminar la dimensión de batch para la salida
    return np.squeeze(binary_mask, axis=0)

def segment_ischemic_with_unet(image_path, model_path, IMG_SIZE = 384, W_BCE  = 0.6, W_DICE = 0.4, PRED_THR = 0.75):
    """
    Segmenta una imagen PNG usando un modelo U-Net pre-entrenado.

    Args:
        image_path (str): Ruta al archivo PNG de la imagen de entrada.
        model_path (str): Ruta al archivo .keras del modelo U-Net.

    Returns:
        np.ndarray: Máscara binaria segmentada (IMG_SIZE, IMG_SIZE), tipo float32.
    """
    # Cargar el modelo con los objetos personalizados
    # Asegúrate de que las funciones bce_dice_weighted_loss, dice_coef, iou_coef y dice_loss
    # estén definidas en el ámbito global del notebook o pasarlas explícitamente.
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'bce_dice_weighted_loss': bce_dice_weighted_loss,
                        'dice_coef': dice_coef,
                        'iou_coef': iou_coef,
                        'dice_loss': dice_loss}
    )

    # Preprocesar la imagen de entrada
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1) # Decodificar a escala de grises (1 canal)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method='bilinear')
    img = tf.expand_dims(img, axis=0) # Añadir dimensión de batch (1, IMG_SIZE, IMG_SIZE, 1)

    # Realizar la predicción
    prediction = model.predict(img, verbose=0)

    # Binarizar la predicción usando el umbral global PRED_THR
    binary_mask = (prediction > PRED_THR).astype(np.float32)

    # Eliminar la dimensión de batch para la salida
    return np.squeeze(binary_mask, axis=0)

'''
# --- Ejemplo de uso ---
print(f"Cargando modelo desde: {BLEEDING_MODEL_PATH}")

# Obtener una ruta de imagen de ejemplo del conjunto de validación
sample_image_path = # Tomar imagen de ejemplo
print(f"Segmentando imagen de ejemplo: {sample_image_path}")

# Llamar a la función de segmentación
segmented_mask = segment_bleeding_with_unet(sample_image_path, BLEEDING_MODEL_PATH)

# Cargar y preprocesar la imagen original para visualización
original_img_display = tf.io.read_file(sample_image_path)
original_img_display = tf.image.decode_png(original_img_display, channels=1)
original_img_display = tf.image.convert_image_dtype(original_img_display, tf.float32)
original_img_display = tf.image.resize(original_img_display, (384, 384), method='bilinear')

# Visualizar la imagen original y la máscara segmentada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(original_img_display), cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_mask, cmap='gray')
plt.title(f'Máscara Segmentada (Umbral: {PRED_THR})')
plt.axis('off')
plt.show()'''