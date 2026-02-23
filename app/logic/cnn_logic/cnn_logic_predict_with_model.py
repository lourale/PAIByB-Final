#!pip install pydicom

import os
import numpy as np
import pydicom
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def dicom_to_array(ds: pydicom.Dataset) -> np.ndarray:
    """Devuelve pixel array como float32, aplicando slope/intercept si existen."""
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    return arr

def apply_window(img_hu: np.ndarray, wc: float, ww: float) -> np.ndarray:
    """Aplica windowing y devuelve float32 en [0,1]."""
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    x = np.clip(img_hu, lo, hi)
    x = (x - lo) / (hi - lo + 1e-9)
    return x.astype(np.float32)

def ensure_png(path: str, wc=40, ww=120):
  ext = os.path.splitext(path)[1].lower()
  if ext == ".dcm":
    ds = pydicom.dcmread(path)
    img_hu = dicom_to_array(ds)

    img01 = apply_window(img_hu, wc, ww)
    img_u8 = (img01 * 255.0).round().astype(np.uint8)

    if img_u8.ndim == 3:
        img_u8 = img_u8[..., 0]
    return (img_u8.astype(np.float32) / 255.0)

  elif ext == ".png":
    im = imageio.imread(path)
    if im.ndim == 3:
      im = (0.2126*im[...,0] + 0.7152*im[...,1] + 0.0722*im[...,2]).astype(np.uint8)
    elif im.ndim == 2:
      im = im.astype(np.uint8)
    else:
      raise ValueError("PNG image has unsupported number of dimensions.")

    return (im.astype(np.float32) / 255.0)

  else:
    raise ValueError("Formato no soportado. Solo .dcm o .png")


class ImageClassificationBase(nn.Module):
    def __init__(self):
        super().__init__()

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res2 = nn.Sequential(ConvBlock(256, 256), ConvBlock(256, 256))

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res3 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, num_diseases)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.res2(out) + out
        out = self.conv4(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out

def predict_with_model(image_path: str, model_path: str, class_names: list):
    """
    Args:
        image_path (str): Path to the input DICOM or PNG image.
        model_path (str): Path to the pre-trained PyTorch model (.pth file).
        class_names (list): A list of class labels corresponding to the model's output.

    Returns:
        tuple: A tuple containing the predicted class name (str)
               and the probabilities tensor (torch.Tensor).
    """
    img_array = ensure_png(image_path)
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)

    num_diseases = len(class_names)
    actual_model = CNN_NeuralNet(in_channels=3, num_diseases=num_diseases)

    state_dict = torch.load(model_path)
    actual_model.load_state_dict(state_dict)

    actual_model.eval()

    with torch.no_grad():
        raw_prediction = actual_model(img_tensor)

    probabilities = F.softmax(raw_prediction, dim=1)
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    predicted_class_name = class_names[predicted_class_idx]

    return predicted_class_name, probabilities