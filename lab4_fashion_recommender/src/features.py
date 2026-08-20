"""Извлечение признаков изображений тремя независимыми подходами."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage.transform import resize

from .utils import l2_normalize


class FeatureExtractor(ABC):
    """Интерфейс, позволяющий честно сравнивать разные векторные представления."""

    name: str

    @abstractmethod
    def transform(self, image_paths: list[str | Path]) -> np.ndarray:
        """Возвращает L2-нормированные эмбеддинги размера (n_images, n_features)."""


class HogPcaExtractor(FeatureExtractor):
    """Классический computer vision baseline: HOG → PCA → cosine KNN."""

    name = "hog_pca"

    def __init__(self, image_size: int = 128, n_components: int = 128):
        self.image_size = image_size
        self.n_components = n_components
        self.pca: PCA | None = None

    def _describe(self, image_path: str | Path) -> np.ndarray:
        with Image.open(image_path) as image:
            grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        scaled = resize(grayscale, (self.image_size, self.image_size), anti_aliasing=True)
        return hog(scaled, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    def transform(self, image_paths: list[str | Path]) -> np.ndarray:
        descriptors = np.vstack([self._describe(path) for path in image_paths]).astype("float32")
        n_components = min(self.n_components, descriptors.shape[0] - 1, descriptors.shape[1])
        if n_components < 2:
            return l2_normalize(descriptors)
        self.pca = PCA(n_components=n_components, random_state=42, whiten=True)
        return l2_normalize(self.pca.fit_transform(descriptors).astype("float32"))


class TorchvisionExtractor(FeatureExtractor):
    """Transfer learning: эмбеддинги предобученной CNN без дообучения на ярлыках."""

    def __init__(self, architecture: str, batch_size: int = 32, device: str = "auto"):
        if architecture not in {"resnet18", "mobilenet_v3_small"}:
            raise ValueError(f"Неизвестная архитектура: {architecture}")
        self.name = architecture
        self.architecture = architecture
        self.batch_size = batch_size
        self.device_name = device

    def _build_model(self):
        # Импорты внутри метода: EDA и тесты не требуют установленного PyTorch.
        import torch
        from torchvision import models

        if self.architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            model.fc = torch.nn.Identity()
        else:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
            model.classifier[-1] = torch.nn.Identity()
        model.eval()
        device = "cuda" if self.device_name == "auto" and torch.cuda.is_available() else self.device_name
        if device == "auto":
            device = "cpu"
        return model.to(device), weights.transforms(), device

    def transform(self, image_paths: list[str | Path]) -> np.ndarray:
        import torch

        model, preprocess, device = self._build_model()
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(image_paths), self.batch_size):
                paths = image_paths[start : start + self.batch_size]
                images = []
                for path in paths:
                    with Image.open(path) as image:
                        images.append(preprocess(image.convert("RGB")))
                tensor = torch.stack(images).to(device)
                batches.append(model(tensor).cpu().numpy().astype("float32"))
        return l2_normalize(np.vstack(batches))


def make_extractor(name: str, batch_size: int = 32, device: str = "auto") -> FeatureExtractor:
    if name == HogPcaExtractor.name:
        return HogPcaExtractor()
    if name in {"resnet18", "mobilenet_v3_small"}:
        return TorchvisionExtractor(name, batch_size=batch_size, device=device)
    raise ValueError(f"Поддерживаемые методы: hog_pca, resnet18, mobilenet_v3_small; получено {name}")
