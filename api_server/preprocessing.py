import cv2
import numpy as np
import torch
from typing import Optional, List, Union, Tuple
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

class SquarePad(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        
    def apply(self, image, **params):
        h, w = image.shape[:2]
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = ((vp, max_wh - h - vp), (hp, max_wh - w - hp), (0, 0))
        image = np.pad(image, padding, mode='constant', constant_values=0)
        return image

class FiveChannelTransform(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        
    def fft(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray_image)
        fshift = np.fft.fftshift(f)
        fimg = np.log(np.abs(fshift) + 1)
        fimg = np.expand_dims(fimg, -1)
        return fimg

    def normalize_matrix(self, matrix):
        minn = np.amin(matrix)
        maxx = np.amax(matrix)
        normalized_matrix = (matrix - minn + 1) / (maxx - minn + 1)
        return normalized_matrix

    def apply(self, image, **params):
        # fft_image = self.fft(image)
        # fft_image = self.normalize_matrix(fft_image)
        image = np.array(image, dtype=np.float32) / 255.0
        # combined_image = np.concatenate((image, fft_image), axis=-1)
        return image.astype(np.float32)

class ImagePreprocessor:
    def __init__(self, input_size: int = 256, multi_channels: bool = False):
        self.input_size = input_size
        self.multi_channels = multi_channels
        self.transforms = self._build_transforms()

    def _build_transforms(self) -> A.Compose:
        transform_list = [
            SquarePad(p=1.0),
            A.Resize(height=self.input_size, width=self.input_size, p=1.0),
        ]
        
        # if self.multi_channels:
        transform_list.append(FiveChannelTransform(p=1.0))
        # else:
        #     transform_list.append(A.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225],
        #         p=1.0
        #     ))
            
        transform_list.append(ToTensorV2(p=1.0))
        return A.Compose(transform_list)

    def load_image(self, image_data: Union[bytes, str, Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_data, str):
            image = cv2.imread(image_data)
        elif isinstance(image_data, Image.Image):
            image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
        elif isinstance(image_data, np.ndarray):
            if len(image_data.shape) == 2:
                image = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
            elif len(image_data.shape) == 3 and image_data.shape[2] == 3:
                image = image_data.copy()
            else:
                raise ValueError("Unsupported numpy array format")
        else:
            raise ValueError("Unsupported image data format")
            
        if image is None:
            raise ValueError("Failed to load image")
            
        return image

    def scale_bbox(self, bbox: List[int], scale: float, image_shape: Tuple[int, ...]) -> List[int]:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        center_x, center_y = x_min + width / 2, y_min + height / 2

        new_width = width * scale
        new_height = height * scale

        x_min = max(0, int(center_x - new_width / 2))
        y_min = max(0, int(center_y - new_height / 2))
        x_max = min(image_shape[1], int(center_x + new_width / 2))
        y_max = min(image_shape[0], int(center_y + new_height / 2))

        return [x_min, y_min, x_max, y_max]

    def preprocess_image(self, 
                        image: np.ndarray, 
                        bbox = None,
                        scale: float = 1.3) -> torch.Tensor:
        if bbox is not None:
            scaled_bbox = self.scale_bbox(bbox, scale, image.shape)
            x_min, y_min, x_max, y_max = scaled_bbox
            image = image[y_min:y_max, x_min:x_max]
        transformed = self.transforms(image=image)
        return transformed['image']
