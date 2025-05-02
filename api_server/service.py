import torch
import logging
import os
import configparser
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Union, List, Dict, Optional
from PIL import Image
from preprocessing import ImagePreprocessor
from nets.utils import get_model  # Import your model utilities

class ModelWrapper:
    def __init__(self, model_path: str, model_arch: str, weight: float,
                 num_classes: int, input_size: int, 
                 multi_channels: bool = False, multiple_scale: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = weight
        self.multi_channels = multi_channels
        self.multiple_scale = multiple_scale
        self.input_size = input_size
        
        self.model = get_model(model_arch, num_classes, multi_channels=multi_channels).to(self.device)
        self.load_checkpoint(model_path)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=False)

class FASInferenceService:
    def __init__(self, config_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: List[ModelWrapper] = []
        self.preprocessors: List[ImagePreprocessor] = []
        self.executor: Optional[ThreadPoolExecutor] = None
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        for section in config.sections():
            model = ModelWrapper(
                model_path=config[section]['model_path'],
                model_arch=config[section]['model_arch'],
                weight=config[section].getfloat('weight'),
                num_classes=config[section].getint('num_classes', 2),
                input_size=config[section].getint('input_size', 256),
                multi_channels=config[section].getboolean('multi_channels', False),
                multiple_scale=config[section].getboolean('multiple_scale', False)
            )
            
            preprocessor = ImagePreprocessor(
                input_size=config[section].getint('input_size', 256),
                multi_channels=config[section].getboolean('multi_channels', False)
            )
            
            self.models.append(model)
            self.preprocessors.append(preprocessor)
        
        self.total_weight = sum(model.weight for model in self.models)
        self.threshold = 0.1
        
        if not self.models:
            raise ValueError("No models found in configuration file")
        
        self.executor = ThreadPoolExecutor(max_workers=len(self.models))
        
        logging.info(f"Initialized FAS Service with {len(self.models)} models on {self.device}")

    def process_image(self, image_data: Union[bytes, str, Image.Image, np.ndarray],
                      bbox: Optional[List[int]] = None, 
                      scales: Optional[List[float]] = None) -> Dict[str, float]:
        # try:
        preprocessor = self.preprocessors[0]
        image = preprocessor.load_image(image_data)
        processed_tensor = preprocessor.preprocess_image(image, bbox)
        
        tensor = processed_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, outputs = self.models[0].model(tensor)
            probs = torch.softmax(outputs, dim=1)
            spoof_prob = probs[:, 1].cpu().numpy()

        return {'spoof_prob': float(spoof_prob[0])}
        # except Exception as e:
        #     logging.error(f"Error processing image for spoof classification: {str(e)}")
        #     raise
