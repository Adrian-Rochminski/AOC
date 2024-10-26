from abc import ABC, abstractmethod
import numpy as np
from PIL.Image import Image

class Segmenter(ABC):
    def __init__(self, class_info):
        self.class_name = class_info['name']
        self.class_index = class_info['classIndex']

    @abstractmethod
    def segment(self, image, mask):
        pass

    @abstractmethod
    def apply_mask(self, image, mask):
        pass