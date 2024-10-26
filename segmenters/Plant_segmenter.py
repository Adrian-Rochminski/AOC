from Segmenter import Segmenter

import numpy as np

class PlantSegmenter(Segmenter):
    def __init__(self, class_info):
        super().__init__(class_info)

    def segment(self, image, mask):
        print(f"Segmenting Plant...")
        # Placeholder segmentation logic for Plant
        segmentation_mask = (mask == self.class_index).astype(np.uint8)
        return self.apply_mask(image, segmentation_mask)

    def apply_mask(self, image, mask):
        pass