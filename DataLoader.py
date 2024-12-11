import os
from PIL import Image
import numpy as np
import random

from matplotlib import pyplot as plt


class DataLoader:
    def __init__(self, class_merge_map=None, images_file='images/images.npy', masks_file='images/masks.npy'):
        self.images_file = images_file
        self.masks_file = masks_file
        self.images = None
        self.masks = None
        self.selected_indices = []
        self.class_merge_map = class_merge_map

    def load_data(self, mode='all'):
        self.images = np.load(self.images_file, mmap_mode='r')
        self.masks = np.load(self.masks_file, mmap_mode='r')

        total_images = self.images.shape[0]
        indices = list(range(total_images))

        if mode == 'all':
            self.selected_indices = indices
        elif mode == 'half':
            self.selected_indices = indices[:total_images // 2]
        elif mode == 'quarter':
            self.selected_indices = indices[:total_images // 4]
        elif isinstance(mode, int):
            self.selected_indices = indices[:mode]
        else:
            raise ValueError("Invalid mode. Use 'all', 'half', 'quarter', or an integer number of images.")

    def display_random_images_with_masks(self, num_examples=5):
        num_examples = min(num_examples, len(self.images))
        random_indices = random.sample(range(len(self.images)), num_examples)
        for idx in random_indices:
            image, mask = self.images[idx] , self.masks[idx]
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
            axes[0].set_title('Image')
            axes[0].axis('off')
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Mask')
            axes[1].axis('off')
            plt.show()

    def get_image(self, idx):
        # if idx < 0 or idx >= len(self.selected_indices):
        #     raise IndexError("Index out of bounds")
        actual_idx = self.selected_indices[idx]
        image = self.images[actual_idx]
        mask = self.masks[actual_idx].copy()

        if self.class_merge_map:
            for src_class, target_class in self.class_merge_map.items():
                mask[mask == src_class] = target_class

        return image, mask

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        return self.get_image(idx)

    def __iter__(self):
        for idx in range(len(self)):
            yield self.get_image(idx)
