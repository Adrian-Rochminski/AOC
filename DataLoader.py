import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

CLASS_GROUPS = {
    "water": [8, 9, 13, 16],
    "rocky": [6, 11, 14],
    "land": [5, 12, 15, 10],
    "vegetation": [2, 4, 7],
    "sky": [1],
    "structures": [0, 3]
}

class DataLoader:
    def __init__(self, images_file='images/images.npy', masks_file='images/masks.npy'):
        self.images_file = images_file
        self.masks_file = masks_file
        self.images = None
        self.masks = None
        self.selected_indices = []
        self.class_map = {}
        for group_id, (group_name, class_ids) in enumerate(CLASS_GROUPS.items()):
            for value in class_ids:
                self.class_map[value] = group_id

    def train_test_split(self, test_size=0.2, random_state=None, stratify_by=None):
        if not self.selected_indices:
            raise ValueError("Load data first using load_data()")

        stratify = None
        if stratify_by == 'class':
            stratify = [np.any(mask != 0) for mask in self.masks[self.selected_indices]]

        train_indices, test_indices = train_test_split(
            self.selected_indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        def create_subset_loader(indices):
            new_loader = DataLoader(self.images_file, self.masks_file)
            new_loader.images = self.images
            new_loader.masks = self.masks
            new_loader.selected_indices = indices
            new_loader.class_map = self.class_map
            return new_loader

        return create_subset_loader(train_indices), create_subset_loader(test_indices)

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

    def convert_mask(self, mask):
        converted_mask = np.copy(mask)
        for orig_value, group_id in self.class_map.items():
            converted_mask[mask == orig_value] = group_id
        return converted_mask

    def display_random_images_with_masks(self, num_examples=5):
        num_examples = min(num_examples, len(self.images))
        random_indices = random.sample(range(len(self.images)), num_examples)
        for idx in random_indices:
            image = self.images[idx]
            mask = self.masks[idx]
            converted_mask = self.convert_mask(mask)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
            axes[0].set_title('Image')
            axes[0].axis('off')
            axes[1].imshow(converted_mask, cmap='gray')
            axes[1].set_title('Converted Mask')
            axes[1].axis('off')
            plt.show()

    def get_image(self, idx):
        if idx < 0 or idx >= len(self.selected_indices):
            raise IndexError("Index out of bounds")
        actual_idx = self.selected_indices[idx]
        image = self.images[actual_idx]
        mask = self.masks[actual_idx]
        converted_mask = self.convert_mask(mask)
        image_float = image.astype(np.float32) / 255.0
        return image_float, converted_mask

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        return self.get_image(idx)

    def __iter__(self):
        for idx in range(len(self)):
            yield self.get_image(idx)
